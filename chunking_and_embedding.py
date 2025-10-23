# Initialise pinecone db

import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import CharacterTextSplitter
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
import time
import json 

load_dotenv()

if not os.getenv("PINECONE_API_KEY"):
    raise ValueError("PINECONE_API_KEY environment variable not set.")

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

# Create index
index_name = "common-index-dense"

embedding_configs = {
    "text-embedding-3-small": {
        "dimensions": 1536,
        "max_retries": 5,
    },
    "all-mpnet-base-v2": {
        "dimensions": 768,
        "max_retries": 5,
    }
}

SYSTEM_PROMPT = (
    "You are an expert text analyst. Your task is to read the provided text {text}"
    "and perform two actions:\n"
    "1. Generate a concise summary of the text, no more than 2-3 lines long.\n"
    "2. Extract a list of exactly 10 relevant keywords or 'labels' "
    "(can be at max 3 words).\n"
    "You must return your analysis in the specified JSON format."
)

# 2. Define the JSON Schema for the structured output
RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "summary": {
            "type": "STRING",
            "description": "A concise 2-3 line summary of the text."
        },
        "labels": {
            "type": "ARRAY",
            "description": "A list of exactly 10 labels.",
            "items": { "type": "STRING" }
        }
    },
    "required": ["summary", "labels"]
}

# Convert above initialisation to a method called init_pinecone() that returns the index object
def init_pinecone():
    pc = Pinecone(api_key=pinecone_api_key)
    if not pc.has_index(index_name):
        pc.create_index(
            name = index_name,
            vector_type = "dense",
            dimension = embedding_configs["all-mpnet-base-v2"]["dimensions"],
            metric = "cosine",
            spec = ServerlessSpec(
                cloud = "aws",
                region = "us-east-1"
            ),
            deletion_protection = "disabled",
            tags = {
                "environment": "development"
            }
        )
    index = pc.Index(index_name)
    print("Pinecone client and index initialized.")
    return index

def load_embedding_model(model_name):
    model = SentenceTransformer(model_name, trust_remote_code=True)
    print(f"Embedding model {model_name} loaded.")
    return model


def get_summary_and_label_metadata(text, system_prompt=SYSTEM_PROMPT, response_schema=RESPONSE_SCHEMA):
    
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=gemini_api_key)

    generation_config_params = {}
    generation_config_params["response_mime_type"] = "application/json"
    generation_config_params["response_schema"] = RESPONSE_SCHEMA

    generation_config = GenerationConfig(**generation_config_params)
    gemini_model = genai.GenerativeModel(model_name="gemini-2.5-pro")

    last_error = None
    max_retries = 5
    retry_delay = 2

    qa_generation_prompt = system_prompt.format(
            text=text
        )

    for attempt in range(max_retries):
        try:
            response = gemini_model.generate_content(
                contents=qa_generation_prompt,
                generation_config=generation_config
            )
            return json.loads(response.text)

        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                print(f"LLM call attempt {attempt + 1} failed: {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"All {max_retries} LLM call attempts failed. Last error: {e}")
    
    return {"summary": "", "labels": ["Indian Constitution Rights", "All"]}

# TODO: Store chunk text data in a NoSQL DB like PostgreSQL or use ElasticSearch 

PINECONE_BATCH_SIZE = 100  # Number of vectors to upsert in each batch

def perform_character_split(chunk_size=1000, chunk_overlap=200):
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator='',
        strip_whitespace=False
    )
    text_splitter


def chunk_file_text(filepath, chunk_size=500, chunk_overlap=150):
    """Reads a text file and chunks it using CharacterTextSplitter."""
    print(f"Reading and chunking file: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        if not text.strip():
            print("Skipping empty file content.")
            return []

        text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            strip_whitespace=False
        )
        chunks = text_splitter.split_text(text)

        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        print(f"Created {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        print(f"*** ERROR reading/chunking file {os.path.basename(filepath)}: {e} ***")
        return []

def upsert_batch_to_pinecone(pinecone_index, batch, context_msg=""):
    """Upserts a batch of vectors to Pinecone and handles errors"""
    if not batch:
        return 0

    num_vectors = len(batch)
    print(f"Upserting {context_msg} batch of {num_vectors} vectors")
    try:
        pinecone_index.upsert(vectors=batch)
        print(f"Batch upserted successfully.")
        return num_vectors
    except Exception as upsert_err:
        print(f"*** ERROR during Pinecone upsert: {upsert_err} ***")
        return 0

# {
#     "filename": "example.txt",
#     "summary": "A short summary of the document",
#     "labels": ["legal", "constitution", "rights"],
#     "source": "extracted_data/pdf_extracted/example.txt"
# }

def embed_and_upsert_chunks(chunks, embed_model, pinecone_index, filename, root_dir, data_dir_base):
    """Generates embeddings for chunks and upserts them to Pinecone."""
    if not chunks:
        return 0

    total_vectors_upserted_for_file = 0
    pinecone_batch = []

    try:
        print(f"Generating embeddings for {len(chunks)} chunks from {filename}...")
        embeddings = embed_model.encode(chunks, batch_size=32, show_progress_bar=True)
        print("Embeddings generated.")

        print("Preparing and upserting vectors to Pinecone...")
        print(root_dir)
        for i, embedding in enumerate(embeddings):
            vector_id = f"{os.path.splitext(filename)[0]}_chunk_{i}"
            response_data = get_summary_and_label_metadata(chunks[i], SYSTEM_PROMPT, RESPONSE_SCHEMA)
            labels = response_data.get("labels", [])
            description = response_data.get("summary", "")
            metadata = {
                "filename": filename,
                "source": os.path.relpath(root_dir, data_dir_base).replace("\\", "/"),
                "summary": description,
                "labels": labels,
            }

            pinecone_batch.append((vector_id, embedding.tolist(), metadata))

            if len(pinecone_batch) >= PINECONE_BATCH_SIZE:
                upserted_count = upsert_batch_to_pinecone(pinecone_index, pinecone_batch, context_msg=f"batch #{total_vectors_upserted_for_file // PINECONE_BATCH_SIZE + 1} for {filename}")
                total_vectors_upserted_for_file += upserted_count
                pinecone_batch = []

        # Upsert any remaining vectors in the last batch for this file using the helper function
        if pinecone_batch:
            upserted_count = upsert_batch_to_pinecone(pinecone_index, pinecone_batch, context_msg=f"final batch for {filename}")
            total_vectors_upserted_for_file += upserted_count
            pinecone_batch = []

        print(f"Finished upserting {total_vectors_upserted_for_file} vectors for {filename}.")
        return total_vectors_upserted_for_file

    except Exception as e:
        print(f"*** ERROR during embedding/upserting for {filename}: {e} ***")
        return 0


def process_directory_and_index(data_dir, pinecone_index, embed_model):
    """Walks directory, chunks files, embeds, and upserts to Pinecone."""
    files_processed = 0
    total_vectors_upserted_all_files = 0

    for root, _, files in os.walk(data_dir):
        files.sort() # Ensure consistent order
        for filename in files:
            if filename.endswith(".txt"):
                filepath = os.path.join(root, filename)
                print(f"\nProcessing file: {filepath}")
                files_processed += 1

                # Step 1: Chunk the file
                chunks = chunk_file_text(filepath)

                # Step 2: Embed and Upsert the chunks
                vectors_upserted = embed_and_upsert_chunks(
                    chunks,
                    embed_model,
                    pinecone_index,
                    filename,
                    root,
                    data_dir
                )
                total_vectors_upserted_all_files += vectors_upserted

    print("\n--- Processing Complete ---")
    print(f"Files processed: {files_processed}")
    print(f"Total vectors upserted across all files: {total_vectors_upserted_all_files}")



if __name__ == "__main__":
    index = init_pinecone()
    model = load_embedding_model("sentence-transformers/all-mpnet-base-v2")


    # Chunk Level 1: Character splitting
    # perform_character_split()

    process_directory_and_index(
        data_dir="cleaned_data",
        pinecone_index=index,
        embed_model=model
    )