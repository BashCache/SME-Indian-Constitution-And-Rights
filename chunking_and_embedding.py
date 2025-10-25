# Initialise pinecone db

import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
import time
import json
import re
from langchain_openai import OpenAIEmbeddings
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

if not os.getenv("PINECONE_API_KEY"):
    raise ValueError("PINECONE_API_KEY environment variable not set.")

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable not set.")

openai_api_key = os.getenv("OPENAI_API_KEY")
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

def chunk_file_text_by_character_splitting(filepath, chunk_size=500, chunk_overlap=150):
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
            separators=["\n\n", "\n", ". ", "? ", "! ", ",", " ", ""],
            strip_whitespace=False
        )
        chunks = text_splitter.create_documents(text)

        docs = text_splitter.create_documents([text])
        print(docs[0])
        # Extract the text content from each Document object
        # No need to strip again if strip_whitespace=True in splitter
        cleaned_chunks = [doc.page_content for doc in docs if doc.page_content]
        print(f"Created {len(cleaned_chunks)} chunks.")
        return cleaned_chunks
    except Exception as e:
        print(f"*** ERROR reading/chunking file {os.path.basename(filepath)}: {e} ***")
        return []

def chunk_file_text_by_recursive_character_splitting(filepath, chunk_size=500, chunk_overlap=150):
    """Reads a text file and chunks it using RecursiveCharacterTextSplitter."""
    print(f"Reading and chunking file: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        if not text.strip():
            print("Skipping empty file content.")
            return []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strip_whitespace=False
        )

        docs = text_splitter.create_documents([text])
        print(docs[0])
        cleaned_chunks = [doc.page_content for doc in docs if doc.page_content] # Check if page_content is non-empty
        print(f"Created {len(cleaned_chunks)} chunks.")
        return cleaned_chunks
    except Exception as e:
        print(f"*** ERROR reading/chunking file {os.path.basename(filepath)}: {e} ***")
        return []

def combine_sentences(sentences, buffer_size=1):
    for i in range(len(sentences)):

        combined_sentence = ''

        for j in range(i - buffer_size, i):
            if j >= 0:
                combined_sentence += sentences[j]['sentence'] + ' '

        combined_sentence += sentences[i]['sentence']

        for j in range(i + 1, i + 1 + buffer_size):
            if j < len(sentences):
                combined_sentence += ' ' + sentences[j]['sentence']

        sentences[i]['combined_sentence'] = combined_sentence

    return sentences

def calculate_cosine_distances(sentences):
    distances = []
    for i in range(len(sentences) - 1):
        embedding_current = sentences[i]['combined_sentence_embedding']
        embedding_next = sentences[i + 1]['combined_sentence_embedding']
        
        similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]
        
        distance = 1 - similarity
        distances.append(distance)

        sentences[i]['distance_to_next'] = distance

    return distances, sentences

def chunk_file_text_by_semantic_splitting(filepath, chunk_size=500, chunk_overlap=150):
    print(f"Reading and chunking file: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        if not text.strip():
            print("Skipping empty file content.")
            return []
        
        # Splitting the essay on '.', '?', and '!'
        single_sentences_list = re.split(r'(?<=[.?!])\s+', text)
        print (f"{len(single_sentences_list)} senteneces were found")

        sentences = [{'sentence': x, 'index' : i} for i, x in enumerate(single_sentences_list)]
        print(sentences[0:3])  # Print first 3 sentences for verification

        sentences = combine_sentences(sentences[0:5])
        print(f"Combined sentences sample: {sentences[0]['combined_sentence']}")

        # oaiembeds = OpenAIEmbeddings()
        embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
        embeddings = embeddings_model.embed_documents([x['combined_sentence'] for x in sentences])

        client = genai.Client()

        embeddings = client.models.embed_content(
        model="gemini-embedding-001",
        content=[x['combined_sentence'] for x in sentences]
        )

        for i, sentence in enumerate(sentences):
            sentence['combined_sentence_embedding'] = embeddings[i]

        distances, sentences = calculate_cosine_distances(sentences)
        print(f"Calculated {len(distances)} cosine distances between sentences.")

        plt.plot(distances)

        y_upper_bound = .2
        plt.ylim(0, y_upper_bound)
        plt.xlim(0, len(distances))

        # We need to get the distance threshold that we'll consider an outlier
        # We'll use numpy .percentile() for this
        breakpoint_percentile_threshold = 95
        breakpoint_distance_threshold = np.percentile(distances, breakpoint_percentile_threshold) # If you want more chunks, lower the percentile cutoff
        plt.axhline(y=breakpoint_distance_threshold, color='r', linestyle='-');

        # Then we'll see how many distances are actually above this one
        num_distances_above_theshold = len([x for x in distances if x > breakpoint_distance_threshold]) # The amount of distances above your threshold
        plt.text(x=(len(distances)*.01), y=y_upper_bound/50, s=f"{num_distances_above_theshold + 1} Chunks");

        # Then we'll get the index of the distances that are above the threshold. This will tell us where we should split our text
        indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold] # The indices of those breakpoints on your list

        # Start of the shading and text
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        for i, breakpoint_index in enumerate(indices_above_thresh):
            start_index = 0 if i == 0 else indices_above_thresh[i - 1]
            end_index = breakpoint_index if i < len(indices_above_thresh) - 1 else len(distances)

            plt.axvspan(start_index, end_index, facecolor=colors[i % len(colors)], alpha=0.25)
            plt.text(x=np.average([start_index, end_index]),
                    y=breakpoint_distance_threshold + (y_upper_bound)/ 20,
                    s=f"Chunk #{i}", horizontalalignment='center',
                    rotation='vertical')

        # # Additional step to shade from the last breakpoint to the end of the dataset
        if indices_above_thresh:
            last_breakpoint = indices_above_thresh[-1]
            if last_breakpoint < len(distances):
                plt.axvspan(last_breakpoint, len(distances), facecolor=colors[len(indices_above_thresh) % len(colors)], alpha=0.25)
                plt.text(x=np.average([last_breakpoint, len(distances)]),
                        y=breakpoint_distance_threshold + (y_upper_bound)/ 20,
                        s=f"Chunk #{i+1}",
                        rotation='vertical')

        plt.title("NCERT Chapter 1 Chunks Based On Embedding Breakpoints")
        plt.xlabel("Index of sentences in book (Sentence Position)")
        plt.ylabel("Cosine distance between sequential sentences")
        plt.show()

        start_index = 0
        chunks = []

        # Iterate through the breakpoints to slice the sentences
        for index in indices_above_thresh:
            end_index = index
            group = sentences[start_index:end_index + 1]
            combined_text = ' '.join([d['sentence'] for d in group])
            chunks.append(combined_text)
            start_index = index + 1

        if start_index < len(sentences):
            combined_text = ' '.join([d['sentence'] for d in sentences[start_index:]])
            chunks.append(combined_text)


        for i, chunk in enumerate(chunks[:2]):
            buffer = 200
            
            print (f"Chunk #{i}")
            print (chunk[:buffer].strip())
            print ("...")
            print (chunk[-buffer:].strip())
            print ("\n")
        
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


def process_directory_and_index(data_dir, pinecone_index, embed_model, chunking_mode="character"):
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
                if chunking_mode == "character":
                    chunks = chunk_file_text_by_character_splitting(filepath)
                elif chunking_mode == "recursive_character":
                    chunks = chunk_file_text_by_recursive_character_splitting(filepath)
                elif chunking_mode == "semantic":
                    chunks = chunk_file_text_by_semantic_splitting(filepath)

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
        data_dir="cleaned_data\\books_extracted",
        pinecone_index=index,
        embed_model=model,
        chunking_mode="recursive_character"
    )

    print("--- Testing CharacterTextSplitter (using create_documents) ---")
    char_chunks = process_directory_and_index(data_dir="cleaned_data\\books_extracted\\keps101_extracted.txt", pinecone_index=index, embed_model=model, chunking_mode="character")
    print(f"Length of char_chunks: {len(char_chunks)}")

    print("\n--- Testing RecursiveCharacterTextSplitter (using create_documents) ---")
    rec_chunks = process_directory_and_index(data_dir="cleaned_data\\books_extracted\\keps101_extracted.txt", pinecone_index=index, embed_model=model, chunking_mode="recursive_character")
    print(f"Length of rec_chunks: {len(rec_chunks)}")

    semantic_chunks = process_directory_and_index(data_dir="cleaned_data", pinecone_index=index, embed_model=model, chunking_mode="semantic")