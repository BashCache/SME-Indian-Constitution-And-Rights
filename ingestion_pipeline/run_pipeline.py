# """
# (Task 2.1 - FINAL-ROBUST-SIMPLE High-Performance Ingestion)
# This pipeline is now fixed to be simple and robust.

# CHANGES:
# 1.  We have GIVEN UP the complex 'chunk_text_by_semantic_splitting'.
#     It was causing a massive lock pile-up and gridlock.
# 2.  We are now using the standard, fast, CPU-only
#     `RecursiveCharacterTextSplitter`. This removes the bottleneck.
# 3.  The worker flow is now simple:
#     1. Chunk (CPU, fast, no lock)
#     2. Embed (GPU, locked, one-at-a-time)
#     3. Get Metadata (CPU/Network, parallel)
#     4. Upsert (Network, parallel)
# """

# import os
# # ... (rest of imports)
# import re
# import time
# import json
# import glob
# import numpy as np
# from pinecone import Pinecone, ServerlessSpec
# from dotenv import load_dotenv
# import google.generativeai as genai
# from google.generativeai.types import GenerationConfig
# from tqdm import tqdm
# import subprocess
# from sentence_transformers import SentenceTransformer
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# import argparse
# import concurrent.futures
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import multiprocessing
# import threading 
# import logging 

# # --- Performance & Safety Configuration ---
# MAX_CPUS = os.cpu_count() or 32 # For I/O tasks (Gemini)
# TIMEOUT_SECONDS = 28800 # 8 hours
# GPU_BATCH_SIZE = 128 # Safe batch size for GPU
# MAX_CHUNK_SIZE = 10000 
# FUTURE_TIMEOUT_SECONDS = 600
# # --- !! ---------------- !! ---


# # --- Setup Logging ---
# def setup_logging():
#     """Configures logging to file and console."""
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s',
#         filename='ingestion.log', # Log to this file
#         filemode='w' # Overwrite old log each run
#     )
#     # Also log to console
#     console_handler = logging.StreamHandler()
#     console_handler.setLevel(logging.INFO)
#     console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
#     logging.getLogger().addHandler(console_handler)
# # --- !! ---------------- !! ---


# # --- Argument Parser ---
# parser = argparse.ArgumentParser(description="Parallel Ingestion Pipeline for A/B Testing.")
# parser.add_argument(
#     '--model_name', 
#     type=str, 
#     required=True, 
#     help="HuggingFace model name (e.g., 'nlpaueb/legal-bert-base-uncased')"
# )
# parser.add_argument(
#     '--index_name', 
#     type=str, 
#     required=True, 
#     help="The Pinecone index name to create/use (e.g., 'index-legal-bert')"
# )
# args = parser.parse_args()

# # --- Configuration ---
# load_dotenv()
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# if not PINECONE_API_KEY:
#     raise ValueError("PINECONE_API_KEY environment variable not set.")
# if not GEMINI_API_KEY:
#     raise ValueError("GEMINI_API_KEY environment variable not set (needed for metadata).")

# # --- Model & Index Config (From Arguments) ---
# MODEL_NAME = args.model_name
# INDEX_NAME = args.index_name
# EMBEDDING_DIMENSION = 768 
# PINECONE_BATCH_SIZE = 100 


# # --- Paths (Corrected) ---
# BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# CLEANED_DATA_DIR = os.path.join(BASE_DIR, 'cleaned_data') 

# # --- Globals for CPU workers (Gemini/Pinecone) ---
# gemini_model_tls = threading.local()
# pinecone_index_tls = threading.local()

# # --- Globals for Single GPU Model ---
# local_embed_model = None
# embedding_lock = threading.Lock()

# # --- Recursive Text Splitter (Fallback) ---
# recursive_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=MAX_CHUNK_SIZE, # Use the same max size
#     chunk_overlap=200 # Add some overlap for sub-chunks
# )
# # --- !! ---------------- !! ---

# def load_all_cleaned_text():
#     """
#     Walks the `cleaned_data` directory and reads all .txt files.
#     """
#     logging.info(f"Loading all .txt files from {CLEANED_DATA_DIR}...")
#     documents = []
    
#     for filepath in glob.glob(os.path.join(CLEANED_DATA_DIR, '**/*.txt'), recursive=True):
#         try:
#             with open(filepath, 'r', encoding='utf-8') as f:
#                 text = f.read()
#             if not text.strip():
#                 logging.warning(f"Skipping empty file: {os.path.basename(filepath)}")
#                 continue
#             text = re.sub(r'\s+', ' ', text).strip().lower()
#             documents.append({
#                 "id": os.path.relpath(filepath, CLEANED_DATA_DIR).replace("\\", "/").replace(".txt", ""),
#                 "text": text,
#                 "source": os.path.relpath(filepath, CLEANED_DATA_DIR).replace("\\", "/")
#             })
#         except Exception as e:
#             logging.error(f"*** ERROR reading {filepath}: {e} ***")
            
#     logging.info(f"Loaded {len(documents)} text documents.")
#     return documents

# # --- Thread-Safe Embedding Function ---
# def get_embeddings(texts):
#     """
#     Gets embeddings using the GLOBAL model.
#     This function is now THREAD-SAFE using a lock.
#     """
#     global local_embed_model, embedding_lock
#     if local_embed_model is None:
#         logging.error("Global embedding model not initialized.")
#         return [[] for _ in texts]
    
#     all_embeddings = []
#     try:
#         # Only one thread can use the GPU at a time.
#         with embedding_lock:
#             for i in range(0, len(texts), GPU_BATCH_SIZE):
#                 batch = texts[i : i + GPU_BATCH_SIZE]
                
#                 embs = local_embed_model.encode(
#                     batch, 
#                     show_progress_bar=False,
#                     batch_size=GPU_BATCH_SIZE
#                 )
#                 all_embeddings.extend([emb.tolist() for emb in embs])
            
#         return all_embeddings
        
#     except Exception as e:
#         # This is the CUDA OOM error
#         logging.error(f"FATAL CUDA Error: {e}. This job failed.")
#         return [[] for _ in texts]


# # --- CPU (Thread) Worker Initialization ---
# def init_cpu_worker(index_name_arg):
#     """
#     This function is called ONCE by each of the 32 CPU threads.
#     It initializes thread-local connections to Gemini and Pinecone.
#     """
#     global gemini_model_tls, pinecone_index_tls
    
#     try:
#         # 1. Configure Gemini
#         genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
#         RESPONSE_SCHEMA = {
#             "type": "OBJECT",
#             "properties": {
#                 "summary": {"type": "STRING", "description": "A concise 2-3 line summary of the text."},
#                 "labels": {"type": "ARRAY", "description": "A list of exactly 10 labels.", "items": {"type": "STRING"}}
#             },
#             "required": ["summary", "labels"]
#         }
#         generation_config = GenerationConfig(
#             response_mime_type="application/json",
#             response_schema=RESPONSE_SCHEMA
#         )
#         gemini_model_tls.model = genai.GenerativeModel(
#             model_name="gemini-2.5-pro",
#             generation_config=generation_config
#         )
        
#         # 2. Configure Pinecone
#         pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
#         pinecone_index_tls.index = pc.Index(index_name_arg)
#     except Exception as e:
#         logging.error(f"CPU Worker {threading.current_thread().name}: FAILED to initialize: {e}")


# # --- Metadata Generation (Gemini API) ---
# def get_summary_and_label_metadata(text):
#     """
#     Calls the Gemini API using the CPU THREAD'S local model.
#     """
#     global gemini_model_tls
#     if not hasattr(gemini_model_tls, 'model'):
#          logging.error(f"CPU Worker: Gemini Model not initialized.")
#          return {"summary": "No summary generated.", "labels": ["Error"]}

#     SYSTEM_PROMPT = (
#         "You are an expert text analyst. Your task is to read the provided text {text} "
#         "and perform two actions:\n"
#         "1. Generate a concise summary of the text, no more than 2-3 lines long.\n"
#         "2. Extract a list of exactly 10 relevant keywords or 'labels' "
#         "(can be at max 3 words).\n"
#         "You must return your analysis in the specified JSON format."
#     )
    
#     prompt_text = text[:MAX_CHUNK_SIZE] 
#     prompt = SYSTEM_PROMPT.format(text=prompt_text)
#     max_retries = 5
#     retry_delay = 5

#     for attempt in range(max_retries):
#         try:
#             response = gemini_model_tls.model.generate_content(contents=prompt)
#             return json.loads(response.text)
#         except Exception as e:
#             if "quota" in str(e).lower() or "leaked" in str(e).lower():
#                  logging.error(f"CPU Worker: GEMINI API KEY ERROR: {e}. Sleeping 60s...")
#                  time.sleep(60) # Sleep to avoid spamming
#             if attempt < max_retries - 1:
#                 time.sleep(retry_delay * (attempt + 1))
#             else:
#                 logging.error(f"CPU Worker: All LLM call attempts failed. {e}")
    
#     return {"summary": "No summary generated.", "labels": ["Error"]}

# # --- Pinecone Upsert Logic ---
# def init_pinecone_index_main(index_name):
#     """
#     The MAIN process calls this ONCE to clear the index.
#     """
#     pc = Pinecone(api_key=PINECONE_API_KEY)
    
#     if index_name not in pc.list_indexes().names():
#         logging.info(f"Creating index {index_name}...")
#         pc.create_index(
#             name=index_name,
#             dimension=EMBEDDING_DIMENSION,
#             metric="cosine",
#             spec=ServerlessSpec(cloud="aws", region="us-east-1")
#         )
#     else:
#         logging.info(f"Index {index_name} already exists. Checking vector count...")
#         index = pc.Index(index_name)
#         stats = index.describe_index_stats()
        
#         if stats.get('total_vector_count', 0) > 0:
#             logging.info(f"Clearing {stats['total_vector_count']} existing vectors for fresh build...")
#             try:
#                 index.delete(delete_all=True)
#                 logging.info("Index cleared.")
#             except Exception as e:
#                 logging.warning(f"Could not clear index: {e}. Retrying...")
#                 time.sleep(5)
#                 index.delete(delete_all=True)
#                 logging.info("Index cleared on retry.")
#         else:
#             logging.info("Index is already empty. No need to clear.")
    
#     index = pc.Index(index_name)
#     logging.info(f"Pinecone client and index '{index_name}' initialized.")
#     return index

# def upsert_batch_to_pinecone(batch_data):
#     """
#     Upserts a batch of data using the CPU THREAD'S local connection.
#     """
#     global pinecone_index_tls
#     if not batch_data:
#         return 0
#     try:
#         if not hasattr(pinecone_index_tls, 'index'):
#              raise Exception(f"CPU Worker: Pinecone not initialized.")
#         pinecone_index_tls.index.upsert(vectors=batch_data)
#         return len(batch_data)
#     except Exception as e:
#         if "Metadata size" in str(e):
#             logging.error(f"*** PINECONE METADATA ERROR: {e} ***")
#             for i, item in enumerate(batch_data):
#                 logging.error(f"Item {i} metadata size: {len(str(item['metadata']))}")
#                 if len(str(item['metadata'])) > 40000:
#                     logging.error(f"Problem chunk (source): {item['metadata'].get('source')}")
#                     logging.error(f"Problem chunk (text): {item['metadata'].get('text', '')[:200]}...")
#         else:
#             logging.error(f"*** ERROR during Pinecone upsert: {e} ***")
#         return 0

# # --- NEW: CPU (Thread) Worker Function ---
# def process_document_worker(doc):
#     """
#     This is the new worker function that runs on a single CPU THREAD.
#     It processes ONE document from start to finish.
#     """
#     try:
#         # --- STEP 1: Chunk (CPU Only, No Lock) ---
#         chunks = recursive_splitter.split_text(doc["text"])
#         if not chunks:
#             logging.warning(f"No chunks created for {doc['id']}.")
#             return doc['id'], 0, "No chunks"
            
#         # --- STEP 2: Get Embeddings (GPU, Locked) ---
#         embeddings = get_embeddings(chunks)
        
#         valid_data = [(chunks[i], embeddings[i]) for i in range(len(chunks)) if embeddings[i]]
#         if not valid_data:
#             logging.error(f"All embeddings failed for {doc['id']}.")
#             return doc['id'], 0, "All embeddings failed"

#         pinecone_batch = []
        
#         # --- STEP 3: Get Metadata (Gemini API, Parallel) ---
#         for i, (chunk_text, embedding) in enumerate(valid_data):
#             if len(chunk_text) > MAX_CHUNK_SIZE:
#                 logging.warning(f"Chunk from {doc['id']} is STILL too large ({len(chunk_text)}). Truncating.")
#                 chunk_text = chunk_text[:MAX_CHUNK_SIZE]

#             metadata = get_summary_and_label_metadata(chunk_text)
            
#             # --- THIS IS THE FIX ---
#             # Lowercase the labels before saving to match the query
#             if "labels" in metadata and isinstance(metadata["labels"], list):
#                 metadata["labels"] = [label.lower() for label in metadata["labels"]]
#             # --- END OF FIX ---
            
#             metadata["text"] = chunk_text 
#             metadata["source"] = doc["source"]
#             vector_id = f"{doc['id']}-chunk-{i}"
            
#             pinecone_batch.append({
#                 "id": vector_id,
#                 "values": embedding,
#                 "metadata": metadata
#             })

#         # --- STEP 4: Upsert to Pinecone (Network, Parallel) ---
#         upserted_count = upsert_batch_to_pinecone(pinecone_batch)
        
#         return doc['id'], upserted_count, "Success"
        
#     except Exception as e:
#         logging.error(f"*** UNHANDLED ERROR in worker for {doc['id']}: {e} ***")
#         return doc['id'], 0, f"Failed: {e}"


# # --- NEW: Main Parallel Pipeline ---
# def main():
#     global local_embed_model
    
#     setup_logging()
#     start_time = time.time()
    
#     docs_processed = 0
#     total_docs = 0 
#     total_vectors_upserted = 0

#     try:
#         # --- NEW: Load model in main thread ONCE ---
#         logging.info(f"--- LOADING MODEL: {MODEL_NAME} onto cuda:0 ---")
#         local_embed_model = SentenceTransformer(MODEL_NAME, device='cuda:0')
#         logging.info("Model loaded successfully.")
#         # --- !! ---------------- !! ---

#         # 1. Load all the .txt files
#         documents = load_all_cleaned_text()
#         total_docs = len(documents)
#         if not documents:
#             logging.error(f"No documents found in {CLEANED_DATA_DIR}. Exiting.")
#             return

#         # 2. Initialize Pinecone (Main thread clears the index)
#         index = init_pinecone_index_main(INDEX_NAME)
        
#         logging.info(f"--- Starting Parallel Ingestion ({MAX_CPUS} CPU threads, 1 GPU) ---")
        
#         # 3. Create ONE Pool (ThreadPool)
#         with ThreadPoolExecutor(
#             max_workers=MAX_CPUS,
#             initializer=init_cpu_worker,
#             initargs=(INDEX_NAME,)
#         ) as cpu_pool:
            
#             # Submit all jobs to the CPU pool
#             futures = [cpu_pool.submit(process_document_worker, doc) for doc in documents]
            
#             pbar = tqdm(total=total_docs, desc="Processing Documents", unit="doc")
            
#             for i, future in enumerate(as_completed(futures)):
                
#                 if time.time() - start_time > TIMEOUT_SECONDS:
#                     logging.warning(f"\n--- TIMEOUT: 8-hour limit reached. ---")
#                     logging.warning("--- Beginning graceful shutdown... ---")
#                     cpu_pool.shutdown(wait=True, cancel_futures=True) 
#                     logging.info("--- Graceful shutdown complete. ---")
#                     break
                    
#                 try:
#                     # --- NEW: Add timeout to future.result() ---
#                     # This will catch any worker stuck in an infinite loop
#                     doc_id, vector_count, status = future.result(timeout=FUTURE_TIMEOUT_SECONDS)
#                     # --- END OF FIX ---
                    
#                     if status == "Success":
#                         total_vectors_upserted += vector_count
#                     else:
#                         logging.warning(f"Warning: Failed to process {doc_id}. Status: {status}")
                
#                 except concurrent.futures.TimeoutError:
#                     logging.error(f"\n--- CRITICAL: Worker for doc {i} TIMED OUT after {FUTURE_TIMEOUT_SECONDS}s. ---")
#                     logging.error(f"--- This worker is stuck (likely Gemini API) and will be abandoned. ---")
                    
#                 except Exception as e:
#                     logging.error(f"\n--- CRITICAL WORKER ERROR: {e} ---")
                
#                 docs_processed += 1
#                 pbar.update(1)
                
#                 if docs_processed > 1:
#                     elapsed = time.time() - start_time
#                     avg_time_per_doc = elapsed / docs_processed
#                     remaining_docs = total_docs - docs_processed
#                     eta_seconds = remaining_docs * avg_time_per_doc
#                     eta_hours = eta_seconds // 3600
#                     eta_minutes = (eta_seconds % 3600) // 60
#                     pbar.set_postfix_str(f"ETA: {int(eta_hours)}h {int(eta_minutes)}m")
            
#             logging.info("CPU pool shut down.")

#     except KeyboardInterrupt:
#         logging.info("\n--- User interrupted. Shutting down. ---")
    
#     finally:
#         logging.info("\n--- Ingestion Pipeline Complete ---")
#         logging.info(f"Processed {docs_processed} / {total_docs} documents.")
#         logging.info(f"Total vectors upserted into '{INDEX_NAME}': {total_vectors_upserted}")
        
#         try:
#             if 'index' in locals() and index:
#                  final_stats = index.describe_index_stats()
#                  logging.info(final_stats)
#             else:
#                  final_index = init_pinecone_index_main(INDEX_NAME)
#                  final_stats = final_index.describe_index_stats()
#                  logging.info(final_stats)
#         except Exception as e:
#             logging.error(f"Could not get final index stats: {e}")

# if __name__ == "__main__":
#     main()


"""
(Task 2.1 - FINAL-KG-BONUS High-Performance Ingestion)

CHANGES:
1.  Implements the Knowledge Graph (BONUS) requirement.
2.  Adds a new function 'get_entities_from_chunk' which calls
    Gemini to extract a list of entities from each chunk.
3.  These entities are saved to a new 'entities' field in
    the Pinecone metadata.
"""

import os
import re
import time
import json
import glob
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from typing import List
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from tqdm import tqdm
import subprocess
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import argparse
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import threading 
import logging 

# --- Performance & Safety Configuration ---
MAX_CPUS = os.cpu_count() or 32 # For I/O tasks (Gemini)
TIMEOUT_SECONDS = 28800 # 8 hours
GPU_BATCH_SIZE = 128 # Safe batch size for GPU
MAX_CHUNK_SIZE = 10000 
FUTURE_TIMEOUT_SECONDS = 600
# --- !! ---------------- !! ---

# --- BM25 Configuration ---
BM25_CORPUS_FILE = 'bm25_corpus.jsonl'
bm25_lock = threading.Lock()
# --- !! ---------------- !! ---

# --- Setup Logging ---
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s',
        filename='ingestion.log',
        filemode='w'
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    logging.getLogger().addHandler(console_handler)
# --- !! ---------------- !! ---

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Parallel Ingestion Pipeline for A/B Testing.")
parser.add_argument(
    '--model_name', 
    type=str, 
    required=True, 
    help="HuggingFace model name (e.g., 'nlpaueb/legal-bert-base-uncased')"
)
parser.add_argument(
    '--index_name', 
    type=str, 
    required=True, 
    help="The Pinecone index name to create/use (e.g., 'index-legal-bert')"
)
args = parser.parse_args()

# --- Configuration ---
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not PINECONE_API_KEY or not GEMINI_API_KEY:
    raise ValueError("PINECONE_API_KEY or GEMINI_API_KEY not set.")

MODEL_NAME = args.model_name
INDEX_NAME = args.index_name
EMBEDDING_DIMENSION = 768 
PINECONE_BATCH_SIZE = 100 

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CLEANED_DATA_DIR = os.path.join(BASE_DIR, 'cleaned_data') 

# --- Globals ---
gemini_model_tls = threading.local()
pinecone_index_tls = threading.local()
local_embed_model = None
embedding_lock = threading.Lock()
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=MAX_CHUNK_SIZE,
    chunk_overlap=200
)
# --- !! ---------------- !! ---

def load_all_cleaned_text():
    logging.info(f"Loading all .txt files from {CLEANED_DATA_DIR}...")
    documents = []
    for filepath in glob.glob(os.path.join(CLEANED_DATA_DIR, '**/*.txt'), recursive=True):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            if not text.strip():
                logging.warning(f"Skipping empty file: {os.path.basename(filepath)}")
                continue
            # We keep the original case for BM25 and entity extraction
            text = re.sub(r'\s+', ' ', text).strip()
            documents.append({
                "id": os.path.relpath(filepath, CLEANED_DATA_DIR).replace("\\", "/").replace(".txt", ""),
                "text": text,
                "source": os.path.relpath(filepath, CLEANED_DATA_DIR).replace("\\", "/")
            })
        except Exception as e:
            logging.error(f"*** ERROR reading {filepath}: {e} ***")
    logging.info(f"Loaded {len(documents)} text documents.")
    return documents

def get_embeddings(texts):
    global local_embed_model, embedding_lock
    if local_embed_model is None:
        logging.error("Global embedding model not initialized.")
        return [[] for _ in texts]
    
    all_embeddings = []
    try:
        with embedding_lock:
            for i in range(0, len(texts), GPU_BATCH_SIZE):
                batch = texts[i : i + GPU_BATCH_SIZE]
                embs = local_embed_model.encode(
                    batch, 
                    show_progress_bar=False,
                    batch_size=GPU_BATCH_SIZE
                )
                all_embeddings.extend([emb.tolist() for emb in embs])
        return all_embeddings
    except Exception as e:
        logging.error(f"FATAL CUDA Error: {e}. This job failed.")
        return [[] for _ in texts]

# --- CPU (Thread) Worker Initialization ---
def init_cpu_worker(index_name_arg):
    """
    This function is called ONCE by each of the CPU threads.
    It initializes thread-local connections to Gemini and Pinecone.
    """
    global gemini_model_tls, pinecone_index_tls
    
    try:
        # 1. Configure Gemini (for Summaries/Labels)
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        SUMMARY_SCHEMA = {
            "type": "OBJECT",
            "properties": {
                "summary": {"type": "STRING", "description": "A concise 2-3 line summary of the text."},
                "labels": {"type": "ARRAY", "description": "A list of exactly 10 labels.", "items": {"type": "STRING"}}
            },
            "required": ["summary", "labels"]
        }
        gemini_model_tls.summary_generator = genai.GenerativeModel(
            model_name="gemini-2.5-pro",
            generation_config=GenerationConfig(
                response_mime_type="application/json",
                response_schema=SUMMARY_SCHEMA
            )
        )
        
        # 2. Configure Gemini (for Entity Extraction)
        ENTITY_SCHEMA = {
            "type": "OBJECT",
            "properties": {
                "entities": {
                    "type": "ARRAY", 
                    "description": "A list of all key entities found in the text.",
                    "items": {"type": "STRING"}
                }
            },
            "required": ["entities"]
        }
        gemini_model_tls.entity_extractor = genai.GenerativeModel(
            model_name="gemini-2.5-pro",
            generation_config=GenerationConfig(
                response_mime_type="application/json",
                response_schema=ENTITY_SCHEMA
            )
        )
        
        # 3. Configure Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        pinecone_index_tls.index = pc.Index(index_name_arg)
    except Exception as e:
        logging.error(f"CPU Worker {threading.current_thread().name}: FAILED to initialize: {e}")


# --- Metadata Generation (Gemini API) ---
def get_summary_and_label_metadata(text):
    global gemini_model_tls
    if not hasattr(gemini_model_tls, 'summary_generator'):
         logging.error(f"CPU Worker: Summary Generator not initialized.")
         return {"summary": "No summary generated.", "labels": ["Error"]}

    SYSTEM_PROMPT = (
        "You are an expert text analyst. Your task is to read the provided text {text} "
        "and perform two actions:\n"
        "1. Generate a concise summary of the text, no more than 2-3 lines long.\n"
        "2. Extract a list of exactly 10 relevant keywords or 'labels' "
        "(can be at max 3 words).\n"
        "You must return your analysis in the specified JSON format."
    )
    
    prompt_text = text[:MAX_CHUNK_SIZE] 
    prompt = SYSTEM_PROMPT.format(text=prompt_text)
    max_retries = 5
    retry_delay = 5

    for attempt in range(max_retries):
        try:
            response = gemini_model_tls.summary_generator.generate_content(contents=prompt)
            return json.loads(response.text)
        except Exception as e:
            if "quota" in str(e).lower() or "leaked" in str(e).lower():
                 logging.error(f"CPU Worker: GEMINI API KEY ERROR: {e}. Sleeping 60s...")
                 time.sleep(60)
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
            else:
                logging.error(f"CPU Worker: All LLM (summary) call attempts failed. {e}")
    
    return {"summary": "No summary generated.", "labels": ["Error"]}

# --- NEW: Knowledge Graph Entity Extractor ---
def get_entities_from_chunk(text: str) -> List[str]:
    """
    Calls Gemini to extract key entities (people, laws, concepts)
    to build the 'Knowledge Graph' metadata.
    """
    global gemini_model_tls
    if not hasattr(gemini_model_tls, 'entity_extractor'):
         logging.error(f"CPU Worker: Entity Extractor not initialized.")
         return []

    SYSTEM_PROMPT = (
        "You are a legal and historical entity extractor. Your task is to read the "
        "provided text and extract a list of all key proper nouns and legal concepts. \n"
        "Examples: ['Article 14', 'Right to Equality', 'Dr. B.R. Ambedkar', 'Kesavananda Bharati v. State of Kerala', 'Fundamental Rights', 'Preamble'].\n"
        "Return your response as a JSON object with a single key 'entities'.\n"
        "Text: {text}"
    )
    
    prompt_text = text[:MAX_CHUNK_SIZE]
    prompt = SYSTEM_PROMPT.format(text=prompt_text)
    max_retries = 5
    retry_delay = 5

    for attempt in range(max_retries):
        try:
            response = gemini_model_tls.entity_extractor.generate_content(contents=prompt)
            # Normalize to lowercase for matching
            data = json.loads(response.text)
            return [str(entity).lower() for entity in data.get("entities", [])]
        except Exception as e:
            if "quota" in str(e).lower() or "leaked" in str(e).lower():
                 logging.error(f"CPU Worker: GEMINI API KEY ERROR: {e}. Sleeping 60s...")
                 time.sleep(60)
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
            else:
                logging.error(f"CPU Worker: All LLM (entity) call attempts failed. {e}")
    
    return [] # Return empty list on failure
# --- !! ---------------- !! ---


# --- Pinecone Upsert Logic ---
def init_pinecone_index_main(index_name):
    """
    The MAIN process calls this ONCE to clear the index.
    """
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    if index_name not in pc.list_indexes().names():
        logging.info(f"Creating index {index_name}...")
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    else:
        logging.info(f"Index {index_name} already exists. Checking vector count...")
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        
        if stats.get('total_vector_count', 0) > 0:
            logging.info(f"Clearing {stats['total_vector_count']} existing vectors for fresh build...")
            try:
                index.delete(delete_all=True)
                logging.info("Index cleared.")
            except Exception as e:
                logging.warning(f"Could not clear index: {e}. Retrying...")
                time.sleep(5)
                index.delete(delete_all=True)
                logging.info("Index cleared on retry.")
        else:
            logging.info("Index is already empty. No need to clear.")
    
    index = pc.Index(index_name)
    logging.info(f"Pinecone client and index '{index_name}' initialized.")
    return index

def upsert_batch_to_pinecone(batch_data):
    """
    Upserts a batch of data using the CPU THREAD'S local connection.
    """
    global pinecone_index_tls
    if not batch_data:
        return 0
    try:
        if not hasattr(pinecone_index_tls, 'index'):
             raise Exception(f"CPU Worker: Pinecone not initialized.")
        pinecone_index_tls.index.upsert(vectors=batch_data)
        return len(batch_data)
    except Exception as e:
        if "Metadata size" in str(e):
            logging.error(f"*** PINECONE METADATA ERROR: {e} ***")
            for i, item in enumerate(batch_data):
                logging.error(f"Item {i} metadata size: {len(str(item['metadata']))}")
                if len(str(item['metadata'])) > 40000:
                    logging.error(f"Problem chunk (source): {item['metadata'].get('source')}")
                    logging.error(f"Problem chunk (text): {item['metadata'].get('text', '')[:200]}...")
        else:
            logging.error(f"*** ERROR during Pinecone upsert: {e} ***")
        return 0

# --- Function to save chunks for BM25 ---
def save_chunk_for_bm25(chunk_id, chunk_text):
    """
    Appends a chunk to the BM25 corpus file in a thread-safe way.
    """
    global bm25_lock
    try:
        with bm25_lock:
            with open(BM25_CORPUS_FILE, 'a', encoding='utf-8') as f:
                # We save the raw text for BM25
                f.write(json.dumps({"id": chunk_id, "text": chunk_text}) + "\n")
    except Exception as e:
        logging.error(f"Failed to write chunk {chunk_id} to BM25 corpus: {e}")

# --- CPU (Thread) Worker Function ---
def process_document_worker(doc):
    """
    This is the new worker function that runs on a single CPU THREAD.
    It processes ONE document from start to finish.
    """
    try:
        # --- STEP 1: Chunk (CPU Only, No Lock) ---
        chunks = recursive_splitter.split_text(doc["text"])
        if not chunks:
            logging.warning(f"No chunks created for {doc['id']}.")
            return doc['id'], 0, "No chunks"
            
        # --- STEP 2: Get Embeddings (GPU, Locked) ---
        embeddings = get_embeddings(chunks)
        
        valid_data = [(chunks[i], embeddings[i]) for i in range(len(chunks)) if embeddings[i]]
        if not valid_data:
            logging.error(f"All embeddings failed for {doc['id']}.")
            return doc['id'], 0, "All embeddings failed"

        pinecone_batch = []
        
        for i, (chunk_text, embedding) in enumerate(valid_data):
            if len(chunk_text) > MAX_CHUNK_SIZE:
                logging.warning(f"Chunk from {doc['id']} is STILL too large ({len(chunk_text)}). Truncating.")
                chunk_text = chunk_text[:MAX_CHUNK_SIZE]

            # --- STEP 3: Get Metadata (Gemini API, Parallel) ---
            metadata = get_summary_and_label_metadata(chunk_text)
            
            # --- NEW: Get Knowledge Graph Entities ---
            entities = get_entities_from_chunk(chunk_text)
            metadata["entities"] = entities # Add entities to metadata
            # --- !! ---------------- !! ---

            if "labels" in metadata and isinstance(metadata["labels"], list):
                metadata["labels"] = [label.lower() for label in metadata["labels"]]
            
            metadata["text"] = chunk_text 
            metadata["source"] = doc["source"]
            vector_id = f"{doc['id']}-chunk-{i}"
            
            pinecone_batch.append({
                "id": vector_id,
                "values": embedding,
                "metadata": metadata
            })

            # Save chunk for BM25
            save_chunk_for_bm25(vector_id, chunk_text)

        # --- STEP 4: Upsert to Pinecone (Network, Parallel) ---
        upserted_count = upsert_batch_to_pinecone(pinecone_batch)
        
        return doc['id'], upserted_count, "Success"
        
    except Exception as e:
        logging.error(f"*** UNHANDLED ERROR in worker for {doc['id']}: {e} ***")
        return doc['id'], 0, f"Failed: {e}"


# --- Main Parallel Pipeline ---
def main():
    global local_embed_model
    
    setup_logging()
    start_time = time.time()
    
    # Clear BM25 corpus file
    if os.path.exists(BM25_CORPUS_FILE):
        os.remove(BM25_CORPUS_FILE)
    logging.info(f"Cleared {BM25_CORPUS_FILE} for new build.")
    
    docs_processed = 0
    total_docs = 0 
    total_vectors_upserted = 0

    try:
        # Load model in main thread ONCE
        logging.info(f"--- LOADING MODEL: {MODEL_NAME} onto cuda:0 ---")
        local_embed_model = SentenceTransformer(MODEL_NAME, device='cuda:0')
        logging.info("Model loaded successfully.")

        # 1. Load all the .txt files
        documents = load_all_cleaned_text()
        total_docs = len(documents)
        if not documents:
            logging.error(f"No documents found in {CLEANED_DATA_DIR}. Exiting.")
            return

        # 2. Initialize Pinecone (Main thread clears the index)
        index = init_pinecone_index_main(INDEX_NAME)
        
        logging.info(f"--- Starting Parallel Ingestion ({MAX_CPUS} CPU threads, 1 GPU) ---")
        
        # 3. Create ONE Pool (ThreadPool)
        with ThreadPoolExecutor(
            max_workers=MAX_CPUS,
            initializer=init_cpu_worker,
            initargs=(INDEX_NAME,)
        ) as cpu_pool:
            
            # Submit all jobs to the CPU pool
            futures = [cpu_pool.submit(process_document_worker, doc) for doc in documents]
            
            pbar = tqdm(total=total_docs, desc="Processing Documents", unit="doc")
            
            for i, future in enumerate(as_completed(futures)):
                
                if time.time() - start_time > TIMEOUT_SECONDS:
                    logging.warning(f"\n--- TIMEOUT: 8-hour limit reached. ---")
                    logging.warning("--- Beginning graceful shutdown... ---")
                    cpu_pool.shutdown(wait=True, cancel_futures=True) 
                    logging.info("--- Graceful shutdown complete. ---")
                    break
                    
                try:
                    doc_id, vector_count, status = future.result(timeout=FUTURE_TIMEOUT_SECONDS)
                    
                    if status == "Success":
                        total_vectors_upserted += vector_count
                    else:
                        logging.warning(f"Warning: Failed to process {doc_id}. Status: {status}")
                
                except concurrent.futures.TimeoutError:
                    logging.error(f"\n--- CRITICAL: Worker for doc {i} TIMED OUT after {FUTURE_TIMEOUT_SECONDS}s. ---")
                    logging.error(f"--- This worker is stuck (likely Gemini API) and will be abandoned. ---")
                    
                except Exception as e:
                    logging.error(f"\n--- CRITICAL WORKER ERROR: {e} ---")
                
                docs_processed += 1
                pbar.update(1)
                
                if docs_processed > 1:
                    elapsed = time.time() - start_time
                    avg_time_per_doc = elapsed / docs_processed
                    remaining_docs = total_docs - docs_processed
                    eta_seconds = remaining_docs * avg_time_per_doc
                    eta_hours = eta_seconds // 3600
                    eta_minutes = (eta_seconds % 3600) // 60
                    pbar.set_postfix_str(f"ETA: {int(eta_hours)}h {int(eta_minutes)}m")
            
            logging.info("CPU pool shut down.")

    except KeyboardInterrupt:
        logging.info("\n--- User interrupted. Shutting down. ---")
    
    finally:
        logging.info("\n--- Ingestion Pipeline Complete ---")
        logging.info(f"Processed {docs_processed} / {total_docs} documents.")
        logging.info(f"Total vectors upserted into '{INDEX_NAME}': {total_vectors_upserted}")
        
        try:
            if 'index' in locals() and index:
                 final_stats = index.describe_index_stats()
                 logging.info(final_stats)
            else:
                 final_index = init_pinecone_index_main(INDEX_NAME)
                 final_stats = final_index.describe_index_stats()
                 logging.info(final_stats)
        except Exception as e:
            logging.error(f"Could not get final index stats: {e}")

if __name__ == "__main__":
    main()