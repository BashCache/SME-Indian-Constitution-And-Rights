"""
This script runs AFTER the main ingestion pipeline.
It reads the 'bm25_corpus.jsonl' file created by the pipeline,
builds a BM25Okapi index, and saves it to a pickle file.
It also saves a simple ID-to-text mapping for easy lookup.
"""

import json
import pickle
import time
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import re

CORPUS_FILE = 'bm25_corpus.jsonl'
BM25_INDEX_FILE = 'bm25_index.pkl'
CHUNK_DB_FILE = 'bm25_chunk_db.json'

def build_index():
    print(f"Loading corpus from {CORPUS_FILE}...")
    corpus = []
    chunk_db = {}
    doc_ids = []
    try:
        with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                corpus.append(data['text'])
                chunk_db[data['id']] = data['text']
                doc_ids.append(data['id'])
    except FileNotFoundError:
        print(f"Error: {CORPUS_FILE} not found. Please run ingestion_pipeline.py first.")
        return
    except Exception as e:
        print(f"Error loading corpus: {e}")
        return

    if not corpus:
        print("Corpus is empty. Exiting.")
        return

    print(f"Loaded {len(corpus)} chunks into memory.")

    # 1. Tokenize the corpus
    print("Tokenizing corpus for BM25 (this may take a moment)...")
    start_time = time.time()
    
    def bm25_tokenizer(text):
        text = text.lower()
        token_pattern = re.compile(r'(?u)\b\w\w+\b')
        return token_pattern.findall(text)

    tokenized_corpus = [bm25_tokenizer(doc) for doc in tqdm(corpus)]
    print(f"Tokenization complete in {time.time() - start_time:.2f}s")

    # 2. Build the BM25 index
    print("Building BM25 index...")
    start_time = time.time()
    bm25 = BM25Okapi(tokenized_corpus)
    print(f"BM25 index built in {time.time() - start_time:.2f}s")

    # 3. Save the index to a pickle file
    print(f"Saving BM25 index to {BM25_INDEX_FILE}...")
    with open(BM25_INDEX_FILE, 'wb') as f:
        pickle.dump(bm25, f)
    
    # 4. Save the chunk ID-to-text mapping
    print(f"Saving chunk database to {CHUNK_DB_FILE}...")
    with open(CHUNK_DB_FILE, 'w', encoding='utf-8') as f:
        # We need to save the chunk_db (id -> text) AND
        # the doc_ids (index -> id)
        save_data = {
            'chunk_db': chunk_db,
            'doc_ids': doc_ids
        }
        json.dump(save_data, f)

    print("\n--- BM25 Build Complete ---")
    print(f"Index saved to: {BM25_INDEX_FILE}")
    print(f"Chunk DB saved to: {CHUNK_DB_FILE}")

if __name__ == "__main__":
    # You must install rank_bm25: pip install rank-bm25
    build_index()