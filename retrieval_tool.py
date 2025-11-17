"""
(Task 2.6 - FINAL-KG-HYBRID VERSION)

CHANGES:
1.  Implements the Knowledge Graph (BONUS) as a metadata filter.
2.  Brings back the Gemini Query Analyzer ('_get_entities_from_query')
    to extract entities (e.g., 'article 14') from the user's query.
3.  The 'search' method now has a 'use_filter: bool' flag,
    which allows you to run the A/B test.
4.  This creates a 3-part hybrid search:
    - Dense (Vector)
    - Keyword (BM25)
    - Knowledge Graph (Entity Filter)
"""

import os
import pinecone
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import Literal, Dict, Any, List, Optional
import time
import json
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
import logging
import pickle
from rank_bm25 import BM25Okapi 
import numpy as np 
import re

# --- BM25 File Paths ---
BM25_INDEX_FILE = 'bm25_index.pkl'
CHUNK_DB_FILE = 'bm25_chunk_db.json'

# --- Model Configuration ---
MODEL_CONFIG = {
    "mpnet": {
        "model_name_or_path": "sentence-transformers/all-mpnet-base-v2",
        "index_name": "index-mpnet"
    },
    "legal-bert": {
        "model_name_or_path": "nlpaueb/legal-bert-base-uncased",
        "index_name": "index-legal-bert"
    }
}
# --- !! ---------------- !! ---

# --- Gemini Query Analyzer Config ---
QUERY_ANALYZER_SYSTEM_PROMPT = (
    "You are a search query analyzer. Your job is to extract the 3-5 most specific, "
    "unique, and important keywords, topics, or legal article numbers from the user's query. \n"
    "CRITICAL: Prioritize specific entities. For example, if the query is "
    "'what is equality in article 14 of the constitution', you should return "
    "['article 14', 'equality'] and NOT ['constitution'].\n"
    "Return your response as a JSON list of strings."
    "\n\nQuery: {query}"
)
QUERY_ANALYZER_SCHEMA = {
    "type": "ARRAY",
    "description": "A list of 3-5 extracted keywords/labels.",
    "items": { "type": "STRING" }
}
# --- !! ---------------- !! ---


class RAGTool:
    def __init__(self, model_key: Literal["mpnet", "legal-bert"] = "mpnet"):
        """
        Initializes the RAGTool for a SPECIFIC model.
        """
        load_dotenv()
        
        if model_key not in MODEL_CONFIG:
            raise ValueError(f"Invalid model_key. Choose from {list(MODEL_CONFIG.keys())}")
        
        config = MODEL_CONFIG[model_key]
        self.model_key = model_key
        self.index_name = config["index_name"]
        self.model_path = config["model_name_or_path"]
        
        print(f"Initializing RAGTool for model: {self.model_key}")
        print(f"Targeting Pinecone index: {self.index_name}")

        # --- Load Config ---
        self.pinecone_key = os.getenv("PINECONE_API_KEY")
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        if not self.pinecone_key or not self.gemini_key:
            raise ValueError("PINECONE_API_KEY or GEMINI_API_KEY not set.")
            
        # --- Initialize Connections ---
        pc = pinecone.Pinecone(api_key=self.pinecone_key)
        
        if self.index_name not in pc.list_indexes().names():
            print(f"Warning: Index '{self.index_name}' does not exist.")
            print("Please run the ingestion pipeline first.")
            self.index = None
        else:
            self.index = pc.Index(self.index_name)
            print("Pinecone index stats:")
            print(self.index.describe_index_stats())
        
        # --- Load local embedding model ---
        print(f"Loading local embedding model: {self.model_path}")
        self.embed_model = SentenceTransformer(self.model_path)
        
        # --- Load local RERANKER model ---
        print("Loading local reranker model (BAAI/bge-reranker-base)...")
        self.reranker = CrossEncoder('BAAI/bge-reranker-base')
        
        # --- Configure Gemini Model (for query analysis) ---
        print("Configuring Gemini for query analysis...")
        genai.configure(api_key=self.gemini_key)
        generation_config = GenerationConfig(
            response_mime_type="application/json",
            response_schema=QUERY_ANALYZER_SCHEMA
        )
        self.gemini_analyzer = genai.GenerativeModel(
            model_name="gemini-2.5-pro",
            generation_config=generation_config
        )
        
        # --- NEW: Load BM25 Index ---
        try:
            print(f"Loading BM25 index from {BM25_INDEX_FILE}...")
            with open(BM25_INDEX_FILE, 'rb') as f:
                self.bm25_index = pickle.load(f)
            print(f"Loading chunk database from {CHUNK_DB_FILE}...")
            with open(CHUNK_DB_FILE, 'r', encoding='utf-8') as f:
                save_data = json.load(f)
                self.chunk_db = save_data['chunk_db']
                self.doc_ids = save_data['doc_ids']
            print("BM25 assets loaded successfully.")
        except Exception as e:
            print(f"--- FATAL ERROR: Could not load BM25 assets ---")
            print(f"--- {e} ---")
            print(f"--- Did you run 'pip install rank-bm25' and 'python build_bm25_index.py' first? ---")
            self.bm25_index = None
            self.chunk_db = None
            self.doc_ids = []
        # --- !! ---------------- !! ---

        print("RAGTool initialized successfully.")

    def _get_entities_from_query(self, query: str) -> List[str]:
        """
        Takes a raw user query and returns a list of lowercase filter entities.
        """
        try:
            prompt = QUERY_ANALYZER_SYSTEM_PROMPT.format(query=query)
            response = self.gemini_analyzer.generate_content(contents=prompt)
            
            entities_list = json.loads(response.text)
            # Normalize to lowercase to match the index
            lowercase_entities = [str(entity).lower() for entity in entities_list]
            print(f"[RAGTool] Generated entities: {lowercase_entities}")
            return lowercase_entities
            
        except Exception as e:
            print(f"Error in RAGTool Query Analyzer: {e}")
            return [] # Return an empty list on failure

    def _get_query_embedding(self, query):
        """
        Gets the query embedding using its specific loaded model.
        """
        try:
            embedding = self.embed_model.encode([query])[0].tolist()
            if not embedding:
                 raise ValueError("Could not generate embedding")
            return embedding
        except Exception as e:
            print(f"Error getting query embedding: {e}")
            return None

    def _reciprocal_rank_fusion(self, results_lists, k=60):
        """
        Performs RRF fusion on multiple result lists.
        k controls the influence of lower-ranked items.
        """
        fused_scores = {}
        
        print(f"Fusing {len(results_lists)} result lists...")
        for results in results_lists:
            for rank, doc_id in enumerate(results):
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = 0
                # RRF formula
                fused_scores[doc_id] += 1.0 / (k + rank + 1)
        
        # Sort by fused score
        reranked_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return reranked_results

    def _hybrid_search(self, query, query_embedding, filter_entities=None, top_k=50):
        """
        Performs TRUE hybrid search (Dense + Keyword + KG Entity Filter)
        """
        if not self.index or not self.bm25_index:
            print(f"Search failed: Index or BM25 not available.")
            return [], [] 

        # --- 1. Dense Search (Pinecone) ---
        metadata_filter = {}
        if filter_entities and isinstance(filter_entities, list):
            print(f"[RAGTool] Applying KG entity filters: {filter_entities}")
            # Use the 'entities' field we created during ingestion
            metadata_filter["entities"] = {"$in": filter_entities}
        else:
            print("[RAGTool] No KG entities extracted, performing broad search.")
        
        vector_results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=False, # We only need IDs
            filter=metadata_filter if metadata_filter else None
        )
        pinecone_matches = vector_results.get("matches", [])
        pinecone_ids = [match['id'] for match in pinecone_matches]
        print(f"Found {len(pinecone_ids)} dense results.")

        # --- 2. Keyword Search (BM25) ---
        token_pattern = re.compile(r'(?u)\b\w\w+\b')
        tokenized_query = token_pattern.findall(query.lower())
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        
        bm25_top_n_indices = np.argsort(bm25_scores)[::-1][:top_k]
        bm25_ids = [self.doc_ids[i] for i in bm25_top_n_indices if bm25_scores[i] > 0]
        print(f"Found {len(bm25_ids)} keyword results.")

        # --- 3. Fuse Results (RRF) ---
        fused_results = self._reciprocal_rank_fusion([pinecone_ids, bm25_ids])
        
        fused_doc_ids = [doc_id for doc_id, score in fused_results][:top_k]
        
        # --- 4. Fetch full data for reranking ---
        if not fused_doc_ids:
            return [], []
            
        fetch_response = self.index.fetch(ids=fused_doc_ids)
        
        if not fetch_response.vectors:
            return [], [] 
        
        candidates_from_pinecone = list(fetch_response.vectors.values())

        print(f"Fetched {len(candidates_from_pinecone)} fused candidates for reranking.")
        
        return candidates_from_pinecone, fused_results


    def _rerank_results(self, query, candidates, fused_results):
        """
        Reranks results using the LOCAL bge-reranker model.
        """
        if not candidates:
            return []
            
        print(f"Reranking {len(candidates)} candidates locally...")
        
        rrf_score_map = dict(fused_results)
        
        candidates_to_rerank = []
        for vector_obj in candidates:
            doc_id = vector_obj.id
            # Pinecone v3+ stores metadata in .metadata
            metadata = vector_obj.metadata if hasattr(vector_obj, 'metadata') else {}
            
            if 'text' not in metadata:
                metadata['text'] = self.chunk_db.get(doc_id, "")
            
            candidates_to_rerank.append({
                "id": doc_id,
                "score": rrf_score_map.get(doc_id, 0.0), # Add the RRF score
                "metadata": metadata
            })
        
        documents_to_rerank_text = [match["metadata"]["text"] for match in candidates_to_rerank]
        pairs = [(query, doc) for doc in documents_to_rerank_text]
        
        try:
            scores = self.reranker.predict(pairs)
            
            final_candidates = []
            for i, score in enumerate(scores):
                candidates_to_rerank[i]["rerank_score"] = float(score) # Ensure score is float
                final_candidates.append(candidates_to_rerank[i])
                
            final_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
            print("Reranking complete.")
            return final_candidates
            
        except Exception as e:
            print(f"Error during local reranking: {e}. Returning non-reranked results.")
            return sorted(candidates_to_rerank, key=lambda x: x.get('score', 0), reverse=True)

    def search(self, query: str, top_k: int = 5, use_filter: bool = True) -> List[Dict[str, Any]]:
        """
        The main public method for Person 3.
        It now performs a true hybrid search.
        'use_filter' flag controls the KG entity filter for evaluation.
        """
        print(f"\nRAGTool ({self.model_key}): Received search query: '{query}'")
        
        filter_entities = None
        if use_filter:
            # --- NEW STEP 1: Get Entities from Query ---
            filter_entities = self._get_entities_from_query(query)
            # --- !! ---------------- !! ---
        else:
            print("[RAGTool] Skipping entity filter for this query.")

        # --- STEP 2: Get Query Embedding ---
        query_embedding = self._get_query_embedding(query)
        if not query_embedding:
            return [] 

        # --- STEP 3: Hybrid Search ---
        candidates, fused_results = self._hybrid_search(
            query, 
            query_embedding, 
            filter_entities=filter_entities, # Pass the generated entities
            top_k=top_k * 10
        )
        
        if not candidates and filter_entities: # Only fallback if we used a filter
            print("[RAGTool] No results found with entity filter. Trying broad search...")
            # Fallback to a broad search WITH NO FILTERS
            candidates, fused_results = self._hybrid_search(
                query,
                query_embedding,
                filter_entities=None, # No filters
                top_k=top_k * 10
            )
            if not candidates:
                print("[RAGTool] No results found even with broad search.")
                return []
            
        # --- STEP 4: Rerank ---
        final_candidates = self._rerank_results(query, candidates, fused_results)
            
        # --- STEP 5: Format and Return ---
        contexts = []
        for match_dict in final_candidates[:top_k]:
            metadata = match_dict.get("metadata", {})
            if not metadata.get("text"):
                metadata["text"] = self.chunk_db.get(match_dict.get("id"), "")
            if not metadata.get("source"):
                 metadata["source"] = "Source not found in vector metadata"

            contexts.append({
                "text": metadata["text"],
                "source": metadata["source"],
                "labels": metadata.get("labels", []),
                "entities": metadata.get("entities", []), # <-- Add entities to final output
                "summary": metadata.get("summary", ""),
                "score": match_dict.get("rerank_score", match_dict.get("score"))
            })
            
        return contexts

# Example of how you can test this file:
if __name__ == "__main__":
    try:
        print("--- Initializing Legal-BERT RAG Tool ---")
        rag_tool_legal = RAGTool(model_key="legal-bert")
        
        query1 = "What is Article 14 of the Indian Constitution?"
        
        print("\n--- TEST 1: WITH KG FILTER ---")
        results = rag_tool_legal.search(query1, top_k=3, use_filter=True)
        
        print(f"\n--- Results for: '{query1}' (using 'legal-bert' WITH filter) ---")
        if results:
            for i, res in enumerate(results):
                print(f"\nResult {i+1} (Score: {res['score']:.4f})")
                print(f"Source: {res['source']}")
                print(f"Entities: {res['entities']}")
                print(f"Text: {res['text'][:250]}...")
        else:
            print("No results found.")
            
        print("\n--- TEST 2: WITHOUT KG FILTER ---")
        results_no_filter = rag_tool_legal.search(query1, top_k=3, use_filter=False)
        
        print(f"\n--- Results for: '{query1}' (using 'legal-bert' WITHOUT filter) ---")
        if results_no_filter:
            for i, res in enumerate(results_no_filter):
                print(f"\nResult {i+1} (Score: {res['score']:.4f})")
                print(f"Source: {res['source']}")
                print(f"Entities: {res['entities']}")
                print(f"Text: {res['text'][:250]}...")
        else:
            print("No results found.")
            
    except ValueError as e:
        print(f"Error initializing RAGTool: {e}")
        print("Reminder: You must set PINECONE_API_KEY and GEMINI_API_KEY in your .env file.")
