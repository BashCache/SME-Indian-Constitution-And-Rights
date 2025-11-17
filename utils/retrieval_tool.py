"""
(Task 2.2 - FINAL VERSION - with Query Expansion)
This is your "read-only" RAG tool.
Give this file to Person 3.

CHANGES:
1.  This tool is now a complete "black box" for retrieval.
2.  Absorbed the logic from 'agent_tools/query_analyzer.py'.
3.  The public '.search()' method no longer takes 'filter_labels'.
4.  It now has a new private method '_get_labels_from_query'
    that calls Gemini to generate labels *internally*.
5.  Loads the reranker model locally.
"""

import os
import requests
import pinecone
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import Literal, Dict, Any, List
import time
import json
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
import logging

# (Task 2.4 - TODO) Import BM25
# from rank_bm25 import BM25Okapi
# import pickle

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
    "You are a search query analyzer. Your job is to extract the 3-5 most important "
    "keywords, topics, or legal article numbers from the user's query. \n"
    "Focus on specific, filterable terms (like 'article 14', 'fundamental rights', 'equality'). \n"
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
    def __init__(self, model_key: Literal["mpnet", "legal-bert"] = "legal-bert"):
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
        
        print("RAGTool initialized successfully.")

    def _get_labels_from_query(self, query: str) -> List[str]:
        """
        Takes a raw user query and returns a list of lowercase filter labels.
        This is the new internal step.
        """
        try:
            prompt = QUERY_ANALYZER_SYSTEM_PROMPT.format(query=query)
            response = self.gemini_analyzer.generate_content(contents=prompt)
            
            labels_list = json.loads(response.text)
            lowercase_labels = [label.lower() for label in labels_list]
            print(f"[RAGTool] Generated labels: {lowercase_labels}")
            return lowercase_labels
            
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

    def _hybrid_search(self, query, query_embedding, filter_labels=None, top_k=50):
        """
        Performs vector search with metadata filtering.
        """
        if not self.index:
            print(f"Search failed: Index '{self.index_name}' is not available.")
            return [] 

        metadata_filter = {}
        if filter_labels and isinstance(filter_labels, list):
            print(f"[RAGTool] Applying metadata filters: {filter_labels}")
            metadata_filter["labels"] = {"$in": filter_labels}
        else:
            print("[RAGTool] No filters applied, performing broad search.")
            
        vector_results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=metadata_filter if metadata_filter else None
        )
        pinecone_matches = vector_results.get("matches", [])
        
        return pinecone_matches 

    def _rerank_results(self, query, candidates):
        """
        Reranks results using the LOCAL bge-reranker model.
        """
        if not candidates:
            return []
            
        print(f"Reranking {len(candidates)} candidates locally...")
        
        documents_to_rerank = [match["metadata"]["text"] for match in candidates]
        pairs = [(query, doc) for doc in documents_to_rerank]
        
        try:
            scores = self.reranker.predict(pairs)
            
            final_candidates = []
            for i, score in enumerate(scores):
                candidates[i]["rerank_score"] = score
                final_candidates.append(candidates[i])
                
            final_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
            print("Reranking complete.")
            return final_candidates
            
        except Exception as e:
            print(f"Error during local reranking: {e}. Returning non-reranked results.")
            return sorted(candidates, key=lambda x: x.get('score', 0), reverse=True)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        The main public method for Person 3.
        It now automatically generates filter labels.
        """
        print(f"\nRAGTool ({self.model_key}): Received search query: '{query}'")
        
        # --- NEW STEP 1: Generate Labels from Query ---
        filter_labels = self._get_labels_from_query(query)
        # --- !! ---------------- !! ---

        # --- STEP 2: Get Query Embedding ---
        query_embedding = self._get_query_embedding(query)
        if not query_embedding:
            return [] 

        # --- STEP 3: Hybrid Search ---
        candidates = self._hybrid_search(
            query, 
            query_embedding, 
            filter_labels=filter_labels, # Pass the generated labels
            top_k=top_k * 10
        )
        
        if not candidates:
            return [] 
            
        # --- STEP 4: Rerank ---
        final_candidates = self._rerank_results(query, candidates)
            
        # --- STEP 5: Format and Return ---
        contexts = []
        for match in final_candidates[:top_k]:
            contexts.append({
                "text": match["metadata"]["text"],
                "source": match["metadata"]["source"],
                "labels": match["metadata"].get("labels", []),
                "summary": match["metadata"].get("summary", ""),
                "score": match.get("rerank_score", match.get("score"))
            })
            
        return contexts

# Example of how you can test this file:
if __name__ == "__main__":
    try:
        print("--- Initializing Legal-BERT RAG Tool ---")
        rag_tool_legal = RAGTool(model_key="legal-bert")
        
        query1 = "What is Article 14 of the Indian Constitution?"
        results = rag_tool_legal.search(query1, top_k=3)
        
        print(f"\n--- Results for: '{query1}' (using 'legal-bert') ---")
        if results:
            for i, res in enumerate(results):
                print(f"\nResult {i+1} (Score: {res['score']:.4f})")
                print(f"Source: {res['source']}")
                print(f"Labels: {res['labels']}")
                print(f"Text: {res['text'][:250]}...")
        else:
            print("No results found. (Have you run the ingestion pipeline for 'index-legal-bert'?)")
            
    except ValueError as e:
        print(f"Error initializing RAGTool: {e}")
        print("Reminder: You must set PINECONE_API_KEY and GEMINI_API_KEY in your .env file.")