
import os
import requests
import pinecone
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import Literal, Dict, Any, List
import time

# (Task 2.4 - TODO) Import BM25
# from rank_bm25 import BM25Okapi
# import pickle

# --- Model Configuration ---
# This dictionary maps a simple key to the REAL model and index.
# This MUST match the parameters you used in `run_pipeline.py`.
MODEL_CONFIG = {
    "mpnet": {
        "model_name_or_path": "sentence-transformers/all-mpnet-base-v2",
        "index_name": "index-mpnet"
    },
    "legal-bert": {
        # --- THIS LINE IS THE FIX ---
        "model_name_or_path": "nlpaueb/legal-bert-base-uncased",
        # --- !! ---------------- !! ---
        "index_name": "index-legal-bert"
    }
}
# --- !! ---------------- !! ---

class RAGTool:
    def __init__(self, model_key: Literal["mpnet", "legal-bert"] = "mpnet"):
        """
        Initializes the RAGTool for a SPECIFIC model.
        Person 3 can create two instances to compare:
        rag_mpnet = RAGTool(model_key="mpnet")
        rag_legal_bert = RAGTool(model_key="legal-bert")
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
        pinecone_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_key:
            raise ValueError("PINECONE_API_KEY not set.")
            
        # --- Initialize Connections ---
        pc = pinecone.Pinecone(api_key=pinecone_key)
        
        if self.index_name not in pc.list_indexes().names():
            print(f"Warning: Index '{self.index_name}' does not exist.")
            print("Please run the ingestion pipeline first.")
            self.index = None
        else:
            self.index = pc.Index(self.index_name)
            print("Pinecone index stats:")
            print(self.index.describe_index_stats())
        
        # --- (Task 2.4) Load BM25 Index ---
        # self.bm25_index, self.bm25_corpus = self._load_bm25_assets()
        # print("BM25 index loaded.")
        
        # --- Load local embedding model ---
        print(f"Loading local embedding model: {self.model_path}")
        self.embed_model = SentenceTransformer(self.model_path)
        
        # --- Load local RERANKER model ---
        print("Loading local reranker model (BAAI/bge-reranker-base)...")
        self.reranker = CrossEncoder('BAAI/bge-reranker-base')
        print("RAGTool initialized successfully.")

    def _load_bm25_assets(self):
        # (Task 2.4 - TODO)
        return None, None 

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
        (Task 2.4)
        Performs vector search with metadata filtering.
        """
        if not self.index:
            print(f"Search failed: Index '{self.index_name}' is not available.")
            return [] # No index to search

        metadata_filter = {}
        if filter_labels and isinstance(filter_labels, list):
            # This is where your Gemini metadata pays off!
            metadata_filter["labels"] = {"$in": filter_labels}
            
        vector_results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=metadata_filter if metadata_filter else None
        )
        pinecone_matches = vector_results.get("matches", [])
        
        # (TODO: Add BM25 and fusion logic here)
        
        return pinecone_matches 

    def _rerank_results(self, query, candidates):
        """
        (Task 2.5)
        Reranks results using the LOCAL bge-reranker model.
        """
        if not candidates:
            return []
            
        print(f"Reranking {len(candidates)} candidates locally...")
        
        # Create pairs of [query, document_text]
        documents_to_rerank = [match["metadata"]["text"] for match in candidates]
        pairs = [(query, doc) for doc in documents_to_rerank]
        
        try:
            # Get scores from the local model
            scores = self.reranker.predict(pairs)
            
            # Combine candidates, scores, and sort
            final_candidates = []
            for i, score in enumerate(scores):
                candidates[i]["rerank_score"] = score
                final_candidates.append(candidates[i])
                
            # Sort by the new rerank_score, highest first
            final_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
            
            print("Reranking complete.")
            return final_candidates
            
        except Exception as e:
            print(f"Error during local reranking: {e}. Returning non-reranked results.")
            # Fallback: just return the original vector search results
            return sorted(candidates, key=lambda x: x.get('score', 0), reverse=True)

    def search(self, query: str, filter_labels: List[str] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        The main public method for Person 3.
        """
        print(f"\nRAGTool ({self.model_key}): Received search query: '{query}'")
        
        query_embedding = self._get_query_embedding(query)
        if not query_embedding:
            return [] 

        # We fetch more (e.g., 50) to give the reranker a good pool of docs
        candidates = self._hybrid_search(
            query, 
            query_embedding, 
            filter_labels=filter_labels, 
            top_k=top_k * 10
        )
        
        if not candidates:
            return [] 
            
        final_candidates = self._rerank_results(query, candidates)
            
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
        print("--- Initializing Legal-BERT RAG Tool (for Person 3) ---")
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
