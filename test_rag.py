from rag_tools.retrieval_tool import RAGTool
import time
import os

def test_advanced_rag():
    print("--- Initializing RAGTool for 'legal-bert' ---")
    try:
        rag_tool = RAGTool(model_key="legal-bert")
        print("--- RAGTool initialized successfully. ---")
    except Exception as e:
        print(f"--- FAILED to initialize RAGTool: {e} ---")
        return

    # --- This is the full, advanced agent workflow ---
    print("\n" + "="*50)
    print("--- TEST: Running Advanced Agentic Search ---")
    
    # 1. User sends a raw query
    user_query = "Tell me what the constitution says about equality, especially in Article 14."
    print(f"User Query: '{user_query}'")

    # 2. (Person 3's Agent) Calls your RAGTool.
    #    The tool will internally print "Step 1: Analyzing query..."
    #    and "Step 2: Performing filtered search..."
    print("\nCalling RAGTool.search()...")
    start_time = time.time()
    contexts = rag_tool.search(
        query=user_query,
        top_k=3
    )
    end_time = time.time()
    print(f"RAGTool.search() complete. Found {len(contexts)} results in {end_time - start_time:.2f}s")

    # 3. Display results
    print("\n" + "="*50)
    print("--- FINAL FILTERED RESULTS ---")
    for i, res in enumerate(contexts):
        print(f"\nResult {i+1} (Score: {res['score']:.4f})")
        print(f"  Source: {res['source']}")
        print(f"  Text: {res['text'][:150]}...")
        print(f"  Labels: {res['labels'][:5]}...")

if __name__ == "__main__":
    test_advanced_rag()
