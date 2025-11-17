"""
This script runs the formal evaluation of our RAG pipeline.
It compares 4 different retrieval strategies:
1.  legal-bert (with KG Filter)
2.  legal-bert (without KG Filter)
3.  mpnet (with KG Filter)
4.  mpnet (without KG Filter)

It measures two metrics:
1.  Latency: Time taken to retrieve results.
2.  Relevance: Graded by an "LLM as a Judge" (Gemini).
"""
import os
import time
import json
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from dotenv import load_dotenv
from rag_tools.retrieval_tool import RAGTool # Import your tool
from tqdm import tqdm
import pandas as pd
import numpy as np

MAX_CHUNK_SIZE = 20000  # Max chars to send to LLM Judge
# --- Configuration ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not set (needed for LLM Judge).")

# --- Golden Query Set ---
# This is our test harness.
GOLDEN_QUERIES = [
    {
        "query": "What is Article 14 of the Indian Constitution?",
        "ground_truth": "Article 14 guarantees the right to equality. It states that the State shall not deny to any person equality before the law or the equal protection of the laws within the territory of India. It applies to all persons, citizens and non-citizens alike."
    },
    {
        "query": "Tell me about the Right to Freedom of Speech and Expression.",
        "ground_truth": "The Right to Freedom of Speech and Expression is a fundamental right guaranteed by Article 19(1)(a) of the Indian Constitution. It allows all citizens to express their views, opinions, and beliefs freely. This right is not absolute and is subject to 'reasonable restrictions' such as public order, decency, or morality."
    },
    {
        "query": "What is the Basic Structure Doctrine?",
        "ground_truth": "The Basic Structure Doctrine is a legal principle established by the Supreme Court of India in the Kesavananda Bharati v. State of Kerala case (1973). It holds that while Parliament has the power to amend the Constitution, it cannot amend or alter the 'basic structure' or fundamental features of the Constitution, such as democracy, secularism, and the rule of law."
    },
    {
        "query": "What are the powers of the President of India?",
        "ground_truth": "The President of India is the head of state and has executive, legislative, judicial, and military powers. Key powers include appointing the Prime Minister and Council of Ministers, giving assent to bills passed by Parliament, appointing judges of the Supreme Court and High Courts, and the power to grant pardons."
    },
    {
        "query": "What is a Writ of Habeas Corpus?",
        "ground_truth": "Habeas Corpus is a Latin term meaning 'you shall have the body'. It is a writ (a legal order) that requires a person who has been arrested or detained to be brought before a court. The court then determines if the person's detention is lawful. It is a fundamental right to protect individuals from illegal detention."
    }
]

# --- LLM as a Judge ---
LLM_JUDGE_PROMPT_TEMPLATE = """
You are an expert legal scholar acting as an "LLM as a Judge". Your task is to evaluate the relevance of a retrieved context in answering a user's query.

You must provide a score from 1 to 5 based on the following criteria:
1 - **Not Relevant:** The context is completely unrelated to the query.
2 - **Slightly Relevant:** The context mentions a general topic (e.g., 'constitution') but does not answer the specific question.
3 - **Relevant:** The context provides a partial or general answer to the query.
4 - **Highly Relevant:** The context directly and clearly answers the user's query.
5 - **Perfect:** The context is a textbook definition or a perfect, complete, and direct answer to the query.

You must also provide a 1-sentence justification for your score.
Return your response as a JSON object with "score" and "justification" keys.

---
User Query:
{query}

Ground Truth (for your reference):
{ground_truth}

Retrieved Context:
{context}
---
"""

LLM_JUDGE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "score": {"type": "NUMBER", "description": "The relevance score from 1-5."},
        "justification": {"type": "STRING", "description": "A 1-sentence justification for the score."}
    },
    "required": ["score", "justification"]
}

def judge_relevance(query, ground_truth, context):
    """
    Calls Gemini as an LLM Judge to get a relevance score.
    """
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        
        generation_config = GenerationConfig(
            response_mime_type="application/json",
            response_schema=LLM_JUDGE_SCHEMA,
            temperature=0.0 # We want deterministic, cold judging
        )
        gemini_model = genai.GenerativeModel(
            model_name="gemini-2.5-pro",
            generation_config=generation_config
        )
        
        prompt = LLM_JUDGE_PROMPT_TEMPLATE.format(
            query=query,
            ground_truth=ground_truth,
            context=context[:MAX_CHUNK_SIZE] # Truncate long contexts
        )
        
        response = gemini_model.generate_content(contents=prompt)
        data = json.loads(response.text)
        return data['score'], data['justification']
        
    except Exception as e:
        print(f"LLM Judge Error: {e}")
        return 0, "Judge error" # Return 0 on failure

# --- Main Evaluation ---
def run_evaluation():
    print("--- Starting RAG Pipeline Evaluation ---")
    
    # 1. Initialize both RAG tools
    print("Initializing RAGTool (legal-bert)...")
    rag_legal_bert = RAGTool(model_key="legal-bert")
    
    print("\nInitializing RAGTool (mpnet)...")
    # This assumes you have run ingestion for 'index-mpnet'
    try:
        rag_mpnet = RAGTool(model_key="mpnet")
        models_to_test = [rag_legal_bert, rag_mpnet]
    except Exception as e:
        print(f"Could not load mpnet: {e}. Proceeding with legal-bert only.")
        models_to_test = [rag_legal_bert]

    
    results = [] # To store all our data

    # Loop over each query
    for item in tqdm(GOLDEN_QUERIES, desc="Evaluating Queries"):
        query = item["query"]
        ground_truth = item["ground_truth"]
        
        # Loop over each model
        for model in models_to_test:
            
            # Loop over filter (True) and no-filter (False)
            for use_filter in [True, False]:
                
                print(f"\n--- Testing: {model.model_key} | Filter: {use_filter} ---")
                
                # --- 1. Measure Latency ---
                start_time = time.time()
                contexts = model.search(query, top_k=3, use_filter=use_filter)
                latency = time.time() - start_time
                
                if not contexts:
                    print("No results found.")
                    results.append({
                        "query": query,
                        "model": model.model_key,
                        "filter": use_filter,
                        "latency": latency,
                        "avg_relevance": 0,
                        "top_1_relevance": 0,
                        "retrieved_context": "No results",
                        "justification": "No results"
                    })
                    continue

                # --- 2. Measure Relevance (LLM as Judge) ---
                scores = []
                justifications = []
                
                # We only judge the TOP 1 result for cost/time
                top_1_context = contexts[0]['text']
                
                score, justification = judge_relevance(query, ground_truth, top_1_context)
                
                scores.append(score)
                justifications.append(justification)
                
                avg_score = np.mean(scores) if scores else 0

                results.append({
                    "query": query,
                    "model": model.model_key,
                    "filter": use_filter,
                    "latency": f"{latency:.2f}s",
                    "avg_relevance": avg_score,
                    "top_1_relevance": scores[0],
                    "retrieved_context": top_1_context[:200] + "...",
                    "justification": justifications[0]
                })

    # --- 3. Print Final Report ---
    print("\n\n" + "="*80)
    print("--- RAG PIPELINE EVALUATION REPORT ---")
    print("="*80)
    
    # Use Pandas for a nice table (install with: pip install pandas)
    df = pd.DataFrame(results)
    
    # Reorder columns for clarity
    df = df[["query", "model", "filter", "top_1_relevance", "latency", "justification"]]
    
    print(df.to_string())

    print("\n\n--- SUMMARY STATISTICS ---")
    
    # Calculate average relevance and latency for each of the 4 combos
    summary = df.groupby(['model', 'filter']).agg(
        avg_relevance=('top_1_relevance', 'mean'),
        avg_latency_str=('latency', lambda x: f"{pd.to_numeric(x.str.replace('s', '')).mean():.2f}s")
    ).reset_index()
    
    print(summary.to_string())
    print("\n--- EVALUATION COMPLETE ---")


if __name__ == "__main__":
    run_evaluation()