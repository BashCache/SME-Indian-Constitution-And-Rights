from typing import Literal, Dict, Any, List
import time

# This is the fake data that mimics a real Pinecone response
DUMMY_DOCUMENTS = [
    {
        "text": "Article 14 states that the State shall not deny to any person equality before the law or the equal protection of the laws within the territory of India. This is a fundamental right for K-12 students to understand.",
        "source": "cleaned_data/books_extracted/ncert_pol_sci_class_8.txt",
        "labels": ["Article 14", "Right to Equality", "Fundamental Right", "K-12", "Definition"],
        "summary": "A concise summary of Article 14 (Right to Equality).",
        "score": 0.958
    },
    {
        "text": "The Right to Freedom, under Article 19, includes the freedom of speech and expression, assembly, association, movement, and residence. This is a key concept in the civics curriculum.",
        "source": "cleaned_data/ppt_extracted/chapter_2_slides.txt",
        "labels": ["Article 19", "Right to Freedom", "Speech and Expression", "Civics", "K-12"],
        "summary": "An overview of the freedoms guaranteed under Article 19.",
        "score": 0.932
    },
    {
        "text": "Article 21 provides that no person shall be deprived of his life or personal liberty except according to procedure established by law. This right is the heart of the Constitution.",
        "source": "cleaned_data/case_studies/maneka_gandhi.txt",
        "labels": ["Article 21", "Right to Life", "Personal Liberty", "Case Study", "Constitution"],
        "summary": "A definition of Article 21 (Right to Life).",
        "score": 0.911
    }
]


class RAGTool:
    def __init__(self, model_key: Literal["mpnet", "legal-bert"] = "mpnet"):
        """
        MOCK Initializer.
        It does nothing but print a warning.
        """
        self.model_key = model_key
        print("="*50)
        print(f"--- WARNING: RAGTool is in MOCK MODE (using model '{self.model_key}') ---")
        print("--- This tool has NO dependencies and returns DUMMY data. ---")
        print("="*50)

    def search(self, query: str, filter_labels: List[str] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        MOCK Search Method.
        It prints the inputs it receives and returns dummy documents.
        """
        print(f"\n[Mock RAGTool] Received query: '{query}'")
        print(f"[Mock RAGTool] Received filters: {filter_labels}")
        print(f"[Mock RAGTool] Received top_k: {top_k}")
        
        # Simulate a network delay
        time.sleep(0.5) 
        
        # Return the dummy data, respecting the top_k parameter
        return DUMMY_DOCUMENTS[:top_k]

# Example of how Person 3 will use this:
if __name__ == "__main__":
    
    print("--- Person 3's Development Test ---")
    
    # Person 3 just needs to do this:
    rag_tool = RAGTool(model_key="mpnet")
    
    # And her code will work!
    my_query = "What is the Right to Equality?"
    contexts = rag_tool.search(my_query, filter_labels=["Article 14"], top_k=2)
    
    print(f"\n[Person 3] Got {len(contexts)} contexts back:")
    for i, context in enumerate(contexts):
        print(f"\nContext {i+1}:")
        print(f"  Text: {context['text'][:50]}...")
        print(f"  Source: {context['source']}")
        print(f"  Score: {context['score']}")