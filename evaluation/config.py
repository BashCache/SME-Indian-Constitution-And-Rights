# Configuration file for Session Log Evaluator

# Evaluation Weights (must sum to 1.0)
EVALUATION_WEIGHTS = {
    "rag_relevance": 0.30,      # How relevant retrieved documents are to query
    "tool_selection": 0.25,     # Whether correct tools were chosen
    "reasoning_quality": 0.25,  # Quality of agent reasoning process
    "response_quality": 0.20    # Final response quality and accuracy
}

# Scoring Thresholds for Performance Classification
PERFORMANCE_THRESHOLDS = {
    "excellent": 8.5,
    "good": 7.0,
    "fair": 5.5,
    "poor": 4.0
    # Below 4.0 is considered "inadequate"
}

# Rate limiting settings (seconds between API calls)
API_RATE_LIMIT_DELAY = 1.0

# Maximum content length for evaluation (to avoid token limits)
MAX_CONTENT_LENGTH = {
    "user_query": 1000,
    "rag_document_content": 500,
    "final_response": 2000,
    "reasoning_step": 300
}

# Expected tools for different query types
EXPECTED_TOOLS_BY_QUERY_TYPE = {
    "informational": ["normal_content_tool"],
    "quiz_request": ["interactive_quiz_tool"],
    "flashcard_request": ["flashcard_generation_tool"],
    "document_export": ["normal_content_tool", "document_export_tool"],
    "email_request": ["normal_content_tool", "send_email_tool"],
    "video_request": ["video_generation_tool"],
    "web_search": ["web_search_tool"]
}

# Evaluation prompt templates (for consistency)
EVALUATION_PROMPTS = {
    "rag_system_message": "You are an expert evaluator specializing in information retrieval and document relevance assessment.",
    "tool_system_message": "You are an expert evaluator specializing in AI agent tool selection and workflow optimization.",
    "reasoning_system_message": "You are an expert evaluator specializing in logical reasoning and decision-making processes.",
    "response_system_message": "You are an expert evaluator specializing in response quality and user satisfaction assessment."
}

# Output formatting settings
OUTPUT_SETTINGS = {
    "save_detailed_json": True,
    "save_summary_report": True,
    "print_progress": True,
    "include_raw_responses": False  # Set to True for debugging
}
