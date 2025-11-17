# Session Log Evaluator

A comprehensive evaluation system that analyzes AI agent performance using LLM-as-a-Judge methodology with Gemini Pro.

## Overview

This evaluation system analyzes session logs to assess agent performance across four key metrics:

1. **RAG Context Relevance** - How relevant are retrieved documents to the user query
2. **Tool Selection Accuracy** - Whether the right tools were chosen for the task
3. **Agent Reasoning Quality** - Quality of the agent's decision-making process
4. **Response Quality** - How well the final response answers the user's query

## Features

- 🤖 **LLM-as-a-Judge Evaluation**: Uses Gemini Pro for objective assessment
- 📊 **Comprehensive Scoring**: 1-10 scale with detailed rubrics
- 📋 **Detailed Reports**: JSON results + human-readable summaries
- 🔍 **Granular Analysis**: Document-level RAG scoring, tool-by-tool analysis
- 📈 **Aggregate Statistics**: Average scores across all sessions
- ⚙️ **Configurable**: Customizable weights and thresholds

## Setup

1. **Install Dependencies**:
   ```bash
   cd evaluation/
   pip install -r requirements.txt
   ```

2. **Set API Key**:
   ```bash
   export GEMINI_API_KEY="your-gemini-api-key"
   ```

3. **Verify Session Logs**: Ensure your session logs are in `agent_data/session_logs/`

## Usage

### Quick Start
```bash
# Run evaluation on default directory
python run_evaluation.py

# Run on specific directory
python run_evaluation.py /path/to/session/logs
```

### Advanced Usage
```python
from session_log_evaluator import SessionEvaluator

# Initialize evaluator
evaluator = SessionEvaluator("path/to/logs")

# Run evaluation
results = evaluator.evaluate_all_sessions()

# Access results
print(f"Overall score: {results['summary']['average_scores']['overall_score']}")
```

## Evaluation Metrics

### 1. RAG Context Relevance (30% weight)
- **Excellent (9-10)**: Directly addresses query, highly relevant
- **Good (7-8)**: Relevant with minor tangential content
- **Fair (5-6)**: Somewhat relevant but missing key aspects
- **Poor (3-4)**: Limited relevance, mostly off-topic
- **Irrelevant (1-2)**: No relevance to the query

### 2. Tool Selection Accuracy (25% weight)
- **Perfect (9-10)**: All tools optimal and necessary
- **Good (7-8)**: Most tools appropriate, minor inefficiencies
- **Adequate (5-6)**: Generally correct with some issues
- **Poor (3-4)**: Wrong tools or missing critical tools
- **Incorrect (1-2)**: Completely wrong tool selection

### 3. Agent Reasoning Quality (25% weight)
- **Excellent (9-10)**: Clear, logical, well-structured reasoning
- **Good (7-8)**: Generally sound with minor gaps
- **Fair (5-6)**: Some logical flow but unclear in parts
- **Poor (3-4)**: Flawed reasoning or significant gaps
- **Incoherent (1-2)**: No clear reasoning or illogical

### 4. Response Quality (20% weight)
- **Excellent (9-10)**: Comprehensive, accurate, directly answers query
- **Good (7-8)**: Good answer with minor omissions
- **Adequate (5-6)**: Partially answers query, some inaccuracies
- **Poor (3-4)**: Limited answer, significant issues
- **Inadequate (1-2)**: Doesn't answer query or major inaccuracies

## Output Files

The evaluator generates several output files in the `evaluation/results/` directory:

### 1. Detailed JSON Results
`session_evaluation_YYYYMMDD_HHMMSS.json`
- Complete evaluation data for all sessions
- Individual document scores
- Tool execution analysis
- Reasoning step assessments
- Raw LLM judgments

### 2. Summary Report
`evaluation_summary_YYYYMMDD_HHMMSS.txt`
- Human-readable performance overview
- Average scores across all metrics
- Individual session summaries
- Key insights and trends

## Expected Session Log Format

The evaluator expects session logs with the following sections:

```
================================================================================
AGENT SESSION LOG
================================================================================
Timestamp: 2025-11-17T21:45:25.875927
Session ID: session_123
Processing Time: 2.45 seconds

================================================================================
USER QUERY
================================================================================
[User's question]

================================================================================
GUARDRAIL VALIDATION
================================================================================
Status: passed
Validation Time: 0.245 seconds
[Additional guardrail info]

================================================================================
RAG CONTEXT RETRIEVED
================================================================================
Document 1 (Score: 0.536):
Source: source_file.txt
Content: [Document content]
Labels: label1, label2, label3

[Additional documents...]

================================================================================
AGENT REASONING & SCRATCHPAD
================================================================================
Step 1 [AIMessage]:
[Reasoning content]

[Additional steps...]

================================================================================
TOOL EXECUTIONS
================================================================================
Tool Execution 1:
[Tool execution details]

[Additional tool executions...]

================================================================================
FINAL RESPONSE
================================================================================
[Agent's final response to user]
```

## Configuration

Modify `config.py` to customize:

- **Evaluation weights** for different metrics
- **Performance thresholds** for classification
- **API rate limiting** settings
- **Content length limits** for token management
- **Expected tools** for different query types

## Troubleshooting

### Common Issues

1. **"No log files found"**
   - Check the logs directory path
   - Ensure `.txt` files exist in the directory

2. **"GEMINI_API_KEY not set"**
   - Set the environment variable: `export GEMINI_API_KEY="your-key"`

3. **"Failed to parse JSON response"**
   - LLM sometimes returns malformed JSON
   - Check the raw response in error output
   - Consider adjusting the prompt templates

4. **Rate limiting errors**
   - Increase `API_RATE_LIMIT_DELAY` in config.py
   - The evaluator includes built-in delays between calls

### Performance Tips

- **Large log files**: The parser truncates content to avoid token limits
- **Many sessions**: Evaluation runs sequentially to avoid rate limits
- **API costs**: Each session requires 4 LLM calls (one per metric)

## Example Output

```
📊 EVALUATION SUMMARY for agent_session_test_20251117_214525.txt
============================================================
RAG Relevance Score:    8.60/10
Tool Selection Score:   9.00/10
Reasoning Quality:      7.50/10
Response Quality:       8.20/10
────────────────────────────────────────────────────────────
OVERALL SCORE:          8.33/10

🎯 RAG Assessment: Highly relevant documents with strong constitutional law coverage
🛠️ Tool Selection: Perfect tool choice for informational query
```

## Extending the Evaluator

### Adding New Metrics

1. Create evaluation method in `LLMJudge` class
2. Add metric to `_calculate_overall_scores()` 
3. Update configuration weights
4. Modify summary generation

### Custom Scoring Rubrics

Edit the rubric dictionaries in `EvaluationCriteria` class to match your specific needs.

### Different LLM Models

Replace the Gemini model initialization in `LLMJudge.__init__()` with your preferred model.

## License

This evaluation system is part of the SME-Indian-Constitution-And-Rights project.
