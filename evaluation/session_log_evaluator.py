#!/usr/bin/env python3
"""
Session Log Evaluator - Analyzes agent performance using LLM-as-a-Judge approach
Evaluates: RAG relevance, Tool selection, Agent reasoning, Response quality
"""

import os
import json
import re
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Configure Gemini Pro
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

@dataclass
class EvaluationCriteria:
    """Evaluation rubrics and scoring criteria"""
    
    # RAG Context Relevance Scoring
    RAG_RELEVANCE_RUBRIC = {
        "excellent": (9, 10),    # Highly relevant, directly addresses query
        "good": (7, 8),          # Relevant with minor tangential content
        "fair": (5, 6),          # Somewhat relevant but missing key aspects
        "poor": (3, 4),          # Limited relevance, mostly off-topic
        "irrelevant": (1, 2)     # No relevance to the query
    }
    
    # Tool Selection Accuracy
    TOOL_SELECTION_RUBRIC = {
        "perfect": (9, 10),      # All tools selected are optimal and necessary
        "good": (7, 8),          # Most tools appropriate, minor inefficiencies
        "adequate": (5, 6),      # Generally correct tools with some issues
        "poor": (3, 4),          # Wrong tools selected or missing critical tools
        "incorrect": (1, 2)      # Completely wrong tool selection
    }
    
    # Agent Reasoning Quality
    REASONING_RUBRIC = {
        "excellent": (9, 10),    # Clear, logical, well-structured reasoning
        "good": (7, 8),          # Generally sound reasoning with minor gaps
        "fair": (5, 6),          # Some logical flow but unclear in parts
        "poor": (3, 4),          # Flawed reasoning or significant gaps
        "incoherent": (1, 2)     # No clear reasoning or completely illogical
    }
    
    # Response Quality and Relevance
    RESPONSE_QUALITY_RUBRIC = {
        "excellent": (9, 10),    # Comprehensive, accurate, directly answers query
        "good": (7, 8),          # Good answer with minor omissions
        "adequate": (5, 6),      # Partially answers query, some inaccuracies
        "poor": (3, 4),          # Limited answer, significant issues
        "inadequate": (1, 2)     # Doesn't answer query or major inaccuracies
    }

class SessionLogParser:
    """Parses session log files and extracts structured data"""
    
    @staticmethod
    def parse_log_file(file_path: str) -> Dict[str, Any]:
        """Parse a session log file and extract all components"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract components using regex patterns
            parsed_data = {
                'file_path': file_path,
                'timestamp': SessionLogParser._extract_timestamp(content),
                'session_id': SessionLogParser._extract_session_id(content),
                'user_query': SessionLogParser._extract_user_query(content),
                'guardrail_result': SessionLogParser._extract_guardrail_result(content),
                'rag_context': SessionLogParser._extract_rag_context(content),
                'agent_reasoning': SessionLogParser._extract_agent_reasoning(content),
                'tool_executions': SessionLogParser._extract_tool_executions(content),
                'final_response': SessionLogParser._extract_final_response(content),
                'processing_time': SessionLogParser._extract_processing_time(content)
            }
            
            return parsed_data
            
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return {}
    
    @staticmethod
    def _extract_timestamp(content: str) -> str:
        match = re.search(r'Timestamp: (.+)', content)
        return match.group(1) if match else ""
    
    @staticmethod
    def _extract_session_id(content: str) -> str:
        match = re.search(r'Session ID: (.+)', content)
        return match.group(1) if match else ""
    
    @staticmethod
    def _extract_user_query(content: str) -> str:
        pattern = r'USER QUERY\n={80}\n(.*?)\n={80}'
        match = re.search(pattern, content, re.DOTALL)
        return match.group(1).strip() if match else ""
    
    @staticmethod
    def _extract_guardrail_result(content: str) -> Dict[str, Any]:
        pattern = r'GUARDRAIL VALIDATION\n={80}\n(.*?)\n={80}'
        match = re.search(pattern, content, re.DOTALL)
        if not match:
            return {}
        
        guardrail_text = match.group(1)
        return {
            'status': SessionLogParser._extract_field(guardrail_text, 'Status'),
            'validation_time': SessionLogParser._extract_field(guardrail_text, 'Validation Time'),
            'checks_performed': SessionLogParser._extract_field(guardrail_text, 'Checks Performed'),
            'security_verdict': SessionLogParser._extract_field(guardrail_text, 'Security Verdict')
        }
    
    @staticmethod
    def _extract_rag_context(content: str) -> List[Dict[str, Any]]:
        pattern = r'RAG CONTEXT RETRIEVED\n={80}\n(.*?)\n={80}'
        match = re.search(pattern, content, re.DOTALL)
        if not match:
            return []
        
        rag_text = match.group(1)
        documents = []
        
        # Extract individual documents
        doc_pattern = r'Document (\d+) \(Score: ([\d.]+)\):\nSource: (.+?)\nContent: (.+?)\nLabels: (.+?)(?=\n\nDocument|\n={80}|$)'
        doc_matches = re.findall(doc_pattern, rag_text, re.DOTALL)
        
        for doc_match in doc_matches:
            documents.append({
                'document_id': int(doc_match[0]),
                'score': float(doc_match[1]),
                'source': doc_match[2].strip(),
                'content': doc_match[3].strip(),
                'labels': [label.strip() for label in doc_match[4].split(',')]
            })
        
        return documents
    
    @staticmethod
    def _extract_agent_reasoning(content: str) -> List[Dict[str, Any]]:
        pattern = r'AGENT REASONING & SCRATCHPAD\n={80}\n(.*?)\n={80}'
        match = re.search(pattern, content, re.DOTALL)
        if not match:
            return []
        
        reasoning_text = match.group(1)
        steps = []
        
        # Extract reasoning steps
        step_pattern = r'Step (\d+) \[(.+?)\]:\n(.*?)(?=\nStep|\n-{50}|$)'
        step_matches = re.findall(step_pattern, reasoning_text, re.DOTALL)
        
        for step_match in step_matches:
            steps.append({
                'step_number': int(step_match[0]),
                'step_type': step_match[1],
                'content': step_match[2].strip()
            })
        
        return steps
    
    @staticmethod
    def _extract_tool_executions(content: str) -> List[Dict[str, Any]]:
        pattern = r'TOOL EXECUTIONS\n={80}\n(.*?)\n={80}'
        match = re.search(pattern, content, re.DOTALL)
        if not match:
            return []
        
        tools_text = match.group(1)
        executions = []
        
        # Extract tool executions (this might need adjustment based on actual format)
        exec_pattern = r'Tool Execution (\d+):\n(.*?)(?=\nTool Execution|\n-{50}|$)'
        exec_matches = re.findall(exec_pattern, tools_text, re.DOTALL)
        
        for exec_match in exec_matches:
            executions.append({
                'execution_number': int(exec_match[0]),
                'details': exec_match[1].strip()
            })
        
        return executions
    
    @staticmethod
    def _extract_final_response(content: str) -> str:
        pattern = r'FINAL RESPONSE\n={80}\n(.*?)\n={80}'
        match = re.search(pattern, content, re.DOTALL)
        return match.group(1).strip() if match else ""
    
    @staticmethod
    def _extract_processing_time(content: str) -> str:
        match = re.search(r'Processing Time: (.+?) seconds', content)
        return match.group(1) if match else ""
    
    @staticmethod
    def _extract_field(text: str, field_name: str) -> str:
        pattern = f'{field_name}: (.+?)(?=\n|$)'
        match = re.search(pattern, text)
        return match.group(1).strip() if match else ""

class LLMJudge:
    """Uses Gemini Pro to evaluate various aspects of the session"""
    
    def __init__(self):
        self.model = genai.GenerativeModel(
            'gemini-pro',
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
    
    def evaluate_rag_relevance(self, user_query: str, documents: List[Dict]) -> Dict[str, Any]:
        """Evaluate how relevant each RAG document is to the user query"""
        
        prompt = f"""
You are an expert evaluator assessing the relevance of retrieved documents to a user query.

USER QUERY: "{user_query}"

EVALUATION TASK: Rate each document's relevance to the user query on a scale of 1-10.

SCORING CRITERIA:
- 9-10 (Excellent): Directly addresses the query, highly relevant content
- 7-8 (Good): Relevant with minor tangential content
- 5-6 (Fair): Somewhat relevant but missing key aspects
- 3-4 (Poor): Limited relevance, mostly off-topic
- 1-2 (Irrelevant): No relevance to the query

DOCUMENTS TO EVALUATE:
"""
        
        for i, doc in enumerate(documents):
            prompt += f"""
Document {doc['document_id']} (Original Score: {doc['score']}):
Source: {doc['source']}
Content: {doc['content'][:500]}...
Labels: {', '.join(doc['labels'])}

"""
        
        prompt += """
For each document, provide:
1. Relevance score (1-10)
2. Brief justification (1-2 sentences)
3. Key matching concepts

Format your response as JSON:
{
    "overall_assessment": "brief overall assessment",
    "document_evaluations": [
        {
            "document_id": 1,
            "relevance_score": 8,
            "justification": "explanation",
            "key_concepts": ["concept1", "concept2"]
        }
    ]
}
"""
        
        try:
            response = self.model.generate_content(prompt)
            return self._parse_json_response(response.text)
        except Exception as e:
            print(f"Error in RAG evaluation: {e}")
            return {"error": str(e)}
    
    def evaluate_tool_selection(self, user_query: str, tool_executions: List[Dict], expected_tools: List[str] = None) -> Dict[str, Any]:
        """Evaluate whether the right tools were selected for the query"""
        
        executed_tools = [exec_detail.get('tool_name', 'Unknown') for exec_detail in tool_executions]
        
        prompt = f"""
You are an expert evaluator assessing tool selection for an AI agent system.

USER QUERY: "{user_query}"

AVAILABLE TOOLS:
- normal_content_tool: Answer questions using RAG knowledge base
- web_search_tool: Search internet for current/recent information
- document_export_tool: Export content as PDF, DOCX, or PPTX
- send_email_tool: Send emails with content or documents
- video_generation_tool: Create educational videos (2-2.5 minutes)
- flashcard_generation_tool: Create interactive study flashcards
- interactive_quiz_tool: Create quizzes with scoring and feedback

TOOLS EXECUTED: {executed_tools}

TOOL EXECUTION DETAILS:
"""
        
        for exec_detail in tool_executions:
            prompt += f"- {exec_detail}\n"
        
        prompt += """
EVALUATION CRITERIA:
- 9-10 (Perfect): All tools selected are optimal and necessary
- 7-8 (Good): Most tools appropriate, minor inefficiencies
- 5-6 (Adequate): Generally correct tools with some issues
- 3-4 (Poor): Wrong tools selected or missing critical tools
- 1-2 (Incorrect): Completely wrong tool selection

Evaluate:
1. Were the selected tools appropriate for the query?
2. Are there any missing tools that should have been used?
3. Are there any unnecessary tools that were used?
4. Overall tool selection score (1-10)

Format as JSON:
{
    "tool_selection_score": 8,
    "appropriate_tools": ["tool1", "tool2"],
    "missing_tools": ["tool3"],
    "unnecessary_tools": [],
    "justification": "explanation",
    "recommendations": "suggestions for improvement"
}
"""
        
        try:
            response = self.model.generate_content(prompt)
            return self._parse_json_response(response.text)
        except Exception as e:
            print(f"Error in tool selection evaluation: {e}")
            return {"error": str(e)}
    
    def evaluate_agent_reasoning(self, user_query: str, reasoning_steps: List[Dict], tool_executions: List[Dict], final_response: str) -> Dict[str, Any]:
        """Evaluate the quality of agent reasoning and decision-making"""
        
        prompt = f"""
You are an expert evaluator assessing the reasoning quality of an AI agent.

USER QUERY: "{user_query}"

AGENT REASONING STEPS:
"""
        
        for step in reasoning_steps:
            prompt += f"Step {step['step_number']} [{step['step_type']}]: {step['content']}\n"
        
        prompt += f"""
TOOL EXECUTIONS:
"""
        for exec_detail in tool_executions:
            prompt += f"- {exec_detail}\n"
        
        prompt += f"""
FINAL RESPONSE: "{final_response}"

EVALUATION CRITERIA:
- 9-10 (Excellent): Clear, logical, well-structured reasoning
- 7-8 (Good): Generally sound reasoning with minor gaps
- 5-6 (Fair): Some logical flow but unclear in parts
- 3-4 (Poor): Flawed reasoning or significant gaps
- 1-2 (Incoherent): No clear reasoning or completely illogical

Assess:
1. Logical flow from query to response
2. Coherence of reasoning steps
3. Appropriateness of decision-making
4. Connection between reasoning and tool selection
5. Overall reasoning quality score (1-10)

Format as JSON:
{
    "reasoning_score": 8,
    "logical_flow": "assessment",
    "coherence": "assessment",
    "decision_quality": "assessment",
    "reasoning_tool_alignment": "assessment",
    "strengths": ["strength1", "strength2"],
    "weaknesses": ["weakness1"],
    "overall_assessment": "summary"
}
"""
        
        try:
            response = self.model.generate_content(prompt)
            return self._parse_json_response(response.text)
        except Exception as e:
            print(f"Error in reasoning evaluation: {e}")
            return {"error": str(e)}
    
    def evaluate_response_quality(self, user_query: str, final_response: str, rag_context: List[Dict]) -> Dict[str, Any]:
        """Evaluate the quality and relevance of the final response"""
        
        context_summary = "\n".join([f"Doc {doc['document_id']}: {doc['content'][:200]}..." for doc in rag_context[:3]])
        
        prompt = f"""
You are an expert evaluator assessing the quality of an AI assistant's response.

USER QUERY: "{user_query}"

AVAILABLE CONTEXT (Top 3 documents):
{context_summary}

FINAL RESPONSE: "{final_response}"

EVALUATION CRITERIA:
- 9-10 (Excellent): Comprehensive, accurate, directly answers query
- 7-8 (Good): Good answer with minor omissions
- 5-6 (Adequate): Partially answers query, some inaccuracies
- 3-4 (Poor): Limited answer, significant issues
- 1-2 (Inadequate): Doesn't answer query or major inaccuracies

Assess:
1. Relevance to the user query
2. Accuracy of information
3. Completeness of the answer
4. Clarity and coherence
5. Use of available context
6. Overall response quality score (1-10)

Format as JSON:
{
    "response_quality_score": 8,
    "relevance": "assessment",
    "accuracy": "assessment", 
    "completeness": "assessment",
    "clarity": "assessment",
    "context_utilization": "assessment",
    "strengths": ["strength1", "strength2"],
    "areas_for_improvement": ["area1"],
    "overall_assessment": "summary"
}
"""
        
        try:
            response = self.model.generate_content(prompt)
            return self._parse_json_response(response.text)
        except Exception as e:
            print(f"Error in response quality evaluation: {e}")
            return {"error": str(e)}
    
    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, handling potential formatting issues"""
        try:
            # Try to extract JSON from markdown code blocks
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Look for JSON-like content
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = response_text
            
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            print(f"Raw response: {response_text}")
            return {"error": "Failed to parse JSON response", "raw_response": response_text}

class SessionEvaluator:
    """Main evaluator class that orchestrates the evaluation process"""
    
    def __init__(self, logs_directory: str = "agent_data/session_logs"):
        self.logs_directory = Path(logs_directory)
        self.parser = SessionLogParser()
        self.judge = LLMJudge()
        self.criteria = EvaluationCriteria()
    
    def evaluate_all_sessions(self) -> Dict[str, Any]:
        """Evaluate all session log files in the directory"""
        
        log_files = list(self.logs_directory.glob("*.txt"))
        if not log_files:
            print(f"No log files found in {self.logs_directory}")
            return {}
        
        print(f"Found {len(log_files)} session log files to evaluate")
        
        all_evaluations = {}
        summary_stats = {
            "total_sessions": len(log_files),
            "successful_evaluations": 0,
            "failed_evaluations": 0,
            "average_scores": {},
            "evaluation_timestamp": datetime.now().isoformat()
        }
        
        for log_file in log_files:
            print(f"\n{'='*50}")
            print(f"Evaluating: {log_file.name}")
            print(f"{'='*50}")
            
            try:
                # Parse the session log
                session_data = self.parser.parse_log_file(str(log_file))
                if not session_data:
                    print(f"Failed to parse {log_file.name}")
                    summary_stats["failed_evaluations"] += 1
                    continue
                
                # Run comprehensive evaluation
                evaluation = self.evaluate_session(session_data)
                all_evaluations[log_file.name] = evaluation
                summary_stats["successful_evaluations"] += 1
                
                # Print summary for this session
                self._print_session_summary(log_file.name, evaluation)
                
            except Exception as e:
                print(f"Error evaluating {log_file.name}: {e}")
                summary_stats["failed_evaluations"] += 1
        
        # Calculate overall statistics
        summary_stats["average_scores"] = self._calculate_average_scores(all_evaluations)
        
        # Save evaluation results
        results = {
            "summary": summary_stats,
            "detailed_evaluations": all_evaluations
        }
        
        self._save_evaluation_results(results)
        
        return results
    
    def evaluate_session(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single session across all metrics"""
        
        user_query = session_data.get('user_query', '')
        rag_context = session_data.get('rag_context', [])
        tool_executions = session_data.get('tool_executions', [])
        reasoning_steps = session_data.get('agent_reasoning', [])
        final_response = session_data.get('final_response', '')
        
        evaluation_results = {
            "session_metadata": {
                "file_path": session_data.get('file_path', ''),
                "timestamp": session_data.get('timestamp', ''),
                "session_id": session_data.get('session_id', ''),
                "processing_time": session_data.get('processing_time', ''),
                "user_query": user_query
            }
        }
        
        print("🔍 Evaluating RAG Context Relevance...")
        time.sleep(1)  # Rate limiting
        evaluation_results["rag_evaluation"] = self.judge.evaluate_rag_relevance(user_query, rag_context)
        
        print("🛠️ Evaluating Tool Selection...")
        time.sleep(1)
        evaluation_results["tool_evaluation"] = self.judge.evaluate_tool_selection(user_query, tool_executions)
        
        print("🧠 Evaluating Agent Reasoning...")
        time.sleep(1)
        evaluation_results["reasoning_evaluation"] = self.judge.evaluate_agent_reasoning(
            user_query, reasoning_steps, tool_executions, final_response
        )
        
        print("💬 Evaluating Response Quality...")
        time.sleep(1)
        evaluation_results["response_evaluation"] = self.judge.evaluate_response_quality(
            user_query, final_response, rag_context
        )
        
        # Calculate overall scores
        evaluation_results["overall_scores"] = self._calculate_overall_scores(evaluation_results)
        
        return evaluation_results
    
    def _calculate_overall_scores(self, evaluation: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall scores from individual evaluations"""
        
        scores = {}
        
        # RAG relevance (average of all document scores)
        rag_eval = evaluation.get("rag_evaluation", {})
        if "document_evaluations" in rag_eval:
            doc_scores = [doc.get("relevance_score", 0) for doc in rag_eval["document_evaluations"]]
            scores["rag_relevance"] = sum(doc_scores) / len(doc_scores) if doc_scores else 0
        
        # Tool selection
        tool_eval = evaluation.get("tool_evaluation", {})
        scores["tool_selection"] = tool_eval.get("tool_selection_score", 0)
        
        # Reasoning quality
        reasoning_eval = evaluation.get("reasoning_evaluation", {})
        scores["reasoning_quality"] = reasoning_eval.get("reasoning_score", 0)
        
        # Response quality
        response_eval = evaluation.get("response_evaluation", {})
        scores["response_quality"] = response_eval.get("response_quality_score", 0)
        
        # Overall weighted average (you can adjust weights as needed)
        weights = {"rag_relevance": 0.25, "tool_selection": 0.25, "reasoning_quality": 0.25, "response_quality": 0.25}
        
        weighted_sum = sum(scores[metric] * weight for metric, weight in weights.items() if metric in scores)
        scores["overall_score"] = weighted_sum
        
        return scores
    
    def _calculate_average_scores(self, all_evaluations: Dict[str, Any]) -> Dict[str, float]:
        """Calculate average scores across all sessions"""
        
        if not all_evaluations:
            return {}
        
        metrics = ["rag_relevance", "tool_selection", "reasoning_quality", "response_quality", "overall_score"]
        averages = {}
        
        for metric in metrics:
            scores = []
            for evaluation in all_evaluations.values():
                overall_scores = evaluation.get("overall_scores", {})
                if metric in overall_scores:
                    scores.append(overall_scores[metric])
            
            if scores:
                averages[metric] = sum(scores) / len(scores)
        
        return averages
    
    def _print_session_summary(self, filename: str, evaluation: Dict[str, Any]):
        """Print a summary of the evaluation results"""
        
        overall_scores = evaluation.get("overall_scores", {})
        
        print(f"\n📊 EVALUATION SUMMARY for {filename}")
        print(f"{'='*60}")
        print(f"RAG Relevance Score:    {overall_scores.get('rag_relevance', 0):.2f}/10")
        print(f"Tool Selection Score:   {overall_scores.get('tool_selection', 0):.2f}/10")
        print(f"Reasoning Quality:      {overall_scores.get('reasoning_quality', 0):.2f}/10")
        print(f"Response Quality:       {overall_scores.get('response_quality', 0):.2f}/10")
        print(f"{'─'*60}")
        print(f"OVERALL SCORE:          {overall_scores.get('overall_score', 0):.2f}/10")
        
        # Quick insights
        rag_eval = evaluation.get("rag_evaluation", {})
        if "overall_assessment" in rag_eval:
            print(f"\n🎯 RAG Assessment: {rag_eval['overall_assessment']}")
        
        tool_eval = evaluation.get("tool_evaluation", {})
        if "justification" in tool_eval:
            print(f"🛠️ Tool Selection: {tool_eval['justification']}")
    
    def _save_evaluation_results(self, results: Dict[str, Any]):
        """Save evaluation results to file"""
        
        # Create evaluation directory
        eval_dir = Path("evaluation/results")
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = eval_dir / f"session_evaluation_{timestamp}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Evaluation results saved to: {results_file}")
        
        # Save summary report
        summary_file = eval_dir / f"evaluation_summary_{timestamp}.txt"
        self._generate_summary_report(results, summary_file)
        
        print(f"📋 Summary report saved to: {summary_file}")
    
    def _generate_summary_report(self, results: Dict[str, Any], output_file: Path):
        """Generate a human-readable summary report"""
        
        summary = results.get("summary", {})
        detailed = results.get("detailed_evaluations", {})
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("SESSION LOG EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Overall statistics
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Sessions Evaluated: {summary.get('total_sessions', 0)}\n")
            f.write(f"Successful Evaluations: {summary.get('successful_evaluations', 0)}\n")
            f.write(f"Failed Evaluations: {summary.get('failed_evaluations', 0)}\n")
            f.write(f"Evaluation Date: {summary.get('evaluation_timestamp', 'Unknown')}\n\n")
            
            # Average scores
            avg_scores = summary.get("average_scores", {})
            if avg_scores:
                f.write("AVERAGE SCORES ACROSS ALL SESSIONS\n")
                f.write("-" * 35 + "\n")
                for metric, score in avg_scores.items():
                    f.write(f"{metric.replace('_', ' ').title():<25}: {score:.2f}/10\n")
                f.write("\n")
            
            # Individual session details
            f.write("INDIVIDUAL SESSION DETAILS\n")
            f.write("-" * 30 + "\n\n")
            
            for filename, evaluation in detailed.items():
                metadata = evaluation.get("session_metadata", {})
                scores = evaluation.get("overall_scores", {})
                
                f.write(f"File: {filename}\n")
                f.write(f"Query: {metadata.get('user_query', 'Unknown')}\n")
                f.write(f"Processing Time: {metadata.get('processing_time', 'Unknown')} seconds\n")
                f.write("Scores:\n")
                for metric, score in scores.items():
                    f.write(f"  {metric.replace('_', ' ').title()}: {score:.2f}/10\n")
                f.write("\n" + "─" * 60 + "\n\n")

def main():
    """Main function to run the evaluation"""
    
    print("🚀 Starting Session Log Evaluation")
    print("=" * 50)
    
    # Check for required API key
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ Error: GEMINI_API_KEY environment variable not set")
        return
    
    # Initialize evaluator
    evaluator = SessionEvaluator()
    
    # Run evaluation on all sessions
    results = evaluator.evaluate_all_sessions()
    
    if results:
        print("\n🎉 Evaluation completed successfully!")
        summary = results.get("summary", {})
        avg_scores = summary.get("average_scores", {})
        
        if avg_scores:
            print("\n📈 OVERALL PERFORMANCE SUMMARY")
            print("=" * 40)
            for metric, score in avg_scores.items():
                print(f"{metric.replace('_', ' ').title():<25}: {score:.2f}/10")
    else:
        print("\n❌ No evaluation results generated")

if __name__ == "__main__":
    main()
