# utils/orchestrator.py
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal, Optional

load_dotenv()

class ToolCall(BaseModel):
    name: str = Field(..., description="The name of the tool to call, e.g., 'document_tool' or 'email_tool'.")
    args: Dict[str, Any] = Field(..., description="The arguments for the tool, e.g., {'content': '...', 'title': '...'}.")

class ToolPlan(BaseModel):
    """The master plan for responding to the user."""
    rag_source: Literal["external_kb", "session_docs", "both", "none"] = Field(..., description="The knowledge base to query for RAG.")
    execution_plan: Optional[List[ToolCall]] = Field(None, description="The sequence of tools to execute. Can be empty.")
    chat_response: Optional[str] = Field(None, description="If no RAG or tools are needed, provide a direct chat response here.")

class LangChainOrchestrator:
    """LLM orchestrator for *planning* a sequence of actions."""

    def __init__(self):
        print("🔧 Initializing LangChain Orchestrator (Planner)...")
        self.llm = ChatNVIDIA(
            model="meta/llama-3.2-1b-instruct",
            temperature=0.0, # Planning should be deterministic
            api_key=os.getenv("NVIDIA_API_KEY")
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a master planner. Your job is to analyze the user's query and the provided context/history, then create a step-by-step JSON plan.
You must always output a JSON object matching the provided schema.

---
Here are your decision rules:

1.  **RAG Source (`rag_source`):**
    * If the user asks a general knowledge question (e.g., 'What is Article 14?'), set to **`external_kb`**.
    * If the user *explicitly* refers to an uploaded file (e.g., 'Summarize the document I sent', 'What is in this file?'), set to **`session_docs`**.
    * If the user asks to compare general knowledge with an uploaded file, set to **`both`**.
    * If the user is just chatting ('Hello') or asking to act on a *previous* answer ('Make a PDF of that'), set to **`none`**.

2.  **Execution Plan (`execution_plan`):**
    * List the tools to run, in order.
    * **Available Tools:**
        * `document_tool(content: str, document_type: str, title: str)`: Generates a file from text.
        * `email_tool(filename: str, recipient: str, subject: str)`: Sends a file.
    * **Placeholders:**
        * If a tool needs the RAG answer, use **`[[RAG_RESULT]]`** as the value (e.g., `"content": "[[RAG_RESULT]]"`).
        * If a tool needs the output of a previous step (e.g., the filename from `document_tool`), use **`[[STEP_N_RESULT]]`** where N is the step number (e.g., `"filename": "[[STEP_1_RESULT]]"`).
        * If a tool needs the *previous* chat answer, use **`[[LAST_ANSWER]]`** (e.g., `"content": "[[LAST_ANSWER]]"`).

3.  **Chat Response (`chat_response`):**
    * If `rag_source` is `none` AND `execution_plan` is `null` (e.g., user says "Hello"), provide a direct response here.

---
**Examples (with escaped braces):**

* User: "What is Article 14?"
    -> {{"rag_source": "external_kb", "execution_plan": null, "chat_response": null}}
* User: "Summarize the file I uploaded."
    -> {{"rag_source": "session_docs", "execution_plan": null, "chat_response": null}}
* User: "Make a PDF of that last answer and email it to me."
    -> {{"rag_source": "none", "execution_plan": [{{"name": "document_tool", "args": {{"content": "[[LAST_ANSWER]]", "title": "Summary"}}}}, {{"name": "email_tool", "args": {{"filename": "[[STEP_1_RESULT]]", "recipient": "user@example.com"}}}}], "chat_response": null}}
* User: "Thanks!"
    -> {{"rag_source": "none", "execution_plan": null, "chat_response": "You're welcome!"}}
* User: "Create a DOCX of my uploaded file's summary."
    -> {{"rag_source": "session_docs", "execution_plan": [{{"name": "document_tool", "args": {{"content": "[[RAG_RESULT]]", "document_type": "docx", "title": "File Summary"}}}}], "chat_response": null}}
"""),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "User Query: {input}\n\n(Reminder: Respond *only* with the JSON plan.)"),
        ])
        
        # This line is correct and should not change
        self.pipeline = self.prompt | self.llm.with_structured_output(ToolPlan)
        print("✅ Orchestrator (Planner) initialized!\n")

    def get_plan(self, query: str, history: list) -> ToolPlan:
        """
        Calls the LLM *once* to get a structured ToolPlan.
        Does NOT execute anything.
        """
        try:
            print(f"\n{'='*70}")
            print(f"🚀 NEW PLAN REQUEST: {query}")
            print(f"{'='*70}")

            chat_history = []
            for msg in history[-10:]:
                if msg["role"] == "user":
                    chat_history.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    chat_history.append(AIMessage(content=msg["content"]))

            print(f"   ...Calling LLM planner...")
            
            # Call the LLM *once*
            plan = self.pipeline.invoke({
                "input": query,
                "chat_history": chat_history,
            })

            print(f"✅ LLM Plan Received:")
            print(f"   - RAG Source: {plan.rag_source}")
            print(f"   - Chat Response: {plan.chat_response}")
            print(f"   - Execution Plan: {plan.execution_plan}")
            return plan

        except Exception as e:
            print(f"❌ ERROR in orchestrator get_plan: {e}")
            import traceback
            traceback.print_exc()
            # Return a default "safe" plan on error
            return ToolPlan(rag_source="none", execution_plan=None, chat_response="Sorry, I encountered an error while planning my response.")

# ============================================
# SINGLETON INSTANCE
# ============================================
_orchestrator_instance = None
def get_orchestrator() -> LangChainOrchestrator:
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = LangChainOrchestrator()
    return _orchestrator_instance

# ============================================
# RUNNER FUNCTION (PLANNER)
# ============================================
def get_unified_plan(query: str, history: list) -> ToolPlan:
    """
    Synchronously runs the orchestrator to get a unified tool plan.
    """
    orchestrator = get_orchestrator()
    return orchestrator.get_plan(query, history)