# # # ============================================
# # # File: utils/orchestrator_langgraph.py
# # # ============================================
# # from typing import TypedDict, Annotated, Sequence, Literal
# # from langgraph.graph import StateGraph, END
# # from langgraph.prebuilt import ToolNode
# # from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
# # from langchain_nvidia_ai_endpoints import ChatNVIDIA
# # from langchain_core.tools import tool
# # import os
# # import json
# # from operator import add

# # from dotenv import load_dotenv
# # load_dotenv()
# # utils/orchestrator_graph.py

# # import os
# # import json
# # from dotenv import load_dotenv
# # from langchain.agents import create_agent
# # # from langchain.agents.agent_types import AgentType
# # from langchain_community.memory import ConversationBufferMemory
# # from langchain_nvidia import ChatNVIDIA  # or ChatOpenAI if preferred
# # from utils.agent_tools import rag_tool, document_tool, email_tool

# # load_dotenv()


# # def build_agent():
# #     """
# #     Create the main orchestrator agent that can call multiple tools.
# #     """

# #     # Initialize model (replace ChatNVIDIA with ChatOpenAI if needed)
# #     llm = ChatNVIDIA(
# #         model="meta/llama3-70b-instruct",
# #         api_key=os.getenv("NVIDIA_API_KEY"),
# #         temperature=0.3,
# #     )

# #     # Register tools
# #     tools = [rag_tool, document_tool, email_tool]

# #     # Add chat memory
# #     memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# #     # Create modern ReAct-style agent
# #     agent = create_agent(
# #         llm=llm,
# #         tools=tools,
# #         memory=memory,
# #         verbose=True,
# #     )

# #     return agent


# # def run_agent(agent, query: str, history: list):
# #     """
# #     Run the agent with full conversation history.
# #     """

# #     try:
# #         # Rebuild memory context from previous turns
# #         memory = agent.memory
# #         for msg in history:
# #             if msg["role"] == "user":
# #                 memory.chat_memory.add_user_message(msg["content"])
# #             else:
# #                 memory.chat_memory.add_ai_message(msg["content"])

# #         # Run the agent (LangChain modern API)
# #         result = agent.invoke({"input": query})

# #         # Depending on LC version, result may be dict or string
# #         if isinstance(result, dict):
# #             output = (
# #                 result.get("output")
# #                 or result.get("agent")
# #                 or str(result)
# #             )
# #         else:
# #             output = str(result)

# #         # Save new messages to memory
# #         memory.chat_memory.add_user_message(query)
# #         memory.chat_memory.add_ai_message(output)

# #         return output

# #     except Exception as e:
# #         return f"❌ Error executing plan: {str(e)}"

# # # ============================================
# # # AGENT STATE DEFINITION
# # # ============================================
# # class AgentState(TypedDict):
# #     """State that flows through the graph"""
# #     messages: Annotated[Sequence[BaseMessage], add]
# #     session_id: str
# #     username: str

# # # ============================================
# # # ENHANCED TOOLS WITH BETTER INTEGRATION
# # # ============================================

# # @tool
# # def rag_tool(query: str, context: str = "") -> str:
# #     """
# #     Retrieve contextual information related to the query.
    
# #     Args:
# #         query: The user's question or request
# #         context: Additional conversation context (optional)
    
# #     Returns:
# #         Retrieved information relevant to the query
# #     """
# #     try:
# #         # In production, this would call your actual RAG system
# #         return (
# #             f"[RAG_Tool] Retrieved information for query: '{query}'\n"
# #             f"Context analyzed: {len(context)} characters\n"
# #             f"Key findings: [Your RAG results would appear here]\n"
# #             f"Sources: document1.pdf, document2.pdf"
# #         )
# #     except Exception as e:
# #         return f"[RAG_Tool] Error: {str(e)}"


# # @tool
# # def document_tool(content: str, document_type: str = "pdf", title: str = "Generated Document") -> str:
# #     """
# #     Generate a document (PDF, DOCX, or PPTX) with the provided content.
    
# #     Args:
# #         content: The text content to include in the document
# #         document_type: Type of document ('pdf', 'docx', or 'pptx')
# #         title: Title for the document
    
# #     Returns:
# #         Path to the generated document
# #     """
# #     try:
# #         from tools.document_generation_tool import DocumentGenerationTool
        
# #         doc_gen = DocumentGenerationTool(output_directory="generated_documents")
# #         result = doc_gen.run({
# #             "content": content,
# #             "document_type": document_type,
# #             "title": title
# #         })
# #         return result
# #     except Exception as e:
# #         return f"[DocumentGenerationTool] Error: {str(e)}"


# # @tool
# # def email_tool(filename: str, recipient: str = "user@example.com", subject: str = "Generated Document") -> str:
# #     """
# #     Send an email with the generated document attached.
    
# #     Args:
# #         filename: Path to the document to attach
# #         recipient: Email recipient address
# #         subject: Email subject line
    
# #     Returns:
# #         Confirmation message
# #     """
# #     try:
# #         # In production, integrate with actual email service
# #         return (
# #             f"[EmailAutomationTool] Email sent successfully!\n"
# #             f"To: {recipient}\n"
# #             f"Subject: {subject}\n"
# #             f"Attachment: {filename}"
# #         )
# #     except Exception as e:
# #         return f"[EmailAutomationTool] Error: {str(e)}"


# # # ============================================
# # # LANGGRAPH ORCHESTRATOR
# # # ============================================

# # class LangGraphOrchestrator:
# #     """LLM-driven agent orchestrator using LangGraph"""
    
# #     def __init__(self):
# #         self.llm = ChatNVIDIA(
# #             model="meta/llama3-70b-instruct",
# #             temperature=0.2,
# #             api_key=os.getenv("NVIDIA_API_KEY")
# #         )
# #         self.tools = [rag_tool, document_tool, email_tool]
# #         self.graph = self._create_graph()
    
# #     def _create_graph(self):
# #         """Build the LangGraph workflow"""
        
# #         # Bind tools to LLM
# #         llm_with_tools = self.llm.bind_tools(self.tools)
        
# #         # Create tool execution node
# #         tool_node = ToolNode(self.tools)
        
# #         # Define the agent node (LLM decides what to do)
# #         def call_model(state: AgentState):
# #             """Invoke LLM to decide next action"""
# #             messages = state["messages"]
# #             response = llm_with_tools.invoke(messages)
# #             return {"messages": [response]}
        
# #         # Define routing logic
# #         def should_continue(state: AgentState) -> Literal["tools", "end"]:
# #             """Determine if we should use tools or end"""
# #             last_message = state["messages"][-1]
            
# #             # If the LLM makes a tool call, route to tools
# #             if hasattr(last_message, "tool_calls") and last_message.tool_calls:
# #                 return "tools"
            
# #             # Otherwise, we're done
# #             return "end"
        
# #         # Build the graph
# #         workflow = StateGraph(AgentState)
        
# #         # Add nodes
# #         workflow.add_node("agent", call_model)
# #         workflow.add_node("tools", tool_node)
        
# #         # Set entry point
# #         workflow.set_entry_point("agent")
        
# #         # Add edges
# #         workflow.add_conditional_edges(
# #             "agent",
# #             should_continue,
# #             {
# #                 "tools": "tools",
# #                 "end": END
# #             }
# #         )
        
# #         # After tools, always go back to agent to decide next step
# #         workflow.add_edge("tools", "agent")
        
# #         return workflow.compile()
    
# #     def run(self, query: str, history: list, session_id: str, username: str) -> str:
# #         """
# #         Execute the agent with conversation history
        
# #         Args:
# #             query: User's current message
# #             history: List of previous messages [{"role": "user/assistant", "content": "..."}]
# #             session_id: Session identifier
# #             username: Username
            
# #         Returns:
# #             Agent's response
# #         """
# #         try:
# #             # Convert history to LangChain messages
# #             messages = []
# #             for msg in history:
# #                 if msg["role"] == "user":
# #                     messages.append(HumanMessage(content=msg["content"]))
# #                 elif msg["role"] == "assistant":
# #                     messages.append(AIMessage(content=msg["content"]))
            
# #             # Add current query
# #             messages.append(HumanMessage(content=query))
            
# #             # Create initial state
# #             initial_state = {
# #                 "messages": messages,
# #                 "session_id": session_id,
# #                 "username": username
# #             }
            
# #             # Run the graph
# #             final_state = self.graph.invoke(initial_state)
            
# #             # Extract the final response
# #             final_messages = final_state["messages"]
            
# #             # Get the last AI message (skip tool messages)
# #             for msg in reversed(final_messages):
# #                 if isinstance(msg, AIMessage) and not hasattr(msg, "tool_calls"):
# #                     return msg.content
# #                 elif isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and not msg.tool_calls:
# #                     return msg.content
            
# #             # Fallback
# #             return "I've processed your request."
            
# #         except Exception as e:
# #             print(f"Error in LangGraph orchestrator: {e}")
# #             import traceback
# #             traceback.print_exc()
# #             return f"❌ Error: {str(e)}"

# # # ============================================
# # # FACTORY FUNCTION
# # # ============================================

# # _orchestrator_instance = None

# # def get_orchestrator() -> LangGraphOrchestrator:
# #     """Get or create singleton orchestrator instance"""
# #     global _orchestrator_instance
# #     if _orchestrator_instance is None:
# #         _orchestrator_instance = LangGraphOrchestrator()
# #     return _orchestrator_instance


# from typing import TypedDict, Annotated, Sequence, Literal
# from langgraph.graph import StateGraph, END
# from langgraph.prebuilt import ToolNode
# from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
# from langchain_nvidia_ai_endpoints import ChatNVIDIA
# from langchain_core.tools import tool
# import os
# import json
# from operator import add

# from dotenv import load_dotenv
# load_dotenv()

# # ============================================
# # AGENT STATE DEFINITION
# # ============================================
# class AgentState(TypedDict):
#     """State that flows through the graph"""
#     messages: Annotated[Sequence[BaseMessage], add]
#     session_id: str
#     username: str

# # ============================================
# # ENHANCED TOOLS WITH BETTER INTEGRATION
# # ============================================

# @tool
# def rag_tool(query: str, context: str = "") -> str:
#     """
#     Retrieve contextual information related to the query.
    
#     Args:
#         query: The user's question or request
#         context: Additional conversation context (optional)
    
#     Returns:
#         Retrieved information relevant to the query
#     """
#     try:
#         # In production, this would call your actual RAG system
#         return (
#             f"[RAG_Tool] Retrieved information for query: '{query}'\n"
#             f"Context analyzed: {len(context)} characters\n"
#             f"Key findings: [Your RAG results would appear here]\n"
#             f"Sources: document1.pdf, document2.pdf"
#         )
#     except Exception as e:
#         return f"[RAG_Tool] Error: {str(e)}"


# @tool
# def document_tool(content: str, document_type: str = "pdf", title: str = "Generated Document") -> str:
#     """
#     Generate a document (PDF, DOCX, or PPTX) with the provided content.
    
#     Args:
#         content: The text content to include in the document
#         document_type: Type of document ('pdf', 'docx', or 'pptx')
#         title: Title for the document
    
#     Returns:
#         Path to the generated document
#     """
#     try:
#         from tools.document_generation_tool import DocumentGenerationTool
        
#         doc_gen = DocumentGenerationTool(output_directory="generated_documents")
#         result = doc_gen.run({
#             "content": content,
#             "document_type": document_type,
#             "title": title
#         })
#         return result
#     except Exception as e:
#         return f"[DocumentGenerationTool] Error: {str(e)}"


# @tool
# def email_tool(filename: str, recipient: str = "user@example.com", subject: str = "Generated Document") -> str:
#     """
#     Send an email with the generated document attached.
    
#     Args:
#         filename: Path to the document to attach
#         recipient: Email recipient address
#         subject: Email subject line
    
#     Returns:
#         Confirmation message
#     """
#     try:
#         # In production, integrate with actual email service
#         return (
#             f"[EmailAutomationTool] Email sent successfully!\n"
#             f"To: {recipient}\n"
#             f"Subject: {subject}\n"
#             f"Attachment: {filename}"
#         )
#     except Exception as e:
#         return f"[EmailAutomationTool] Error: {str(e)}"


# # ============================================
# # LANGGRAPH ORCHESTRATOR
# # ============================================

# class LangGraphOrchestrator:
#     """LLM-driven agent orchestrator using LangGraph"""
    
#     def __init__(self):
#         self.llm = ChatNVIDIA(
#             model="mistralai/mixtral-8x7b-instruct-v0.1",
#             temperature=0.2,
#             api_key=os.getenv("NVIDIA_API_KEY")
#         )
#         self.tools = [rag_tool, document_tool, email_tool]
#         self.graph = self._create_graph()
    
#     def _create_graph(self):
#         """Build the LangGraph workflow"""
        
#         # Bind tools to LLM
#         llm_with_tools = self.llm.bind_tools(self.tools)
        
#         # Create tool execution node
#         tool_node = ToolNode(self.tools)
        
#         # Define the agent node (LLM decides what to do)
#         def call_model(state: AgentState):
#             """Invoke LLM to decide next action"""
#             from langchain_core.messages import SystemMessage
            
#             messages = state["messages"]
            
#             # Add system prompt to guide tool usage
#             system_prompt = SystemMessage(content="""You are a helpful AI assistant with access to the following tools:

# 1. **rag_tool**: Use this to search for and retrieve information from documents/knowledge base
# 2. **document_tool**: Use this to generate PDF, DOCX, or PPTX documents with content
# 3. **email_tool**: Use this to send emails with attachments

# IMPORTANT INSTRUCTIONS:
# - When users ask you to "create a report", "generate a document", "make a PDF", or similar requests, you MUST call the document_tool
# - When users ask to "search", "find information", or "look up", use the rag_tool first
# - When users ask to "email" or "send", use the email_tool
# - You can chain tools: For example, if asked to "research X and create a report", first call rag_tool, then use its output to call document_tool
# - Always call tools when appropriate - don't just describe what you would do

# Examples:
# - "Create a report about AI" → Call document_tool with content about AI
# - "Search for ML info and generate a PDF" → Call rag_tool, then document_tool with the results
# - "Make a document about climate change" → Call document_tool
# - "Find info about quantum computing" → Call rag_tool

# Be proactive in using tools to complete the user's request.""")
            
#             # Insert system message at the beginning if not already present
#             if not messages or not isinstance(messages[0], SystemMessage):
#                 messages = [system_prompt] + messages
            
#             response = llm_with_tools.invoke(messages)
#             return {"messages": [response]}
        
#         # Define routing logic
#         def should_continue(state: AgentState) -> Literal["tools", "end"]:
#             """Determine if we should use tools or end"""
#             last_message = state["messages"][-1]
#             print(f"Last message should continue in orchestrator: {last_message}")
            
#             # If the LLM makes a tool call, route to tools
#             if hasattr(last_message, "tool_calls") and last_message.tool_calls:
#                 return "tools"
            
#             # Otherwise, we're done
#             return "end"
        
#         # Build the graph
#         workflow = StateGraph(AgentState)
        
#         # Add nodes
#         workflow.add_node("agent", call_model)
#         workflow.add_node("tools", tool_node)
        
#         # Set entry point
#         workflow.set_entry_point("agent")
        
#         # Add edges
#         workflow.add_conditional_edges(
#             "agent",
#             should_continue,
#             {
#                 "tools": "tools",
#                 "end": END
#             }
#         )
        
#         # After tools, always go back to agent to decide next step
#         workflow.add_edge("tools", "agent")
        
#         return workflow.compile()
    
#     def run(self, query: str, history: list, session_id: str, username: str) -> str:
#         """
#         Execute the agent with conversation history
        
#         Args:
#             query: User's current message
#             history: List of previous messages [{"role": "user/assistant", "content": "..."}]
#             session_id: Session identifier
#             username: Username
            
#         Returns:
#             Agent's response
#         """
#         try:
#             # Convert history to LangChain messages
#             messages = []
#             for msg in history:
#                 if msg["role"] == "user":
#                     messages.append(HumanMessage(content=msg["content"]))
#                 elif msg["role"] == "assistant":
#                     messages.append(AIMessage(content=msg["content"]))
            
#             # Add current query
#             messages.append(HumanMessage(content=query))
            
#             # Create initial state
#             initial_state = {
#                 "messages": messages,
#                 "session_id": session_id,
#                 "username": username
#             }
            
#             # Run the graph
#             final_state = self.graph.invoke(initial_state)
            
#             # Extract the final response
#             final_messages = final_state["messages"]
            
#             # Get the last AI message (skip tool messages)
#             for msg in reversed(final_messages):
#                 if isinstance(msg, AIMessage) and not hasattr(msg, "tool_calls"):
#                     return msg.content
#                 elif isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and not msg.tool_calls:
#                     return msg.content
            
#             # Fallback
#             return "I've processed your request."
            
#         except Exception as e:
#             print(f"Error in LangGraph orchestrator: {e}")
#             import traceback
#             traceback.print_exc()
#             return f"❌ Error: {str(e)}"


# # ============================================
# # FACTORY FUNCTION
# # ============================================

# _orchestrator_instance = None

# def get_orchestrator() -> LangGraphOrchestrator:
#     """Get or create singleton orchestrator instance"""
#     global _orchestrator_instance
#     if _orchestrator_instance is None:
#         _orchestrator_instance = LangGraphOrchestrator()
#     return _orchestrator_instance