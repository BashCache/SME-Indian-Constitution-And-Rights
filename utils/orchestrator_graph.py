# # utils/orchestrator_graph.py

# import os
# import json
# from dotenv import load_dotenv
# from langchain.agents import create_agent
# # from langchain.agents.agent_types import AgentType
# from langchain_community.memory import ConversationBufferMemory
# from langchain_nvidia import ChatNVIDIA  # or ChatOpenAI if preferred
# from utils.agent_tools import rag_tool, document_tool, email_tool

# load_dotenv()


# def build_agent():
#     """
#     Create the main orchestrator agent that can call multiple tools.
#     """

#     # Initialize model (replace ChatNVIDIA with ChatOpenAI if needed)
#     llm = ChatNVIDIA(
#         model="meta/llama3-70b-instruct",
#         api_key=os.getenv("NVIDIA_API_KEY"),
#         temperature=0.3,
#     )

#     # Register tools
#     tools = [rag_tool, document_tool, email_tool]

#     # Add chat memory
#     memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

#     # Create modern ReAct-style agent
#     agent = create_agent(
#         llm=llm,
#         tools=tools,
#         memory=memory,
#         verbose=True,
#     )

#     return agent


# def run_agent(agent, query: str, history: list):
#     """
#     Run the agent with full conversation history.
#     """

#     try:
#         # Rebuild memory context from previous turns
#         memory = agent.memory
#         for msg in history:
#             if msg["role"] == "user":
#                 memory.chat_memory.add_user_message(msg["content"])
#             else:
#                 memory.chat_memory.add_ai_message(msg["content"])

#         # Run the agent (LangChain modern API)
#         result = agent.invoke({"input": query})

#         # Depending on LC version, result may be dict or string
#         if isinstance(result, dict):
#             output = (
#                 result.get("output")
#                 or result.get("agent")
#                 or str(result)
#             )
#         else:
#             output = str(result)

#         # Save new messages to memory
#         memory.chat_memory.add_user_message(query)
#         memory.chat_memory.add_ai_message(output)

#         return output

#     except Exception as e:
#         return f"❌ Error executing plan: {str(e)}"
