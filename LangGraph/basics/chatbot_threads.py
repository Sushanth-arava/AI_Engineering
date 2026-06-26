from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


llm = ChatOpenAI()


def chat_node(state: ChatState):
    return {"messages": [llm.invoke(state["messages"])]}


graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

# Checkpointer enables state persistence across calls
checkpointer = MemorySaver()
chatbot = graph.compile(checkpointer=checkpointer)


def chat(thread_id: str, user_message: str):
    config = {"configurable": {"thread_id": thread_id}}
    result = chatbot.invoke(
        {"messages": [HumanMessage(content=user_message)]},
        config=config,
    )
    print(f"[{thread_id}] User: {user_message}")
    print(f"[{thread_id}] Bot : {result['messages'][-1].content}\n")


# Thread "alice" — her own conversation history
chat("alice", "Hi! My name is Alice.")
chat("alice", "What is my name?")   # remembers "Alice"

# Thread "bob" — completely separate history, knows nothing about Alice
chat("bob", "What is my name?")     # has no idea
