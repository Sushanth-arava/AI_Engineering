from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from dotenv import load_dotenv

load_dotenv()


class ChatState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]  


llm=ChatOpenAI()

def chat_node(state:ChatState):
    messages=state['messages']

    response=llm.invoke(messages)

    return {'messages':[response]}


checkpointer=MemorySaver()



graph=StateGraph(ChatState)

graph.add_node('chat_node',chat_node)
graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)

chatbot=graph.compile(checkpointer=checkpointer)

initial_state={'messages':[HumanMessage(content="What is the capital of France?")]}


# print(result['messages'][-1].content)

while True:

    user_message=input("Type: ")

    print("User asked: ", user_message)

    if user_message.strip().lower() in ['exit','quit','bye']:
        break

    result = chatbot.invoke({"messages":[HumanMessage(content=user_message)]})

    print("AI: ", result['messages'][-1].content)