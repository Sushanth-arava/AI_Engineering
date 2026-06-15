from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()


class LLMState(TypedDict):
    question: str
    answer: str


def llm_qa(state: LLMState) -> LLMState:
    # TODO: extract the question from state
    question = state["question"]

    # TODO: form a prompt"
    prompt = f"Answer the following question {question}"
    # TODO: ask the question again
    answer = model.invoke(prompt).content

    return {"answer": answer}


graph = StateGraph(LLMState)

graph.add_node("llm_qa", llm_qa)

graph.add_edge(START, "llm_qa")

graph.add_edge("llm_qa", END)

workflow = graph.compile()

initial_state = {"question": "What is the capital of Uruguay?"}

final_state = workflow.invoke(initial_state)

print(final_state["answer"])
