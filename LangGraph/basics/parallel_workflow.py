from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()


class ParallelState(TypedDict):
    a: int
    b: int
    add_value: int
    prod_value: int
    sub_value: int


def add_variables(state: ParallelState):
    return {"add_value": state["a"] + state["b"]}


def sub_variables(state: ParallelState):
    return {"sub_value": state["a"] - state["b"]}


def prod_variables(state: ParallelState):
    return {"prod_value": state["a"] * state["b"]}


def finalOutput(state: ParallelState):
    print(
        f"Add-{state['add_value']},Minus-{state['sub_value']},Multiply-{state['prod_value']}"
    )
    return {}


graph = StateGraph(ParallelState)

graph.add_node("add_variables", add_variables)
graph.add_node("sub_variables", sub_variables)
graph.add_node("prod_variables", prod_variables)
graph.add_node("finalOutput", finalOutput)

graph.add_edge(START, "add_variables")
graph.add_edge(START, "sub_variables")
graph.add_edge(START, "prod_variables")

graph.add_edge("add_variables", "finalOutput")
graph.add_edge("sub_variables", "finalOutput")
graph.add_edge("prod_variables", "finalOutput")

graph.add_edge("finalOutput", END)

workflow = graph.compile()

initial_state = {"a": 10, "b": 5}

final_state = workflow.invoke(initial_state)

with open("graph.png", "wb") as f:
    f.write(workflow.get_graph().draw_mermaid_png())

print(final_state)
