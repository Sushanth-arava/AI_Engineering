from typing_extensions import TypedDict
import random
from typing import Literal
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    graph_info: str


# ──────────────────────────────────────────────────────────────────
# Nodes of a langgraph
# ──────────────────────────────────────────────────────────────────
def start_play(state: State):
    print("Start play node has been called")
    return {"graph_info": state["graph_info"] + "I am planning to play"}


def cricket(state: State):
    print("Cricket node has been called")
    return {"graph_info": state["graph_info"] + "Cricket"}


def badminton(state: State):
    print("Badminton node has been called")
    return {"graph_info": state["graph_info"] + "Badminton"}


# ──────────────────────────────────────────────────────────────────
# Conditional Edge
# ──────────────────────────────────────────────────────────────────
def choose_sport(state: State) -> Literal["cricket", "badminton"]:
    if random.random() > 0.5:
        return "cricket"
    else:
        return "badminton"


# ──────────────────────────────────────────────────────────────────
# Build the graph
# ──────────────────────────────────────────────────────────────────
graph = StateGraph(State)

graph.add_node("start_play", start_play)
graph.add_node("cricket", cricket)
graph.add_node("badminton", badminton)


graph.add_edge(START, "start_play")
# **: conditions
graph.add_conditional_edges("start_play", choose_sport)
graph.add_edge("cricket", END)
graph.add_edge("badminton", END)


graph_builder = graph.compile()

 with open("graph.png", "wb") as f:
     f.write(graph_builder.get_graph().draw_mermaid_png())


graph_builder.invoke({"graph_info": "My name is none"})
