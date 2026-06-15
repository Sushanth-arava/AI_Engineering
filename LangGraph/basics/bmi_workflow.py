from typing_extensions import TypedDict
import random
from typing import Literal
from langgraph.graph import StateGraph, START, END


class BMIState(TypedDict):
    weight_kg: float
    height_m: float
    bmi: float


def calculate_bmi(state: BMIState) -> BMIState:
    weight = state["weight_kg"]
    height = state["height_m"]
    bmi = weight / (height**2)
    state["bmi"] = round(bmi, 2)
    return state


graph = StateGraph(BMIState)

graph.add_node("calculate_bmi", calculate_bmi)

graph.add_edge(START, "calculate_bmi")

graph.add_edge("calculate_bmi", END)

workflow = graph.compile()


final_state = workflow.invoke({"weight_kg": 80, "height_m": 1.73})
print(final_state)
