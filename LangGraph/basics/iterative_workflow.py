from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal, Annotated
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()


generator_llm = ChatOpenAI(model="gpt-4o")
evaluator_llm = ChatOpenAI(model="gpt-4o-mini")
optimization_llm = ChatOpenAI(model="o3")


class TweetEvaluationSchema(BaseModel):
    evaluation: Literal["approved", "needs improvement"]
    feedback: str


class TweetState(TypedDict):
    topic: str
    tweet: str
    evaluation: Literal["approved", "needs_improvement"]
    feedback: str
    iteration: int
    max_interaton: int


def generate_tweet(state: TweetState):
    # prompt
    messages = [
        SystemMessage(content="You are a funny and clever Twitter/X influencer."),
        HumanMessage(content=f"""
Write a short, original, and hilarious tweet on the topic: "{state['topic']}".

Rules:
- Do NOT use question-answer format.
- Max 280 characters.
- Use observational humor, irony, sarcasm, or cultural references.
- Think in meme logic, punchlines, or relatable takes.
- Use simple, day to day english
- This is version {state['iteration'] + 1}.
"""),
    ]
    response = generator_llm.invoke(messages)
    return {"tweet": response.content, "iteration": state["iteration"] + 1}


def evaluate_tweet(state: TweetState):
    messages = [
        SystemMessage(
            content="You are a ruthless no laugh given Twitter critic. You evaluate tweets based on humor, originality, virality, and tweet format."
        ),
        HumanMessage(content=f"""You have to evaluate the following tweet: 
        {state['tweet']}
        
        Criteria:
        - Originality: Check if it is fresh and not something you've seen 100 times before.
        - Humor and Punchiness.
        - Virality Potential.
        - Format.
        
        Auto-rejection rules:
        Automatically reject the tweet if it is in a question-answer format, exceeds 280 characters, or uses traditional jokes.
        
        Respond only in structured format:
        - Evaluation: 'approved' or 'needs improvement'
        - Feedback: One paragraph detailing the strengths and weaknesses."""),
    ]
    structured_evaluator_llm = evaluator_llm.with_structured_output(
        TweetEvaluationSchema
    )
    response = structured_evaluator_llm.invoke(messages)

    return {"evaluation": response.evluation, "feedback": response.feedback}


graph = StateGraph(TweetState)

graph.add_node("generate_tweet", generate_tweet)
graph.add_node("evaluate_tweet", evaluate_tweet)
graph.add_node("optimize_tweet", optimize_tweet)
