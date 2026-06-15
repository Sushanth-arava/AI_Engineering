from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")


class EvaluationSchema(BaseModel):
    feedback: str = Field(description="Detailed evaluation of the essay")
    scores: int = Field(description="Score out of 10", ge=0, le=10)


structured_model = model.with_structured_output(EvaluationSchema)

essay = """Artificial Intelligence (AI) refers to the capability of machines to perform tasks that typically require human intelligence, such as learning, reasoning, problem-solving, and decision-making. In recent years, AI has emerged as a transformative technology with applications across healthcare, education, agriculture, governance, and industry. By enabling data-driven decision-making and automation, AI has the potential to enhance productivity, improve service delivery, and accelerate economic growth. For a developing country like India, AI presents significant opportunities to address complex developmental challenges and strengthen public welfare systems.

However, the rapid advancement of AI also raises several concerns. Automation may lead to job displacement in certain sectors, requiring large-scale reskilling of the workforce. Issues related to data privacy, algorithmic bias, cybersecurity, and the ethical use of AI have become increasingly important. Furthermore, unequal access to AI technologies could widen the digital divide between regions and social groups. Therefore, the adoption of AI must be guided by principles of transparency, accountability, inclusivity, and human-centric development.

Recognizing both its opportunities and challenges, India has taken steps to promote responsible AI through initiatives such as the National Strategy for Artificial Intelligence. Investments in digital infrastructure, research and development, and skill development are crucial for maximizing the benefits of AI. Going forward, a balanced approach that combines technological innovation with robust regulatory frameworks will be essential. If harnessed responsibly, AI can become a powerful tool for achieving sustainable development, improving governance, and enhancing the quality of life for millions of people.
"""

prompt = f"Evaluate the  language quality of the following essay {essay} and provide a feedback and assign a suitable scores from 0-10"

print(structured_model.invoke(prompt).feedback)
