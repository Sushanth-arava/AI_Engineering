from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
)

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


load_dotenv()


model = ChatOpenAI()

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template="Write a detailed report on {topic}", input_variables=["topic"]
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template="Write a 5 line summary on the following text. /n {text}",
    input_variables=["text"],
)

# parser = StrOutputParser()

# chain = template1 | model | parser | template2 | model | parser

# result = chain.invoke({"topic": "black hole"})

# print(result)


# parser = JsonOutputParser()

# template = PromptTemplate(
#     template="Give me 5 facts about {topic} \n {format_instruction}",
#     input_variables=["topic"],
#     partial_variables={"format_instruction": parser.get_format_instructions()},
# )

# chain = template | model | parser

# result = chain.invoke({"topic": "black hole"})

# print(result["fact1"])


class Person(BaseModel):

    name: str = Field(description="Name of the person")
    age: int = Field(gt=18, description="Age of the person")
    city: str = Field(description="Name of the city the person belongs to")


parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Generate the name, age and city of a fictional {place} person \n {format_instruction}",
    input_variables=["place"],
    partial_variables={"format_instruction": parser.get_format_instructions()},
)

chain = template | model | parser

final_result = chain.invoke({"place": "sri lankan"})

print(final_result)
