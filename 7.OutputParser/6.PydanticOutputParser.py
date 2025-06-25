from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# Defin the model
llm = HuggingFaceEndpoint(model="google/gemma-2-2b-it", task="text-generation")

model = ChatHuggingFace(llm=llm)


class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(description="Age of the person")
    city: str = Field(description="Name of the city of the person belong to")


parser = PydanticOutputParser(pydantic_object=Person)
tempate = PromptTemplate(
    template="Generate the name, age and city of a fictional {place} person \n {format_instruction}",
    input_variables=['place'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

# prompt = tempate.invoke({'place': "India"})
# result = model.invoke(prompt)
# final_result = parser.parse(result.content)
# print(final_result)

chain = tempate | model | parser
final_result = chain.invoke({'place': "India"})
print(final_result)
