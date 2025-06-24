from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

# Defin the model
llm = HuggingFaceEndpoint(
    model="google/gemma-2-2b-it", task="text-generation")

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me 5 fact about {topic} in steps \n {format_instruction}",
    input_variables=['topic'],
    partial_variables={
        "format_instruction": parser.get_format_instructions()}

)
# prompt = template.format()

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)
# print(final_result)
# print(type(final_result))


chain = template | model | parser

result = chain.invoke({'topic': "Black hole"})

print(result)
