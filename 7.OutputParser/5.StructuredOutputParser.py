from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Defin the model
llm = HuggingFaceEndpoint(
    model="google/gemma-2-2b-it", task="text-generation")

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name="fact_1", description="Fact 1 about the topic"),
    ResponseSchema(name="fact_2", description="Fact 2 about the topic"),
    ResponseSchema(name="fact_3", description="Fact 3 about the topic"),
]


parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="Give me 3 fact about {topic} \n {format_instruction}",
    input_variables=['topic'],
    partial_variables={
        "format_instruction": parser.get_format_instructions()}
)

# prompt = template.format(topic="Black hole")
# result = model.invoke(prompt)
# final_result = parser.parse(result.content)
# print(final_result)

# Chain
chain = template | model | parser
result = chain.invoke({'topic': "Black hole"})
print(result)
