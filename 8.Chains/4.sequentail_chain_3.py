from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
load_dotenv()


prompt_1 = PromptTemplate(
    template='Generate a detailed report on a {topic}',
    input_variables=['topic']
)

prompt_2 = PromptTemplate(
    template="Generate a 5 pointer summary of following text \n {text}",
    input_variables=['text']
)

llm = HuggingFaceEndpoint(model="google/gemma-2-2b-it", task="text-generation")

model_1 = ChatHuggingFace(llm=llm)
model_2 = ChatOpenAI(model='gpt-4')

parser = StrOutputParser()

chain = prompt_1 | model_1 | parser | prompt_2 | model_2 | parser

result = chain.invoke({'topic': "Unemployment in India"})

print(result)
