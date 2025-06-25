from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


prompt_1 = PromptTemplate(
    template='Generate a detailed report on a {topic}',
    input_variables=['topic']
)

prompt_2 = PromptTemplate(
    template="Generate a 5 pointer summary of following text \n {text}",
    input_variables=['text']
)
model = ChatOpenAI(model='gpt-3.5-trubo')

parser = StrOutputParser()

chain = prompt_1 | model | parser | prompt_2 | model | parser

result = chain.invoke({'topic': "Unemployment in India"})

print(result)
