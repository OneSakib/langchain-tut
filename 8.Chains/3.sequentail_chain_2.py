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
model_1 = ChatOpenAI(model='gpt-3.5-turbo')
model_2 = ChatOpenAI(model='gpt-4')

parser = StrOutputParser()

chain = prompt_1 | model_1 | parser | prompt_2 | model_2 | parser

result = chain.invoke({'topic': "Unemployment in India"})

print(result)
