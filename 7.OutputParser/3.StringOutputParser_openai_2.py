from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model="gpt-3.5-turbo", temperature=0.7
)

# 1st prompt > Details report

template1 = PromptTemplate(
    template="Write a detailed report on  {topic}",
    input_variables=["topic"]
)

# 2nd prompt > Summary report

template2 = PromptTemplate(
    template="Write a 5 line summary on the following text /n {text}",
    input_variables=["text"]
)

parser = StrOutputParser()


chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic': "Black hole"})

print(result)
