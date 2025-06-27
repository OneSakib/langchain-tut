from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


loader = TextLoader('cricket.txt', encoding='utf-8')
document = loader.load()
model = ChatOpenAI(model='gpt-3.5-turbo')

prompt = PromptTemplate(
    template="Write a summary for the following poem \n {poem}",
    input_variables=['poem']
)

parser = StrOutputParser()


chain = prompt | model | parser
result = chain.invoke({'poem': document[0].page_content})

print(result)
