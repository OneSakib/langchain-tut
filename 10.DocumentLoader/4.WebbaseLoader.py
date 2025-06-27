from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

url = "https://www.flipkart.com/apple-macbook-air-m2-16-gb-256-gb-ssd-macos-sequoia-mc7x4hn-a/p/itmdc5308fa78421?pid=COMH64PY76CJKBYU&lid=LSTCOMH64PY76CJKBYUOL7TOK&marketplace=FLIPKART&store=6bo%2Fb5g&spotlightTagId=default_BestsellerId_6bo%2Fb5g&srno=b_1_3&otracker=browse&fm=organic&iid=823d3bd3-952d-48b2-985e-2aa8db84de14.COMH64PY76CJKBYU.SEARCH&ppt=None&ppn=None&ssid=0ybjd9nyr40000001751044075892"
loader = WebBaseLoader(
    web_path=url
)
docs = loader.load()

prompt = PromptTemplate(
    template="Answer the following question \n {question} -\n  {text}",
    input_variables=['question', 'text']

)
model = ChatOpenAI(model='gpt-3.5-turbo')
parser = StrOutputParser()


chain = prompt | model | parser

result = chain.invoke(
    {'question': 'What is the price of this product?', 'text': docs[0].page_content})

print(result)
