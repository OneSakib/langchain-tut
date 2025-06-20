from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


load_dotenv()


model = ChatOpenAI(model="gpt-4", temperature=0)

results = model.invoke("Write 5 line poem on cricket")

print(results.content)
