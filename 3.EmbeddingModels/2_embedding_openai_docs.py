from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=32)
documents = [
    "Delhi is the capital of India.",
    "Mumbai is the financial capital of India.",
    "Kolkata is known for its cultural heritage.",
    "Chennai is famous for its cuisine and temples.",
]
result = embedding.embed_documents(documents)
print(result)
