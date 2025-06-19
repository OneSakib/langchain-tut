import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=300)
documents = [
    "Virat Kohli is a famous cricketer. he is know for his batting skills.",
    "Sachin Tendulkar is a legendary cricketer. he is known for his batting skills.",
    "Rohit Sharma is a great cricketer. he is known for his batting skills.",
    "MS Dhoni is a legendary cricketer. he is known for his wicketkeeping skills.",
    "Ravindra Jadeja is a great all-rounder. he is known for his fielding skills.",
    "Hardik Pandya is a great all-rounder. he is known for his batting and bowling skills.",
]

query = "Tell me about Rohit Sharma?"
documents_embeddings = np.array(embedding.embed_documents(documents))
query_embedding = np.array(embedding.embed_query(query)).reshape(1, -1)
cosine_similarities = cosine_similarity(
    query_embedding,
    documents_embeddings
)[0]
index, score = sorted(
    enumerate(cosine_similarities), key=lambda x: x[1])[-1]
print(">>>>>", documents[index])
print(">>>>>", score)
