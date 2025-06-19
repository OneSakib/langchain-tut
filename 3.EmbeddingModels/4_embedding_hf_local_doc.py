from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",)
documents = [
    "Delhi is the capital of India.",
    "Mumbai is the financial capital of India.",
    "Kolkata is known for its cultural heritage.",
    "Chennai is famous for its cuisine and temples.",
]
result = embedding.embed_documents(documents)
print(result)
