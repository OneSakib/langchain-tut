from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers.pipelines import pipeline
from dotenv import load_dotenv

load_dotenv()

pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    model_kwargs={"temperature": 0.7}
)

llm = HuggingFacePipeline(pipeline=pipe)

model = ChatHuggingFace(llm=llm)

# Call the model
result = model.invoke("Who is Salman Khan?")
print(result.content)
