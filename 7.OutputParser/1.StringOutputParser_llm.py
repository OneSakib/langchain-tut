from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    model="google/gemma-2-2b-it", task="text-generation")

model = ChatHuggingFace(llm=llm)

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


prompt1 = template1.invoke({'topic': "Black hole"})
result1 = model.invoke(prompt1)

prompt2 = template2.invoke({'text': result1})
result2 = model.invoke(prompt2)

print(result2.content)
