from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence

load_dotenv()


prompt = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=['topic']
)


model = ChatOpenAI(model="gpt-3.5-turbo")

parser = StrOutputParser()


prompt2 = PromptTemplate(
    template="Emplain the following joke - {text}",
    input_variables=['text']
)
chain = RunnableSequence(prompt, model, parser, prompt2, model, parser)
result = chain.invoke({"topic": "Python"})

print(result)
