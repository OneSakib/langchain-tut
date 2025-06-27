from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableParallel, RunnableSequence

load_dotenv()


prompt_1 = PromptTemplate(
    template="Generate a tweet about {topic}",
    input_variables=['topic']
)
prompt_2 = PromptTemplate(
    template="Generate a Linkdein post about {topic}",
    input_variables=['topic']
)


model = ChatOpenAI(model="gpt-3.5-turbo")
parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        'tweet': RunnableSequence(prompt_1, model, parser),
        'linkdin': RunnableSequence(prompt_2, model, parser)
    }
)

result = parallel_chain.invoke({'topic': "AI"})
print(result)
