from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableParallel, RunnableSequence, RunnablePassthrough

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


joke_gen_chain = RunnableSequence(prompt_1, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'exmplain': RunnableSequence(prompt_2, model, parser)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = final_chain.invoke({'topic': 'Cricket'})

print(result)
