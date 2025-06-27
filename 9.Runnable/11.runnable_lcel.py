from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableParallel, RunnableSequence, RunnablePassthrough, RunnableLambda, RunnableBranch

load_dotenv()

prompt_1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=['topic']
)

prompt_2 = PromptTemplate(
    template="Summarize the following text \n {text}",
    input_variables=['text']
)

model = ChatOpenAI(model="gpt-3.5-turbo")
parser = StrOutputParser()

report_gen_chain = RunnableSequence(prompt_1, model, parser)

branch_chain = RunnableBranch(
    (lambda x: len(str(x).split()) > 100,
     RunnableSequence(prompt_2, model, parser)),
    RunnablePassthrough()
)


final_chain = RunnableSequence(report_gen_chain, branch_chain)

result = final_chain.invoke({'topic': "Russia vs Ukraine"})
print(result)
