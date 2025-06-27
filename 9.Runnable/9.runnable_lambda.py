from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableParallel, RunnableSequence, RunnablePassthrough, RunnableLambda

load_dotenv()


def word_counter(text: str):
    return len(text.split())


prompt = PromptTemplate(
    template="Generate a joke about {topic}",
    input_variables=['topic']
)

model = ChatOpenAI(model="gpt-3.5-turbo")
parser = StrOutputParser()

runnable_word_counter = RunnableLambda(word_counter)

joke_gen_chain = RunnableSequence(prompt, model, parser)

parallel_chain = RunnableParallel(
    {
        'joke': RunnablePassthrough(),
        'word_count': RunnableLambda(word_counter)
    }
)

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)
result = final_chain.invoke({'topic': "AI"})
final_result = """{} \nword count - {}""".format(
    result['joke'], result['word_count'])
print(final_result)
