from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate

llm = OpenAI(model_name='gpt-3.5-turbo', temperature=0.7)

prompt = PromptTemplate(
    template="Suggest a catchy blog title about {topic}",
    input_variables=['topic']
)

topic = input("Enter a topic:\n")

formatted_prompt = prompt.format(topic=topic)
blog_title = llm.predict(formatted_prompt)

print('Generated Blog Title: ', blog_title)
