from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


chat_template = ChatPromptTemplate([
    SystemMessage(content="You are a helpful {domain} export."),
    HumanMessage(content="Explain in simple terms what {topic} is."),
])
prompt = chat_template.invoke(
    {'domain': 'Python', 'topic': 'decorators'})

print(prompt)
