from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


model = ChatOpenAI(model='gpt-4', temperature=0)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Tell me a joke"),
]
result = model.invoke(messages)

messages.append(AIMessage(result.content))


print(messages)
