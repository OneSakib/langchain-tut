from langchain_openai import ChatOpenAI
from typing import List, Union
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()


model = ChatOpenAI(model="gpt-4", temperature=0)


chat_history: List[Union[SystemMessage, HumanMessage, AIMessage]] = [
    SystemMessage(content="You are a helpful research assistant. for python quizs and exercises. so you have to ask question and then wait for the answer. If the answer is correct, you can ask another question. If the answer is wrong, you can give a hint or explanation.")
]

while True:
    user_input = input(
        "You: \n Enter your research prompt (or 'exit' to quit): ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() == 'exit' or user_input.lower() == 'quit':
        break
    results = model.invoke(chat_history)
    chat_history.append(AIMessage(results.content))
    print(f"AI: {results.content} \n")


print(f"History: {chat_history}")
