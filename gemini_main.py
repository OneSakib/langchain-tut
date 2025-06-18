from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

import os

from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",
                             temperature=0,
                             max_tokens=None,
                             timeout=None,
                             max_retries=2,)
# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17")
# example
# Send a message
# response = llm.invoke([
#     HumanMessage(content="What is Pyton?")
# ])

# print(response.content)

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)
print("ai_msg", ai_msg)
