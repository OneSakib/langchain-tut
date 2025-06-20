from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


model = ChatOpenAI(model="gpt-4", temperature=0)

while True:
    user_input = input(
        "You: \n Enter your research prompt (or 'exit' to quit): ")
    if user_input.lower() == 'exit' or user_input.lower() == 'quit':
        break
    results = model.invoke(user_input)
    print("Response:")
    print("AI:", results.content)
