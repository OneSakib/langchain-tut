from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# chat template
chat_template = ChatPromptTemplate([
    ('system', "You are a helpful customer support agent."),
    MessagesPlaceholder(variable_name="chat_history"),
    ('human', '{query}')
])
# load chat history
chat_history = []
with open('chat_history.txt', 'r') as file:
    chat_history.extend(file.readlines())

print("Chat history loaded successfully.", chat_history)

# create project
prompt = chat_template = chat_template.invoke({
    'chat_history': chat_history,
    'query': "where is my refund?",
})

print(">>>PRomp", prompt)
