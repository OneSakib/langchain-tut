from langchain.text_splitter import RecursiveCharacterTextSplitter


text = """
My name is Sakib Malik. I am from Saharanpur. I have done MCA from dev bhoomi college sahranpur. Currently i am working as a python Developer in Wisdominfosoft. 
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=10,
    chunk_overlap=2
)


result = splitter.split_text(text)
print(result)
