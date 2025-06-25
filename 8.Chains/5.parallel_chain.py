from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.schema.runnable import RunnableParallel
load_dotenv()


llm = HuggingFaceEndpoint(model="google/gemma-2-2b-it", task="text-generation")

model_1 = ChatHuggingFace(llm=llm)
model_2 = ChatOpenAI(model='gpt-4')

prompt_1 = PromptTemplate(
    template='Generate short and simple notes form the following text \n {text}',
    input_variables=['text']
)

prompt_2 = PromptTemplate(
    template="Generate 5 short question answers from the following text  \n {text}",
    input_variables=['text']
)
prompt_3 = PromptTemplate(
    template="Merge the following notes and quiz into a single document \n notes ->  {notes} and quiz -> {quiz}",
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

chain_1 = prompt_1 | model_1 | parser
chain_2 = prompt_2 | model_2 | parser

parallel_chain = RunnableParallel({
    'notes': chain_1,
    'quiz': chain_2
})

chain_3 = prompt_3 | model_1 | parser

final_chain = parallel_chain | chain_3 | parser

text = """
Python, a versatile and beginner-friendly programming language, is experiencing continued growth and relevance in 2025. Its popularity stems from its clean syntax, extensive libraries, and applications in various fields like web development, data science, and artificial intelligence. For those interested in learning or staying updated on Python, numerous blogs and online resources are available. 
Popular Python Blogs and Resources:
Real Python: A comprehensive platform offering tutorials, articles, and courses for Python developers of all levels. 
Planet Python: A go-to resource for Python news and blog postings from various sources. 
Full Stack Python: A detailed guide to Python covering a wide range of topics. 
Python Library Blog: Provides in-depth tutorials and explorations of Python's ecosystem. 
TalkPython.fm: A Python blog in audio format, offering interviews and discussions with Python experts. 
Python Software Foundation: The official Python organization's blog, providing news and updates on the language. 
Finxter: A blog focused on providing practical Python guides and solutions. 
The Python Guru: Offers a wide range of Python-related content, including tutorials and tips. 
Draft.dev: Features blog posts, tutorials, and articles on Python programming. 
LearnPython.com: A good resource for beginners to start learning Python. 
Why Python Remains Popular:
Ease of Learning:
Python's syntax is designed to be readable and easy to understand, making it a great language for beginners. 
Versatility:
Python is used in diverse fields, including web development (with frameworks like Django and Flask), data science (with libraries like NumPy, Pandas, and Scikit-learn), and machine learning/AI (with libraries like TensorFlow and PyTorch). 
Vibrant Community:
A large and active community provides ample support, resources, and libraries for Python developers. 
Continuous Development:
The Python Software Foundation ensures ongoing development and updates to the language. 
Future-Proof:
Python is expected to remain a relevant and powerful language in the years to come, with ongoing advancements in areas like AI and cloud computing. 
"""

result = final_chain.invoke({'text': text})

print(result)
