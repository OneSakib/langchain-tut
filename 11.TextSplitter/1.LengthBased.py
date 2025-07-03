from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


text = """
Cricket is more than just a sport. It's an emotion, a tradition, and for many, a way of life. From the green fields of England to the bustling stadiums of India, cricket has captured the hearts of millions for generations.
Cricket traces its origins back to 16th-century England. What started as a village pastime evolved into a formalized sport in the 18th century. The Marylebone Cricket Club (MCC) helped standardize the laws of the game, and soon, international matches began to take shape — with the first official Test match played between England and Australia in 1877.
Today, cricket is played and followed by billions across the globe. Countries like India, Pakistan, Australia, England, South Africa, and New Zealand dominate the international scene. The game has also found passionate fans in nations like Bangladesh, Afghanistan, Sri Lanka, and the West Indies, making it truly global.Cricket has evolved with time. It now exists in multiple formats, each with its own charm:

Test Matches (5 Days): The purest form of cricket. A test of skill, patience, and endurance.

One Day Internationals (ODIs): A faster-paced format with 50 overs per side. The ICC World Cup is held in this format.

T20s: The shortest and most explosive format. Matches last about 3 hours and are packed with entertainment. Leagues like the Indian Premier League (IPL) have made T20s wildly popular.
"""

loader = PyPDFLoader('dl-curriculum.pdf')

docs = loader.load()


splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=""
)


# result = splitter.split_text(text)
result = splitter.split_documents(docs)
print(result[0].page_content)
