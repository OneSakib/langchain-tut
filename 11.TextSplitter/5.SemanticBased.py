from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

splitter = SemanticChunker(
    OpenAIEmbeddings(), breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=3
)
sample = """
Farmers are the silent heroes who work tirelessly to feed the world. From sunrise to sunset, they toil in the fields — braving the sun, rain, and uncertainties — to grow the grains, fruits, and vegetables that sustain our lives.
Without farmers, there is no food. Without food, there is no life.
They represent dedication, patience, and resilience. Despite facing challenges like changing weather, market fluctuations, and limited resources, they continue to nurture the earth and provide for others.
We must respect and support our farmers — not just with words, but with fair policies, technology, education, and gratitude.

Terrorism is one of the gravest threats facing the world today. It spreads fear, destroys innocent lives, and disrupts peace in societies. Whether driven by hate, ideology, or politics, terrorism never brings real justice — only suffering and destruction.
Terrorist acts target not just individuals, but the core values of humanity — freedom, unity, and coexistence. The victims of terrorism are often ordinary people, caught in the crossfire of someone else's agenda.
Fighting terrorism is not just a job for governments or armies — it requires global unity, awareness, education, and compassion. The root causes like poverty, hate, extremism, and injustice must be addressed to truly defeat terrorism.
Let us stand together for peace, for humanity, and for a future where no child grows up in fear.
"""

docs = splitter.create_documents([sample])
print(len(docs))
print(docs)
