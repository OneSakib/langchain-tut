from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()

st.title("Research Tool")
st.header("Research Prompt")

paper_input = st.selectbox(
    "Select Research paper names", [
        "Attention is all you need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3 Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis",
    ]
)

style_input = st.selectbox(
    "Select Explanation Style", [
        "Beginner Friendly",
        "Technical"
        "Code Oriented",
        "Mathematical",
    ]
)
length_input = st.selectbox(
    "Select Length of Explanation", [
        "Short (1-2 paragraphs)",
        "Medium (3-5 paragraphs)",
        "Long (detailed explanation with examples)",
    ]
)

model = ChatOpenAI(model="gpt-4", temperature=0)
# templates
template = load_prompt('template.json')

if st.button("Submit"):
    st.write("Generating explanation...")
    # Chain
    chain = template | model
    results = chain.invoke({
        'paper_name': paper_input,
        'style': style_input,
        'length': length_input
    })
    st.write("Response:")
    st.write(results.content)
    print(results.content)
