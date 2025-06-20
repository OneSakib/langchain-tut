from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

st.title("Research Tool")

st.header("Research Prompt")

user_input = st.text_area("Enter your research prompt here:")
if st.button("Submit"):
    if user_input:
        model = ChatOpenAI(model="gpt-4", temperature=0)
        results = model.invoke(user_input)
        st.write("Response:")
        st.write(results.content)
    else:
        st.error("Please enter a prompt before submitting.")
