import streamlit as st
import json
from utils import *


EMBEDDING_MODEL = "jinaai/jina-embeddings-v3"
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct-1M"
VECTORSTORE_PATH = "data/database_no_student_journey/jina-embeddings-v3/"
QUESTIONS_JSON = "data/questions.json" 

@st.cache_resource
def get_vectorstore():
    return build_vectorstore(persist_path=VECTORSTORE_PATH, model_name=EMBEDDING_MODEL)

@st.cache_resource
def get_llm():
    return load_local_llm(LLM_MODEL)

@st.cache_data
def get_suggested_questions(path):
    return prepare_question(path)

# UI Layout
st.title("📘 Fulbright QA - Ask Me Anything on Academic Policy")

# Load resources
vectorstore = get_vectorstore()
llm_pipe = get_llm()
suggested_questions = get_suggested_questions(QUESTIONS_JSON)

# Ask user for input
question_mode = st.radio("Choose your input mode:", ("Type your own", "Pick a suggested question"))

if question_mode == "Type your own":
    user_question = st.text_input("Enter your question:")
else:
    user_question = st.selectbox("Choose a question:", suggested_questions)

if user_question:
    with st.spinner("Generating answer..."):
        context, answer = ask_question(llm_pipe, vectorstore, user_question, top_k=5)
        st.markdown(f"### 🧠 Answer:\n{answer}")
        with st.expander("📚 Sources"):
            st.text(context)
