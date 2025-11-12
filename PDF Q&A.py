import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq.chat_models import ChatGroq
import os

os.environ["GROQ_API_KEY"] = "Your key"
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=1.2)

def read_file(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def create_vector_store(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
    chunks = splitter.split_text(text)
    embedding_function = FastEmbedEmbeddings()
    vector_store = Chroma.from_texts(chunks, embedding_function)
    return vector_store

def answer_query(vector_store, query, llm):
    docs = vector_store.similarity_search(query, k=2)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"Answer the question based on the following context:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = llm.invoke([("human", prompt)], max_completion_tokens=500)
    return response.content

st.title("PDF Q&A with Groq LLM")
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    text = read_file(uploaded_file)
    vector_store = create_vector_store(text)
    
    query = st.text_input("Ask a question about the PDF:")
    if query:
        answer = answer_query(vector_store, query, llm)
        st.write("💬 Answer:")
        st.write(answer)
