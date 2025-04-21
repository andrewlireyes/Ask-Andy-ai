import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub

st.title("Ask Andy AI")

# Load Hugging Face API key from Streamlit Secrets
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

@st.cache_resource
def load_resume():
    loader = PyPDFLoader("Social Media Marketing manager_Resume_Final.pdf")
    documents = loader.load_and_split()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

vectorstore = load_resume()

# Use a known-compatible model with Hugging Face hosted inference
qa_chain = RetrievalQA.from_chain_type(
    llm=HuggingFaceHub(
        repo_id="tiiuae/falcon-rw-1b",
        model_kwargs={"temperature": 0.7, "max_new_tokens": 512}
    ),
    retriever=vectorstore.as_retriever(),
    return_source_documents=False
)

# User interface
query = st.text_input("Ask a question about Andy's resume:")
if query:
    response = qa_chain.run(query)
    st.write(response)
