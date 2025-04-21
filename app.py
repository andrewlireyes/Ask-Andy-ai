import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub

st.title("Ask Andy AI")

@st.cache_resource
def load_resume():
    loader = PyPDFLoader("Social Media Marketing manager_Resume_Final.pdf")
    documents = loader.load_and_split()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

# Load Hugging Face token from Streamlit secrets
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

vectorstore = load_resume()

qa_chain = RetrievalQA.from_chain_type(
    llm=HuggingFaceHub(
        repo_id="tiiuae/falcon-7b-instruct",
        model_kwargs={"temperature": 0.7, "max_new_tokens": 512}
    ),
    retriever=vectorstore.as_retriever(),
    return_source_documents=False
)

query = st.text_input("Ask a question about Andy's resume:")
if query:
    response = qa_chain.run(query)
    st.write(response)
