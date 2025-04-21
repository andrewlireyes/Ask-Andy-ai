import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

st.title("Ask Andy AI")

# Load and process resume
@st.cache_resource
def load_resume():
    loader = PyPDFLoader("Social Media Marketing manager_Resume_Final.pdf")
    documents = loader.load_and_split()
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

vectorstore = load_resume()
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4"),
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# User input
query = st.text_input("Ask a question about Andy's resume:")
if query:
    response = qa_chain.run(query)
    st.write(response)
