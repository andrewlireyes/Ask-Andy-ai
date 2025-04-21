import streamlit as st
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

vectorstore = load_resume()

qa_chain = RetrievalQA.from_chain_type(
    llm=HuggingFaceHub(
        repo_id="google/flan-t5-large",
        model_kwargs={"temperature": 0, "max_length": 512}
    ),
    retriever=vectorstore.as_retriever(),
    return_source_documents=False
)

query = st.text_input("Ask a question about Andy's resume:")
if query:
    response = qa_chain.run(query)
    st.write(response)
