
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

API_KEY = "API KEY"

st.title("Chat with your PDF")


file_upload = st.file_uploader("Upload a PDF", type="pdf")

if file_upload:
    with open("copy.pdf", "wb") as f:
        f.write(file_upload.read())

    
    loader = PyMuPDFLoader("copy.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    st.success(f"Successfully processed {file_upload.name}")

    
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedding=embedding)
    retriever = vectorstore.as_retriever()


    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        google_api_key=API_KEY
    )


    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )

   
    query = st.text_input("Ask a question about the PDF:")
    if query:
        answer = qa_chain.run(query)
        st.write("Answer:", answer)
