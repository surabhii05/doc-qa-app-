import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader as Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI

import tempfile
import os

# Streamlit page config
st.set_page_config(page_title="📄 Document QA", layout="wide")
st.title("📄 Document-Based Question Answering")

# Load OpenAI API key from Streamlit secrets
openai_api_key = st.secrets.get("OPENAI_API_KEY")
if not openai_api_key:
    st.error("⚠️ Please add your OpenAI API key in the 'Secrets' tab!")
    st.stop()

# File upload
uploaded_file = st.file_uploader("Upload a PDF or Word Document", type=["pdf", "docx"])
query = st.text_input("Ask a question based on the document:")

if uploaded_file and query:
    with st.spinner("Processing..."):

        # Save uploaded file temporarily
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())
        temp_file.close()

        # Load document
        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(temp_file.name)
        else:
            loader = Docx2txtLoader(temp_file.name)
        documents = loader.load()

        # Check if document has content
        if not documents:
            st.error("❌ Document is empty or unreadable. Try another file.")
            os.remove(temp_file.name)
            st.stop()

        # Chunk and embed
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)

        if not chunks:
            st.error("❌ Could not create text chunks. Try a different file.")
            os.remove(temp_file.name)
            st.stop()

        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        db = FAISS.from_documents(chunks, embeddings)

        # Similarity search
        relevant_docs = db.similarity_search(query)

        # Answer generation
        llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")
        chain = load_qa_chain(llm, chain_type="stuff")
        result = chain.run(input_documents=relevant_docs, question=query)

        st.success("Answer:")
        st.write(result)

        os.remove(temp_file.name)
