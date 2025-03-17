import os
import streamlit as st
from pdf2image import convert_from_path
import pytesseract
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from docx import Document
from tempfile import NamedTemporaryFile

# Set Streamlit page configuration
st.set_page_config(page_title="Document Query System", layout="wide")

# Function to extract text from scanned PDFs using OCR
def ocr_pdf(file_path):
    images = convert_from_path(file_path)
    text = "\n".join([pytesseract.image_to_string(img) for img in images])
    return text.strip() if text.strip() else None

# Function to load text from PDF or DOCX files
def load_document(file_path):
    if file_path.endswith(".pdf"):
        try:
            doc_loader = PyPDFLoader(file_path)
            pages = doc_loader.load()
            extracted_text = "\n".join([page.page_content for page in pages])
            if not extracted_text.strip():
                extracted_text = ocr_pdf(file_path)
            return extracted_text if extracted_text else None
        except Exception:
            return None
    elif file_path.endswith(".docx"):
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    return None

# Function to split and embed text using Langchain
def process_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.create_documents([text])
    embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en")
    db = FAISS.from_texts([chunk.page_content for chunk in chunks], embeddings)
    return db

# Function to generate response using Groq API
def generate_response(db, query):
    contexts = db.similarity_search(query, k=5)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert at answering questions based on the extracted document context: {context}"),
        ("human", "{question}")
    ])
    model = ChatGroq(model_name="llama3-8b-8192")
    chain = prompt | model
    response = chain.invoke({
        "context": "\n\n".join([c.page_content for c in contexts]),
        "question": query
    })
    return response.content

# Streamlit UI
st.title("📄 Document Query System")

uploaded_file = st.file_uploader("Upload a PDF or DOCX document", type=["pdf", "docx"])
query = st.text_input("Enter your query:")

if uploaded_file and query:
    file_extension = ".pdf" if uploaded_file.name.endswith(".pdf") else ".docx"
    with NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    st.write("⏳ Processing document...")
    with st.spinner("Extracting and analyzing text..."):
        document_text = load_document(temp_file_path)

    if document_text:
        with st.spinner("Generating response..."):
            db = process_text(document_text)
            response = generate_response(db, query)
            st.success("Response Generated")
            st.write(response)
    else:
        st.error("Could not extract text. The file may be scanned or unsupported.")
