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

# Set wide layout for full screen usage
st.set_page_config(page_title="RAG-based Document Query System", layout="wide")

# Updated Custom CSS for a light theme with modern styling and smooth animations
st.markdown("""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');

        /* Global styles for light theme */
        body {
            background: #f4f4f4;
            font-family: 'Montserrat', sans-serif;
            color: #333;
        }

        /* Main container styling */
        .stApp {
            background-color: #fff;
            padding: 2rem;
            border-radius: 16px;
            box-shadow: 0px 6px 18px rgba(0,0,0,0.1);
            margin-top: 2rem;
            width: 100%;
        }

        /* Title Styling */
        .title {
            color: #007BFF;
            font-weight: 700;
            text-align: center;
            font-size: 3rem;
            margin-bottom: 2rem;
        }

        /* File uploader area styling */
        .css-1d391kg, .css-1cpxqw2, .stFileUploader {
            background-color: #e9ecef;
            border: 2px dashed #007BFF;
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
            transition: background 0.3s ease, transform 0.3s ease;
        }
        .css-1d391kg:hover, .css-1cpxqw2:hover, .stFileUploader:hover {
            background-color: #dee2e6;
            transform: scale(1.02);
        }

        /* Button Styling */
        button {
            background-color: #007BFF;
            color: #fff;
            font-weight: 600;
            border-radius: 10px;
            padding: 14px 28px;
            border: none;
            cursor: pointer;
            transition: background 0.3s ease, transform 0.3s ease;
            width: 100%;
            margin-top: 1rem;
        }
        button:hover {
            background-color: #0056b3;
            transform: translateY(-2px);
        }

        /* Text Input Styling for Query Box */
        input[type="text"], input, textarea {
            background-color: #fff !important;
            border: 1px solid #ced4da;
            color: #333 !important;
            padding: 0.75rem;
            border-radius: 8px;
            width: 100%;
            margin-top: 1rem;
            font-size: 1rem;
            transition: border 0.3s ease, box-shadow 0.3s ease;
        }
        input[type="text"]:focus, input:focus, textarea:focus {
            border: 1px solid #007BFF;
            box-shadow: 0 0 5px rgba(0, 123, 255, 0.5);
            outline: none;
        }

        /* Response Styling */
        .response-box {
            background-color: #e9ecef;
            padding: 1rem;
            border-radius: 8px;
            border-left: 5px solid #007BFF;
            font-size: 16px;
            color: #333;
            margin-top: 1.5rem;
            word-wrap: break-word;
        }

        /* Error Styling */
        .error {
            color: #dc3545;
            font-size: 16px;
            font-weight: bold;
            margin-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

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
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, length_function=len, is_separator_regex=False
    )
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
st.markdown('<h1 class="title">📄 RAG-based Document Query System</h1>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("**Upload a PDF or DOCX document**", type=["pdf", "docx"])
query = st.text_input("**Enter your query:**")

if uploaded_file and query:
    # Determine file extension and create a temporary file accordingly
    file_extension = ".pdf" if uploaded_file.name.endswith(".pdf") else ".docx"
    with NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    st.write("⏳ **Processing document... Please wait.**")
    with st.spinner("Extracting and analyzing text..."):
        document_text = load_document(temp_file_path)

    if document_text:
        with st.spinner("Generating response..."):
            db = process_text(document_text)
            response = generate_response(db, query)
            st.markdown(f'<div class="response-box">✅ <b>Response:</b><br>{response}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="error">❌ Could not extract text. The file may be scanned or unsupported.</p>', unsafe_allow_html=True)
