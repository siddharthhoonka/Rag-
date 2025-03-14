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

os.environ["GROQ_API_KEY"] = "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"  

# Custom CSS for styling
st.markdown("""
    <style>
        /* Background and Container Styling */
        .main {
            background-color: #f4f4f9;
            padding: 2rem;
        }
        .stApp {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
            margin-top: 1rem;
            max-width: 800px;
            margin-left: auto;
            margin-right: auto;
        }

        /* Title Styling */
        .title {
            color: #1f77b4;
            font-weight: bold;
            text-align: center;
            font-size: 2.5rem;
            margin-bottom: 1.5rem;
        }

        /* Upload Area Styling */
        .upload-area {
            border: 2px dashed #1f77b4 !important;
            padding: 1.5rem;
            border-radius: 12px;
            background-color: #fafafa;
            text-align: center;
            transition: background 0.3s ease;
        }
        .upload-area:hover {
            background-color: #e6f4ff;
        }

        /* Button Styling */
        .button {
            background-color: #1f77b4 !important;
            color: white !important;
            font-weight: bold !important;
            border-radius: 8px !important;
            padding: 12px 24px !important;
            border: none !important;
            cursor: pointer !important;
            transition: background 0.3s ease !important;
            margin-top: 1rem;
            width: 100%;
        }
        .button:hover {
            background-color: #175d92 !important;
        }

        /* Response Styling */
        .response-box {
            background-color: #eef6ff;
            padding: 1rem;
            border-radius: 8px;
            border-left: 5px solid #1f77b4;
            font-size: 16px;
            color: #333333;
            margin-top: 1.5rem;
            word-wrap: break-word;
        }

        /* Error Styling */
        .error {
            color: red;
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
    with NamedTemporaryFile(delete=False, suffix=".pdf" if uploaded_file.name.endswith(".pdf") else ".docx") as temp_file:
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
