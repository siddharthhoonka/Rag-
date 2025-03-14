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
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ API key is missing. Please set it in the environment.")
    st.stop()

def ocr_pdf(file_path):
    try:
        images = convert_from_path(file_path)
        text = "\n".join([pytesseract.image_to_string(img) for img in images])
        return text.strip() if text.strip() else None
    except Exception as e:
        st.error(f"Error during OCR: {e}")
        return None

def load_document(file_path):
    try:
        if file_path.endswith(".pdf"):
            doc_loader = PyPDFLoader(file_path)
            pages = doc_loader.load()
            extracted_text = "\n".join([page.page_content for page in pages])
            if not extracted_text.strip():
                extracted_text = ocr_pdf(file_path)
            return extracted_text if extracted_text else None

        elif file_path.endswith(".docx"):
            doc = Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        st.error(f"Failed to load document: {e}")
        return None

def process_text(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50, length_function=len, is_separator_regex=False
        )
        chunks = text_splitter.create_documents([text])
        
        # Embeddings
        model_name = "BAAI/bge-small-en"
        embeddings = HuggingFaceBgeEmbeddings(model_name=model_name)
        
        chunk_texts = [chunk.page_content for chunk in chunks]
        db = FAISS.from_texts(chunk_texts, embeddings)
        return db
    except Exception as e:
        st.error(f"Error during text processing: {e}")
        return None

def generate_response(db, query):
    try:
        contexts = db.similarity_search(query, k=5)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at answering questions based on the extracted document context: {context}"),
            ("human", "{question}"),
        ])
        
        model = ChatGroq(model_name="llama3-8b-8192")
        chain = prompt | model
        
        response = chain.invoke({
            "context": "\n\n".join([c.page_content for c in contexts]),
            "question": query
        })
        return response.content
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return None

# Streamlit UI
st.title("RAG-based Document Query System")

uploaded_file = st.file_uploader("Upload a PDF or DOCX document", type=["pdf", "docx"])
query = st.text_input("Enter your query:")

if uploaded_file and query:
    with NamedTemporaryFile(delete=False, suffix=".pdf" if uploaded_file.name.endswith(".pdf") else ".docx") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name
    
    document_text = load_document(temp_file_path)
    
    if document_text:
        db = process_text(document_text)
        if db:
            response = generate_response(db, query)
            if response:
                st.subheader("Response:")
                st.write(response)
            else:
                st.error("Failed to generate response.")
        else:
            st.error("Failed to process text.")
    else:
        st.error("Could not extract text. The PDF may contain scanned images or unsupported formats.")

# Clean up temp file
try:
    os.remove(temp_file_path)
except Exception as e:
    st.warning(f"Failed to delete temp file: {e}")
