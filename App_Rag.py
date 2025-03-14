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

# Set API key
os.environ["GROQ_API_KEY"] = "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"  # Replace with your actual key

def ocr_pdf(file_path):
    """Extracts text using OCR from image-based PDFs."""
    images = convert_from_path(file_path)
    text = "\n".join([pytesseract.image_to_string(img) for img in images])
    return text.strip() if text.strip() else None

def load_document(file_path):
    """Loads text from a PDF or DOCX file."""
    if file_path.endswith(".pdf"):
        try:
            doc_loader = PyPDFLoader(file_path)
            pages = doc_loader.load()
            extracted_text = "\n".join([page.page_content for page in pages])
            
            # If extracted text is empty, fallback to OCR
            if not extracted_text.strip():
                extracted_text = ocr_pdf(file_path)

            return extracted_text if extracted_text else None
        except Exception:
            return None

    elif file_path.endswith(".docx"):
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    
    return None

def process_text(text):
    """Splits text into chunks and creates embeddings."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, length_function=len, is_separator_regex=False
    )
    chunks = text_splitter.create_documents([text])
    
    # Embeddings
    model_name = "BAAI/bge-small-en"
    embeddings = HuggingFaceBgeEmbeddings(model_name=model_name)
    
    chunk_texts = [chunk.page_content for chunk in chunks]
    embedding_vectors = embeddings.embed_documents(chunk_texts)
    
    db = FAISS.from_texts(chunk_texts, embeddings)
    return db

def generate_response(db, query):
    """Retrieves relevant chunks and generates response using Groq API."""
    contexts = db.similarity_search(query, k=5)
    
    # Construct prompt
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

# Streamlit UI
st.title("RAG-based Document Query System")

uploaded_file = st.file_uploader("Upload a PDF or DOCX document", type=["pdf", "docx"])
query = st.text_input("Enter your query:")

if uploaded_file and query:
    with NamedTemporaryFile(delete=False, suffix=".pdf" if uploaded_file.name.endswith(".pdf") else ".docx") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name  # Ensure we get the correct temp file path
    
    document_text = load_document(temp_file_path)
    
    if document_text:
        db = process_text(document_text)
        response = generate_response(db, query)
        st.subheader("Response:")
        st.write(response)
    else:
        st.error("Could not extract text. The PDF may contain scanned images or unsupported formats.")
