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

# Custom CSS for improved styling
st.markdown("""
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f5f5f5;
        }
        .stApp {
            background-color: #ffffff;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
            margin-top: 20px;
        }
        .title {
            color: #1f77b4;
            font-weight: bold;
            text-align: center;
            font-size: 2rem;
            margin-bottom: 20px;
        }
        .upload-area {
            border: 2px dashed #1f77b4;
            padding: 20px;
            border-radius: 12px;
            background-color: #fafafa;
            text-align: center;
            margin-bottom: 20px;
        }
        .button {
            background-color: #1f77b4;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 10px 20px;
            border: none;
            cursor: pointer;
            transition: background 0.3s;
        }
        .button:hover {
            background-color: #175d92;
        }
        .response-box {
            background-color: #eef6ff;
            padding: 20px;
            border-radius: 8px;
            border-left: 5px solid #1f77b4;
            font-size: 16px;
            color: #333333;
        }
        .error {
            color: red;
            font-size: 16px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

def ocr_pdf(file_path):
    images = convert_from_path(file_path)
    text = "\n".join([pytesseract.image_to_string(img) for img in images])
    return text.strip() if text.strip() else None

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

def process_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, length_function=len, is_separator_regex=False
    )
    chunks = text_splitter.create_documents([text])
    model_name = "BAAI/bge-small-en"
    embeddings = HuggingFaceBgeEmbeddings(model_name=model_name)
    chunk_texts = [chunk.page_content for chunk in chunks]
    embedding_vectors = embeddings.embed_documents(chunk_texts)
    db = FAISS.from_texts(chunk_texts, embeddings)
    return db

def generate_response(db, query):
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

# Streamlit UI
st.markdown('<h1 class="title">RAG-based Document Query System</h1>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["pdf", "docx"], help="Upload a PDF or DOCX file")

query = st.text_input("Enter your query:")

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
            st.markdown('<div class="response-box">✅ <b>Response:</b><br>' + response + '</div>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="error">❌ Could not extract text. The PDF may contain scanned images or unsupported formats.</p>', unsafe_allow_html=True)

st.markdown(
    """
    ---
    Built with ❤️ using Streamlit, Langchain, FAISS, and Groq.
    """,
    unsafe_allow_html=True
)
