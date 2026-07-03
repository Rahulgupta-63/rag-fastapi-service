import os
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from bs4 import BeautifulSoup
import pdfplumber
from dotenv import load_dotenv

load_dotenv()

_embedding_model = None
_groq_client = None

def load_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedding_model

def get_groq_client():
    global _groq_client
    if _groq_client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not found in environment")
        _groq_client = Groq(api_key=api_key)
    return _groq_client

def load_html(content):
    soup = BeautifulSoup(content, "html.parser")
    chunks = []
    for code in soup.find_all(["pre", "code"]):
        text = code.get_text(separator=" ").strip()
        if len(text) > 30:
            chunks.append(text)
    for tag in soup.find_all(["p", "h1", "h2", "h3", "li"]):
        text = tag.get_text(separator=" ").strip()
        if len(text) > 50:
            chunks.append(text)
    return chunks

def load_pdf(file):
    chunks = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                for line in text.split("\n"):
                    if len(line.strip()) > 50:
                        chunks.append(line.strip())
    return chunks

def setup_collection(chunks):
    embedding_model = load_embedding_model()
    client = chromadb.Client()
    try:
        client.delete_collection("study_notes")
    except:
        pass
    collection = client.create_collection("study_notes")
    embeddings = embedding_model.encode(chunks).tolist()
    ids = [f"note{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)
    return collection

def retrieve(question, collection):
    embedding_model = load_embedding_model()
    q_embedding = embedding_model.encode(question).tolist()
    results = collection.query(query_embeddings=[q_embedding], n_results=10)
    return results['documents'][0]