import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from ctransformers import AutoModelForCausalLM
import tempfile
import os

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Resume Analyzer (Local AI)", page_icon="🤖")
st.title("Local Resume Analyzer (RAG + Mistral 7B)")
st.write("Upload a resume and ask questions like a recruiter!")

# -----------------------------
# Embedding Model
# -----------------------------
class LocalEmbeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode(text).tolist()


@st.cache_resource
def load_embeddings():
    return LocalEmbeddings()


embeddings = load_embeddings()

# -----------------------------
# Load LLM
# -----------------------------
@st.cache_resource
def load_llm():
    llm = AutoModelForCausalLM.from_pretrained(
        "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        model_file="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        model_type="mistral",
    )
    return llm


llm = load_llm()

# -----------------------------
# Upload PDF
# -----------------------------
uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    st.success("Resume uploaded successfully!")

    # -----------------------------
    # Load + Chunk + Vector DB
    # -----------------------------
    @st.cache_resource
    def build_vectorstore(pdf_path):
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=220,
            chunk_overlap=40
        )

        docs = splitter.split_documents(documents)

        vectorstore = FAISS.from_documents(docs, embeddings)
        return vectorstore

    vectorstore = build_vectorstore(temp_path)

    # -----------------------------
    # Ask Question
    # -----------------------------
    query = st.text_input("Ask a question about the candidate")

    if st.button("Analyze") and query:

        with st.spinner("Thinking like a recruiter..."):

            docs = vectorstore.similarity_search(query, k=3)
            context = "\n".join([d.page_content for d in docs])

            prompt = f"""
You are an experienced technical recruiter evaluating a candidate.

Your job:
- Explain the candidate professionally
- Summarize strengths clearly
- Never invent information
- If missing say: Not found in document

Context:
{context}

Question:
{query}

Answer:
"""

            response = llm(prompt, max_new_tokens=180, temperature=0.3)

        st.subheader("AI Recruiter Response")
        st.write(response)
