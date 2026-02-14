import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from ctransformers import AutoModelForCausalLM

# ---------------- PAGE ----------------
st.set_page_config(page_title="Local Document Chatbot", layout="wide")
st.title("Local AI Document Chatbot")

# ---------------- EMBEDDINGS ----------------
class LocalEmbeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode(text).tolist()

# ---------------- LLM ----------------
@st.cache_resource
def load_llm():
    llm = AutoModelForCausalLM.from_pretrained(
        "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        model_file="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        model_type="mistral",
        gpu_layers=0
    )
    return llm

llm = load_llm()

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

# ---------------- VECTORSTORE CREATION ----------------
@st.cache_resource(show_spinner=False)
def create_vectorstore(file):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file.read())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    embeddings = LocalEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    return vectorstore

vectorstore = None
if uploaded_file is not None:
    vectorstore = create_vectorstore(uploaded_file)

# ---------------- CHAT MEMORY ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---------------- USER INPUT ----------------
query = st.chat_input("Ask questions about the uploaded document")

if query:

    if vectorstore is None:
        st.warning("Please upload a PDF first")
        st.stop()

    # Save user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    # Retrieval
    docs_scores = vectorstore.similarity_search_with_score(query, k=4)
    docs = [doc for doc, score in docs_scores if score < 0.6]

    # If no relevant context → stop hallucination
    if len(docs) == 0:
        response = "Not found in document"
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.write(response)
        st.stop()

    context = "\n".join([d.page_content for d in docs])

    # -------- Stage 1: Fact Extraction --------
    fact_prompt = f"""
Extract only the important factual points from the context relevant to the question.
Do not explain. Do not add new information.

Context:
{context}

Question:
{query}

Facts:
"""
    facts = llm(fact_prompt)

    # -------- Stage 2: Final Answer --------
    final_prompt = f"""
You are a professional assistant.

Using ONLY these facts, write a clear natural language answer.
If facts are empty say: Not found in document

Facts:
{facts}

Question:
{query}

Answer:
"""
    response = llm(final_prompt)

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.write(response)
