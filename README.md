#  RAG Document Chatbot

This project is an AI assistant that reads a PDF and answers questions from it.

Instead of guessing answers, the system searches the document and then generates the response.  
This helps reduce wrong answers (hallucinations).

---

## 🚀 Features
- Ask questions from any PDF
- Summarize resumes or notes
- Answers based only on document content
- Works locally and on Google Colab
- Chat interface using Streamlit

---

## 🧠 Models Used

### Local
- **Mistral-7B-Instruct (compressed / quantized)** → generates answers
- **all-MiniLM-L6-v2 embeddings** → understands meaning of text
- **FAISS** → searches similar text in document

### Colab (GPU)
- Same Mistral model but full version (no compression)
- Gives better quality responses

---

## Why Google Colab was used
My laptop cannot run the full model because it needs a GPU and large memory.  
So locally I used a compressed model (faster but less accurate).

Colab provides a free GPU, so the full model can run and give better answers.

Local shows the project can run anywhere.  
Colab shows the real capability of the model.

---

## 🏗️ How it works

PDF → Split into chunks → Convert to vectors → Search relevant text → Send to AI → Generate answer

The AI does not guess — it answers from the document.

---
