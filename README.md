# CBSE Science Tutor — AI-Powered RAG Chatbot

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cbse-science-tutor.streamlit.app)

An intelligent tutoring chatbot for Class 10 NCERT Science, built using Retrieval-Augmented Generation (RAG).

## Features
-  Answers questions strictly from NCERT Class 10 Science textbooks
-  Conversation memory — understands follow-up questions
-  Multilingual — English, Hindi, Marathi (powered by Sarvam AI)
-  Interactive quiz with scoring on any topic
-  Out-of-syllabus detection with honest labeling

## Tech Stack
| Component | Technology |
|---|---|
| LLM | Sarvam-M (via Sarvam AI) |
| RAG Framework | LangChain |
| Vector Store | FAISS |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Frontend | Streamlit |
| Document Parsing | PyMuPDF |

## Architecture
NCERT PDFs → PyMuPDF → Text Chunks → HuggingFace Embeddings → FAISS Index
User Query → FAISS Retrieval → LangChain RAG Chain → Sarvam-M LLM → Answer

## Run Locally
```bash
git clone https://github.com/sayalidudhane22/CBSE_TUTOR_rag_llm_AI
cd CBSE_TUTOR_rag_llm_AI
pip install -r requirements.txt
streamlit run app.py
```
Add your `SARVAM_API_KEY` in a `.env` file before running.