# 🧠 RAG-Based AI Quality Monitoring System

An end-to-end Retrieval-Augmented Generation (RAG) pipeline with automated AI quality evaluation and hallucination detection using semantic similarity scoring.

---

## 📌 Overview

This project implements a production-style RAG architecture that retrieves semantically relevant document chunks using vector similarity search and evaluates LLM responses using groundedness scoring to ensure reliability and reduce hallucination risk.

The system integrates retrieval, generation, evaluation, logging, and reporting into a unified AI observability framework.

---

## ⚙️ Architecture

Documents → Sentence Embeddings → FAISS Vector Index  
                                     ↓  
                             Top-K Context Retrieval  
                                     ↓  
                             Local LLM (Mistral via Ollama)  
                                     ↓  
                             Groundedness Evaluation (Cosine Similarity)  
                                     ↓  
                             SQLite Logging & Analytics  

---

## 🚀 Key Features

- Semantic search using SentenceTransformers
- Efficient vector indexing with FAISS (L2 distance)
- Local LLM inference using Ollama (Mistral)
- Automated batch AI testing framework
- Hallucination detection via cosine similarity scoring
- AI observability using SQLite database logging
- Performance distribution analysis and reporting

---

## 📊 Results

- Average groundedness score: **0.83**
- Consistent retrieval precision
- Reduced hallucination risk through context grounding

---

## 🛠 Tech Stack

Python  
SentenceTransformers  
FAISS  
Ollama (Local Mistral LLM)  
SQLite  
Scikit-learn  
Matplotlib  

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python build_vector_db.py
python auto_rag_test.py
python quality_report.py
