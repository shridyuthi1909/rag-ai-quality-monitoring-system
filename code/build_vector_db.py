from sentence_transformers import SentenceTransformer
import faiss
import os

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Read documents
with open("../data/docs.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Simple chunking (split by paragraphs)
chunks = [p.strip() for p in text.split("\n\n") if p.strip()]

print(f"Total chunks: {len(chunks)}")

# Create embeddings
embeddings = model.encode(chunks)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save index
faiss.write_index(index, "vector.index")

print("Vector database built successfully!")