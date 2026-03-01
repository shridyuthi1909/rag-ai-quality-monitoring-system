from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load saved index
index = faiss.read_index("vector.index")

# Reload original chunks
with open("../data/docs.txt", "r", encoding="utf-8") as f:
    text = f.read()

chunks = [p.strip() for p in text.split("\n\n") if p.strip()]

# Ask question
query = input("Enter your question: ")

# Convert question to embedding
query_embedding = model.encode([query])

# Search in FAISS
k = 2
distances, indices = index.search(np.array(query_embedding), k)

print("\nTop Retrieved Chunks:\n")

for i in indices[0]:
    print("----")
    print(chunks[i])