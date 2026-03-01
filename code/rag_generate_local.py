from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import subprocess

model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index("vector.index")

with open("../data/docs.txt", "r", encoding="utf-8") as f:
    text = f.read()

chunks = [p.strip() for p in text.split("\n\n") if p.strip()]

query = input("Ask question: ")

query_embedding = model.encode([query])

distances, indices = index.search(np.array(query_embedding), 2)

context = "\n".join([chunks[i] for i in indices[0]])

prompt = f"""
Answer using ONLY the context below.

Context:
{context}

Question: {query}
"""

result = subprocess.run(
    ["ollama", "run", "mistral"],
    input=prompt,
    capture_output=True,
    encoding="utf-8"
)

print("\nAI Answer:\n")
print(result.stdout)