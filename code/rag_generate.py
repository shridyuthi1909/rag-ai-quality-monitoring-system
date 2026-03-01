from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from openai import OpenAI

client = OpenAI()

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
Answer the question using ONLY the context below.
If answer is not in context, say "Not found in provided documents".

Context:
{context}

Question: {query}
"""

response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": prompt}],
)

print("\nAI Answer:\n")
print(response.choices[0].message.content)