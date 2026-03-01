from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import subprocess
from evaluator import groundedness_score
from db_manager import init_db, insert_result

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load vector DB
index = faiss.read_index("vector.index")

# Load documents
with open("../data/docs.txt", "r", encoding="utf-8") as f:
    text = f.read()

chunks = [p.strip() for p in text.split("\n\n") if p.strip()]

# Load test prompts
with open("../data/test_prompts.txt", "r") as f:
    prompts = [line.strip() for line in f if line.strip()]

def ask_llm(context, question):
    prompt = f"""
Answer using ONLY the context below.

Context:
{context}

Question: {question}
"""
    result = subprocess.run(
        ["ollama", "run", "mistral"],
        input=prompt,
        capture_output=True,
        encoding="utf-8"
    )
    return result.stdout.strip()

print("\nRunning AI Quality Tests...\n")
init_db()
for q in prompts:
    query_embedding = model.encode([q])
    _, indices = index.search(np.array(query_embedding), 2)

    context = "\n".join([chunks[i] for i in indices[0]])

    answer = ask_llm(context, q)
    score = groundedness_score(context, answer)

    print("QUESTION:", q)
    print("ANSWER:", answer)
    print("-" * 60)
    print("GROUNDEDNESS SCORE:", round(score, 3))
    insert_result(q, answer, score)
    