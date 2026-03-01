from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def groundedness_score(context, answer):
    context_embedding = model.encode([context])
    answer_embedding = model.encode([answer])

    score = cosine_similarity(context_embedding, answer_embedding)[0][0]
    return float(score)
