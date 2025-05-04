import json
import faiss
import numpy as np
from fastapi import FastAPI, Query
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware

from sklearn.preprocessing import normalize

# Load model, index, metadata
model = SentenceTransformer('all-mpnet-base-v2') # Mean Recall@10: 0.098 , Mean MAP@10: 0.093

index = faiss.read_index("shl_index.faiss")

with open("shl_index_metadata.json", "r") as f:
    assessments = json.load(f)
    

# FastAPI setup
app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],    
)

class QueryRequest(BaseModel):
    query: str
    
@app.get('/')
def read_root():
    return {"msg" : "SHL Assessment Recommedation API is running!"}

@app.get('/health')
def health():
    return {"status": "healthy"}

@app.post('/recommend')
def recommend(req: QueryRequest):
    query_embedding = model.encode([req.query])
    # normalize query
    query_embedding = normalize(query_embedding)
    
    scores, indices = index.search(query_embedding, 10)
    
    results = []
    for i, ind in enumerate(indices[0]):
        results.append({
            "url": assessments[ind]['url'],
            "adaptive_support": assessments[ind]['adaptive_support'],
            "description": assessments[ind]['description'],
            "duration": assessments[ind]['duration'],
            'remote_support': assessments[ind]['remote_support'],
            "test_types": assessments[ind]['test_types'],
            # "name": assessments[ind]['name'],
            # "cosine_similarity_score": float(scores[0][i])  # Cosine similarity ∈ [-1, 1]
        })
        
    return {"results": results}