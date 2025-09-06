# === FILE: backend/app/main.py ===
import os
from functools import lru_cache
from fastapi import FastAPI
from pydantic import BaseModel

from fastapi_emotion import router as emotion_router  # <-- ADICIONE
app = FastAPI(title="Datathon API", version="1.0.0")
app.include_router(emotion_router)  # <-- ADICIONE

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"service": "datathon-backend", "ok": True}

class MatchInput(BaseModel):
    job_text: str
    resume_text: str

@lru_cache(maxsize=1)
def get_embedder():
    import torch
    from sentence_transformers import SentenceTransformer
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    model_name = os.getenv("ST_MODEL", "sentence-transformers/paraphrase-MiniLM-L3-v2") 
    model = SentenceTransformer(model_name, device="cpu")
    return model

def cosine_sim(a, b):
    import numpy as np
    na = a / (np.linalg.norm(a) + 1e-9)
    nb = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(na, nb))

@app.post("/match")
def match(payload: MatchInput):
    model = get_embedder()
    import torch
    with torch.no_grad():
        job_vec = model.encode(payload.job_text, normalize_embeddings=False)
        res_vec = model.encode(payload.resume_text, normalize_embeddings=False)
    score = cosine_sim(job_vec, res_vec)
    return {"score": score}
