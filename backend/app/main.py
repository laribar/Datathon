# === FILE: backend/app/main.py ===
import os
from functools import lru_cache
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Datathon API", version="1.0.0")

# ---- Health check leve (Render detecta a porta) ----
@app.get("/healthz")
def healthz():
    return {"status": "ok"}

# ---- Exemplo: endpoint root opcional ----
@app.get("/")
def root():
    return {"service": "datathon-backend", "ok": True}

# ======== LAZY LOAD DO MODELO ========
# Carregue peso pesado apenas quando necessário
class MatchInput(BaseModel):
    job_text: str
    resume_text: str

@lru_cache(maxsize=1)
def get_embedder():
    # Importes pesados aqui dentro (evita carregar no startup)
    import torch
    from sentence_transformers import SentenceTransformer

    # Limita CPU threads para não estourar memória
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    # Use um modelo pequeno para caber em 512Mi
    model_name = os.getenv("ST_MODEL", "sentence-transformers/paraphrase-MiniLM-L3-v2")
    model = SentenceTransformer(model_name, device="cpu")
    return model

def cosine_sim(a, b):
    import numpy as np
    # a, b são vetores 1d
    na = a / (np.linalg.norm(a) + 1e-9)
    nb = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(na, nb))

@app.post("/match")
def match(payload: MatchInput):
    # Lazy load do modelo aqui
    model = get_embedder()

    # Inferência sem grad (economia de memória)
    import torch
    with torch.no_grad():
        job_vec = model.encode(payload.job_text, normalize_embeddings=False)
        res_vec = model.encode(payload.resume_text, normalize_embeddings=False)

    score = cosine_sim(job_vec, res_vec)
    return {"score": score}
