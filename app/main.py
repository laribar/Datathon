import os
import re
import itertools
from typing import List, Optional, Tuple

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

APP_NAME = "MatchAPI"
APP_VERSION = "1.0.0"

MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
SCORE_LIMIAR = float(os.getenv("SCORE_LIMIAR", "0.75"))

# ---------- Utilidades ----------
_whitespace_re = re.compile(r"\s+")

def clean_text(text: str) -> str:
    if text is None:
        return ""
    text = text.strip().lower()
    text = _whitespace_re.sub(" ", text)
    return text

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))

def proportional_score(similarity: float, limiar: float) -> float:
    if similarity >= limiar:
        return 100.0
    return max(0.0, (similarity / limiar) * 100.0)

# ---------- Schemas ----------
class MatchRequest(BaseModel):
    cv_text: str = Field(..., description="Texto puro do currículo.")
    vaga_text: str = Field(..., description="Texto puro da descrição da vaga.")
    clean: bool = Field(default=True, description="Se True, aplica limpeza básica aos textos.")

class MatchResponse(BaseModel):
    success: bool
    similarity: float = Field(..., ge=0.0, le=1.0)
    score: float = Field(..., ge=0.0, le=100.0)
    limiar_usado: float = Field(..., ge=0.0, le=1.0)
    passed_threshold: bool
    model_name: str
    details: Optional[str] = None

class PairItem(BaseModel):
    cv_text: str
    vaga_text: str

class BatchMatchRequest(BaseModel):
    pairs: List[PairItem]
    clean: bool = True

class BatchMatchResponseItem(BaseModel):
    index: int
    similarity: float
    score: float
    passed_threshold: bool

class BatchMatchResponse(BaseModel):
    success: bool
    limiar_usado: float
    model_name: str
    results: List[BatchMatchResponseItem]

# === Explain Schemas ===
class ExplainRequest(BaseModel):
    cv_text: str = Field(..., description="Texto do currículo (puro).")
    vaga_text: str = Field(..., description="Texto da vaga (puro).")
    clean: bool = True
    top_n: int = Field(default=3, ge=1, le=10)
    min_chars: int = Field(default=25, ge=1, le=500, description="Descarta sentenças muito curtas.")

class ExplainItem(BaseModel):
    rank: int
    similarity: float
    cv_index: int
    vaga_index: int
    cv_snippet: str
    vaga_snippet: str

class ExplainResponse(BaseModel):
    success: bool
    model_name: str
    limiar_usado: float
    overall_similarity: float
    overall_score: float
    top_pairs: List[ExplainItem]

# ---------- App ----------
app = FastAPI(title=APP_NAME, version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ajuste p/ domínios específicos em produção
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carrega o modelo na inicialização do servidor
sbert_encoder = SentenceTransformer(MODEL_NAME)

# (Opcional) Rotas externas
# Se quiser usar um router separado em app/match_routes.py:
# - garanta que exista app/__init__.py
# - e descomente as 2 linhas abaixo
# from .match_routes import router as match_router
# app.include_router(match_router)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "model_name": MODEL_NAME}

@app.get("/version")
def version():
    return {"app": APP_NAME, "version": APP_VERSION}

@app.post("/match", response_model=MatchResponse)
def match(req: MatchRequest):
    cv_text = clean_text(req.cv_text) if req.clean else req.cv_text
    vaga_text = clean_text(req.vaga_text) if req.clean else req.vaga_text

    cv_vec = sbert_encoder.encode(cv_text)
    vaga_vec = sbert_encoder.encode(vaga_text)

    sim = cosine_similarity(np.array(cv_vec), np.array(vaga_vec))
    score = proportional_score(sim, SCORE_LIMIAR)
    passed = sim >= SCORE_LIMIAR

    return MatchResponse(
        success=True,
        similarity=round(sim, 6),
        score=round(score, 2),
        limiar_usado=SCORE_LIMIAR,
        passed_threshold=passed,
        model_name=MODEL_NAME,
        details=None,
    )

@app.post("/match/batch", response_model=BatchMatchResponse)
def match_batch(req: BatchMatchRequest):
    if not req.pairs:
        return BatchMatchResponse(
            success=True, limiar_usado=SCORE_LIMIAR, model_name=MODEL_NAME, results=[]
        )

    cvs = [clean_text(p.cv_text) if req.clean else p.cv_text for p in req.pairs]
    vagas = [clean_text(p.vaga_text) if req.clean else p.vaga_text for p in req.pairs]

    cv_vecs = sbert_encoder.encode(cvs)
    vaga_vecs = sbert_encoder.encode(vagas)

    results: List[BatchMatchResponseItem] = []
    for idx, (cvv, vvv) in enumerate(zip(cv_vecs, vaga_vecs)):
        sim = cosine_similarity(np.array(cvv), np.array(vvv))
        score = proportional_score(sim, SCORE_LIMIAR)
        results.append(
            BatchMatchResponseItem(
                index=idx,
                similarity=round(sim, 6),
                score=round(score, 2),
                passed_threshold=(sim >= SCORE_LIMIAR),
            )
        )

    return BatchMatchResponse(
        success=True,
        limiar_usado=SCORE_LIMIAR,
        model_name=MODEL_NAME,
        results=results,
    )

# ---------- Explain utils ----------
_SENT_SPLIT_RE = re.compile(r"(?<=[\.\?\!\;\:]|\n)\s+")

def split_sentences(text: str, min_chars: int = 25) -> List[str]:
    if not text:
        return []
    t = clean_text(text)
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(t) if p and p.strip()]
    parts = [p for p in parts if len(p) >= min_chars]
    return parts if parts else [t]

def top_n_pairs_by_cosine(
    A: List[str],
    B: List[str],
    encoder: "SentenceTransformer",
    top_n: int = 3,
) -> List[Tuple[int, int, float]]:
    if not A or not B:
        return []
    A_vecs = encoder.encode(A)
    B_vecs = encoder.encode(B)
    pairs = []
    for i, j in itertools.product(range(len(A)), range(len(B))):
        sim = cosine_similarity(np.array(A_vecs[i]), np.array(B_vecs[j]))
        pairs.append((i, j, float(sim)))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:max(1, top_n)]

@app.post("/match/explain", response_model=ExplainResponse)
def match_explain(req: ExplainRequest):
    cv_raw = clean_text(req.cv_text) if req.clean else req.cv_text
    vaga_raw = clean_text(req.vaga_text) if req.clean else req.vaga_text

    cv_vec = sbert_encoder.encode(cv_raw)
    vaga_vec = sbert_encoder.encode(vaga_raw)
    overall_sim = cosine_similarity(np.array(cv_vec), np.array(vaga_vec))
    overall_score = proportional_score(overall_sim, SCORE_LIMIAR)

    cv_sents = split_sentences(cv_raw, min_chars=req.min_chars)
    vaga_sents = split_sentences(vaga_raw, min_chars=req.min_chars)

    pairs = top_n_pairs_by_cosine(cv_sents, vaga_sents, sbert_encoder, top_n=req.top_n)

    top_items: List[ExplainItem] = []
    for rank, (i, j, sim) in enumerate(pairs, start=1):
        top_items.append(
            ExplainItem(
                rank=rank,
                similarity=round(sim, 6),
                cv_index=i,
                vaga_index=j,
                cv_snippet=cv_sents[i],
                vaga_snippet=vaga_sents[j],
            )
        )

    return ExplainResponse(
        success=True,
        model_name=MODEL_NAME,
        limiar_usado=SCORE_LIMIAR,
        overall_similarity=round(overall_sim, 6),
        overall_score=round(overall_score, 2),
        top_pairs=top_items
    )
