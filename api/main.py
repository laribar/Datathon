import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import joblib
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

# ============================
# Configurações do ambiente
# ============================
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
ENCODER_NAME = "sbert_encoder"

# Define o limiar de score perfeito para o cálculo proporcional
SCORE_LIMIAR = 0.75

# Instancia a aplicação FastAPI
app = FastAPI(title="Serviço de Ranking de Candidatos")

# ============================
# Carregar os recursos na inicialização da API
# ============================
def load_resources():
    try:
        # Carrega o encoder de Deep Learning (SentenceTransformer)
        encoder = SentenceTransformer(str(MODEL_DIR / ENCODER_NAME))
        return encoder

    except Exception as e:
        raise RuntimeError(f"Erro ao carregar recursos: {e}")

encoder = load_resources()

# ============================
# Funções de auxílio
# ============================
def clean_text(text: str) -> str:
    """Limpa o texto para pré-processamento."""
    text = text.lower()
    text = ' '.join(text.split())
    return text

# ============================
# Definição do esquema de entrada da API
# ============================
class MatchInput(BaseModel):
    cv_text: str
    vaga_text: str

# ============================
# Endpoint da API
# ============================
@app.post("/match")
async def calculate_match(input_data: MatchInput):
    """
    Recebe o texto de um currículo e uma vaga, e retorna o score de match.
    """
    try:
        # Limpa os textos
        cv_text = clean_text(input_data.cv_text)
        vaga_text = clean_text(input_data.vaga_text)
        
        # Gera os embeddings
        cv_embedding = encoder.encode([cv_text])
        vaga_embedding = encoder.encode([vaga_text])

        # Calcula a similaridade de cosseno
        similarity = cosine_similarity(cv_embedding, vaga_embedding)[0][0]

        # Aplica a lógica proporcional
        if similarity >= SCORE_LIMIAR:
            final_score = 100.0
        else:
            final_score = (similarity / SCORE_LIMIAR) * 100
        
        return {"match_percentage": round(final_score, 2)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================
# Endpoint de saúde da API
# ============================
@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "API de match está funcionando!"}
