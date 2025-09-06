import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import joblib
import numpy as np
import os

# ============================
# Configurações do ambiente
# ============================
# Define o caminho base do projeto. No Render, ele será a raiz do seu repositório.
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

# Instancia a aplicação FastAPI
app = FastAPI(title="Serviço de Ranking de Candidatos")

# ============================
# Carregar os modelos e dados na inicialização da API
# ============================
def load_resources():
    try:
        # Carrega o encoder
        encoder = SentenceTransformer(str(MODEL_DIR / "sbert_encoder"))
        
        # Carrega o dataset completo dos candidatos
        data_path = PROJECT_ROOT / "data" / "processed" / "pairs.parquet"
        df_candidates = pd.read_parquet(data_path)
        df_candidates = df_candidates.fillna({"cv_text": ""})
        
        # Gerar os embeddings dos candidatos (ou carregar se já existirem)
        print("Gerando embeddings dos candidatos para a API...")
        cv_embeddings = encoder.encode(df_candidates["cv_text"].tolist(), convert_to_tensor=True, show_progress_bar=False)
        df_candidates['cv_embeddings'] = cv_embeddings.tolist()

        return encoder, df_candidates

    except Exception as e:
        raise RuntimeError(f"Erro ao carregar recursos: {e}")

# Executa o carregamento dos recursos na inicialização da API
encoder, df_candidates = load_resources()

# ============================
# Definição do esquema de entrada da API
# ============================
class VagaInput(BaseModel):
    vaga_text: str
    top_n: int = 5 # Valor padrão

# ============================
# Endpoint da API
# ============================

@app.get("/rankear_candidatos")
async def rankear_get():
    return {"info": "Use POST com vaga_text e top_n para rankear candidatos"}


@app.post("/rankear_candidatos")
async def rankear_candidatos(input_data: VagaInput):
    """
    Recebe o texto de uma vaga e retorna um ranking dos candidatos mais relevantes.
    """
    try:
        # Gerar o embedding da vaga
        vaga_embedding = encoder.encode(input_data.vaga_text, convert_to_tensor=True)

        # Calcular a similaridade de cosseno
        scores = util.cos_sim(vaga_embedding, df_candidates['cv_embeddings'].tolist())[0]

        # Criar DataFrame com scores
        df_scores = pd.DataFrame({
            'id_vaga': df_candidates['id_vaga'],
            'id_candidato': df_candidates['id_candidato'],
            'score_similaridade': scores.tolist()
        })
        
        # Ordenar e retornar o ranking
        ranking = df_scores.sort_values(by='score_similaridade', ascending=False)
        
        # Selecionar os top N
        top_candidatos = ranking.head(input_data.top_n).to_dict(orient='records')
        
        return {"ranking": top_candidatos}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================
# Endpoint de saúde da API
# ============================
@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "API de ranking está funcionando!"}