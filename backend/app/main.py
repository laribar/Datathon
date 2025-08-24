import os
import openai
from functools import lru_cache
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

# Certifique-se de que a biblioteca openai está instalada: pip install openai

# Adicione sua chave de API do GPT como uma variável de ambiente
# Ex: openai.api_key = os.getenv("OPENAI_API_KEY")
# Acessa a variável de ambiente OPENAI_API_KEY
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI(title="Datathon API", version="1.0.0")

# --- Configuração de CORS ---
origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- Fim Configuração de CORS ---

@app.get("/healthz")
def healthz():
    """Rota de verificação de saúde da API."""
    return {"status": "ok"}

@app.get("/")
def root():
    """Ponto de entrada principal da API."""
    return {"service": "datathon-backend", "ok": True}

# --- Rota para Análise de Correspondência (Mantida) ---
class MatchInput(BaseModel):
    job_text: str
    resume_text: str

@lru_cache(maxsize=1)
def get_embedder():
    """Carrega e armazena em cache o modelo de embeddings de texto."""
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
    """Calcula a similaridade de cosseno entre dois vetores."""
    import numpy as np
    na = a / (np.linalg.norm(a) + 1e-9)
    nb = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(na, nb))

@app.post("/match")
def match(payload: MatchInput):
    """Calcula a similaridade de correspondência entre dois textos."""
    model = get_embedder()
    import torch
    with torch.no_grad():
        job_vec = model.encode(payload.job_text, normalize_embeddings=False)
        res_vec = model.encode(payload.resume_text, normalize_embeddings=False)
    score = cosine_sim(job_vec, res_vec)
    return {"score": score}

# --- Nova Rota para Análise Combinada de Emoção e Texto ---
class InterviewAnalysisInput(BaseModel):
    image_base64: str
    transcribed_text: Optional[str] = None # Texto da frase para análise

# Função para chamar a API do GPT para análise de sentimento
def analyze_text_sentiment_with_gpt(text: str):
    """Envia o texto para a API do GPT para análise de sentimento."""
    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=f"Analise o sentimento do texto a seguir. Classifique como 'Positivo', 'Negativo' ou 'Neutro'.\n\nTexto: {text}\nSentimento:",
            max_tokens=50
        )
        return response.choices[0].text.strip()
    except Exception as e:
        # Trate erros de API ou de rede
        return f"Erro na análise de sentimento: {e}"

# Rota para receber dados de imagem (emoção facial) e texto (sentimento)
@app.post("/api/analyze")
async def analyze_interview_data(payload: InterviewAnalysisInput):
    """
    Combina a análise de emoção facial com a análise de sentimento do texto
    transcrito para fornecer uma análise mais completa.
    """
    
    # 1. Análise de Emoção Facial (usando sua rota já existente)
    # Você precisaria de uma função que chame a lógica do seu 'emotion_router'
    # Como não temos acesso a esse código, vamos simular uma resposta.
    # Em um projeto real, você chamaria sua função de análise de emoção aqui.
    facial_emotion_data = {
        "dominant_overall": {"label": "neutro", "score": 0.75},
        "faces": []
    }

    # 2. Análise de Sentimento do Texto (via GPT)
    text_sentiment = "N/A"
    if payload.transcribed_text:
        text_sentiment = analyze_text_sentiment_with_gpt(payload.transcribed_text)
        
    # 3. Combinação e Retorno dos Resultados
    return {
        "facial_emotion": facial_emotion_data,
        "text_sentiment": text_sentiment
    }