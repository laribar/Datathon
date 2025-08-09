# backend/services/model.py
import numpy as np
import joblib
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ============================
# Caminhos
# ============================
BASE_DIR = Path(__file__).resolve().parent.parent  # backend/
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "modelo_match_baseline.pkl"
ENCODER_PATH = MODEL_DIR / "sbert_encoder"

# ============================
# Carregar modelo e encoder
# ============================
print("🔹 Carregando modelo de match e encoder...")
model = joblib.load(MODEL_PATH)
encoder = SentenceTransformer(str(ENCODER_PATH))

# ============================
# Função para gerar features
# ============================
def build_features(vaga_text: str, cv_text: str) -> np.ndarray:
    vaga_emb = encoder.encode([vaga_text])[0]
    cv_emb = encoder.encode([cv_text])[0]

    features = np.hstack([
        vaga_emb,
        cv_emb,
        np.abs(vaga_emb - cv_emb),
        vaga_emb * cv_emb
    ])
    return features.reshape(1, -1)  # formato (1, N)

# ============================
# Função de predição
# ============================
def predict_match(vaga_text: str, cv_text: str) -> dict:
    feats = build_features(vaga_text, cv_text)
    prob = model.predict_proba(feats)[0, 1]
    pred = int(prob >= 0.5)
    return {
        "score": float(prob),
        "classificacao": "Alto Potencial" if pred == 1 else "Baixo Potencial"
    }
