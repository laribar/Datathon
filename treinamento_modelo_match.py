# treinamento_modelo_match.py
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ============================
# Configurações
# ============================
DATA_PATH = Path("data/processed/pairs.parquet")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "modelo_match_baseline.pkl"
ENCODER_NAME = "sbert_encoder"

SBERT_MODEL = "all-MiniLM-L6-v2"  # leve e rápido

# ============================
# 1) Carregar dataset
# ============================
print("🔹 Carregando dataset...")
df = pd.read_parquet(DATA_PATH)

# Garantir que não haja NaN
df = df.fillna({"vaga_text": "", "cv_text": ""})

# ============================
# 2) Gerar embeddings
# ============================
print("🔹 Carregando SentenceTransformer...")
encoder = SentenceTransformer(SBERT_MODEL)

print("🔹 Gerando embeddings das vagas...")
vaga_embeddings = encoder.encode(df["vaga_text"].tolist(), show_progress_bar=True)

print("🔹 Gerando embeddings dos candidatos...")
cv_embeddings = encoder.encode(df["cv_text"].tolist(), show_progress_bar=True)

# ============================
# 3) Criar features combinadas
# ============================
print("🔹 Combinando embeddings...")
# Estratégia simples: concatenação + diferença absoluta + produto elemento a elemento
features = np.hstack([
    vaga_embeddings,
    cv_embeddings,
    np.abs(vaga_embeddings - cv_embeddings),
    vaga_embeddings * cv_embeddings
])

labels = df["label"].values

# ============================
# 4) Dividir treino/teste
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels
)

# ============================
# 5) Treinar modelo baseline
# ============================
print("🔹 Treinando modelo Logistic Regression...")
model = LogisticRegression(max_iter=200, class_weight="balanced")
model.fit(X_train, y_train)

# ============================
# 6) Avaliar modelo
# ============================
y_pred = model.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, digits=3))

print("📌 Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

# ============================
# 7) Salvar modelo e encoder
# ============================
print("💾 Salvando modelo e encoder...")
joblib.dump(model, MODEL_DIR / MODEL_NAME)
encoder.save(MODEL_DIR / ENCODER_NAME)

print(f"✅ Modelo salvo em: {MODEL_DIR / MODEL_NAME}")
print(f"✅ Encoder salvo em: {MODEL_DIR / ENCODER_NAME}")
