import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier # Importando o modelo mais avançado

# ============================
# Configurações
# ============================
# Usando a biblioteca 'pathlib' para manter a consistência
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "pairs.parquet"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "modelo_match_xgboost.pkl"
ENCODER_NAME = "sbert_encoder"

SBERT_MODEL = "all-MiniLM-L6-v2" # leve e rápido

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
# 5) Treinar e Otimizar modelo XGBoost
# ============================
print("🔹 Otimizando modelo XGBoost...")

# Calculando a proporção de casos positivos para lidar com o desequilíbrio de classes
pos_count = np.sum(y_train == 1)
neg_count = np.sum(y_train == 0)
scale_pos_weight = neg_count / pos_count

# Definindo o espaço de parâmetros para a busca aleatória
param_distributions = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 10],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# Configurando a busca aleatória (RandomizedSearchCV)
# Usamos f1 como métrica para balancear precisão e recall
xgb_model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1,
    scale_pos_weight=scale_pos_weight # Ajusta para o desequilíbrio
)

search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_distributions,
    n_iter=20, # Número de combinações a testar, aumente para mais precisão
    scoring='f1', # Otimizando para o f1-score
    cv=5, # 5-fold cross-validation
    verbose=1,
    random_state=42
)

# Treinando o modelo com a busca aleatória
search.fit(X_train, y_train)

# Obtendo o melhor modelo encontrado
best_model = search.best_estimator_

print(f"✅ Melhor F1-score: {search.best_score_:.4f}")
print(f"✅ Melhores parâmetros: {search.best_params_}")

# ============================
# 6) Avaliar o melhor modelo
# ============================
y_pred = best_model.predict(X_test)
print("\n📊 Classification Report (XGBoost Otimizado):")
print(classification_report(y_test, y_pred, digits=3))

print("📌 Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

# ============================
# 7) Salvar o melhor modelo e encoder
# ============================
print("💾 Salvando o melhor modelo e encoder...")
joblib.dump(best_model, MODEL_DIR / MODEL_NAME)
encoder.save(MODEL_DIR / ENCODER_NAME)

print(f"✅ Modelo salvo em: {MODEL_DIR / MODEL_NAME}")
print(f"✅ Encoder salvo em: {MODEL_DIR / ENCODER_NAME}")