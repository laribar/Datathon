# === BEGIN FILE: backend/services/model.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Any
import numpy as np
import joblib

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

# Locais possíveis (compatível com seu print e com quem usa /models fora do backend)
CANDIDATE_DIRS = [
    Path("backend/models"),
    Path("models"),
]

MODEL_FILENAME = "modelo_match_baseline.pkl"
ENCODER_DIRNAME = "sbert_encoder"
DEFAULT_THRESHOLD = 0.5  # seu treino não salva threshold; usamos 0.5

class MatchModelService:
    def __init__(self) -> None:
        self.model_path, self.encoder_path = self._resolve_paths()
        self.encoder = self._load_encoder(self.encoder_path)
        self.model: LogisticRegression | None = self._load_model(self.model_path)
        self.threshold: float = DEFAULT_THRESHOLD

    @staticmethod
    def _resolve_paths() -> tuple[Path, Path]:
        model_path = None
        encoder_path = None
        for base in CANDIDATE_DIRS:
            m = base / MODEL_FILENAME
            e = base / ENCODER_DIRNAME
            if model_path is None and m.exists():
                model_path = m
            if encoder_path is None and e.exists():
                encoder_path = e
        # fallback: permite rodar com só encoder (baseline por cosseno)
        if encoder_path is None:
            # último recurso: baixar do hub pelo nome usado no treino
            # (não falha se sem internet; só dá erro quando tentar predição)
            encoder_path = Path("all-MiniLM-L6-v2")
        if model_path is None:
            # baseline sem modelo
            model_path = Path("__missing__.pkl")
        return model_path, encoder_path

    @staticmethod
    def _load_encoder(path: Path) -> SentenceTransformer:
        # Se existe pasta salva via .save(), carregar local;
        # senão, carregar por nome do hub (ex.: "all-MiniLM-L6-v2").
        try:
            if path.exists() and path.is_dir():
                return SentenceTransformer(str(path))
            else:
                return SentenceTransformer(str(path))  # nome no hub
        except Exception as e:
            raise RuntimeError(f"Falha ao carregar encoder em '{path}': {e}")

    @staticmethod
    def _load_model(path: Path) -> LogisticRegression | None:
        try:
            if path.exists():
                return joblib.load(path)
            return None
        except Exception:
            return None

    # === FEATURES: **iguais ao seu treino** ===
    @staticmethod
    def _combine_features(vaga_emb: np.ndarray, cv_emb: np.ndarray) -> np.ndarray:
        # concatenação + diferença absoluta + produto elemento a elemento
        return np.hstack([
            vaga_emb,
            cv_emb,
            np.abs(vaga_emb - cv_emb),
            vaga_emb * cv_emb,
        ])

    def _embed(self, text: str) -> np.ndarray:
        # no treino você não normalizou; aqui mantemos o padrão
        return self.encoder.encode([text], show_progress_bar=False)[0]

    def predict_proba(self, vaga_text: str, cv_text: str) -> float:
        u = self._embed(vaga_text)
        v = self._embed(cv_text)

        if self.model is None:
            # Baseline por cosseno (normaliza só para o cálculo do cosseno)
            u_n = u / (np.linalg.norm(u) + 1e-12)
            v_n = v / (np.linalg.norm(v) + 1e-12)
            cos = float(np.dot(u_n, v_n))
            return (cos + 1.0) / 2.0

        X = self._combine_features(u, v).reshape(1, -1)
        return float(self.model.predict_proba(X)[0, 1])

    def decision(self, vaga_text: str, cv_text: str) -> Dict[str, Any]:
        score = self.predict_proba(vaga_text, cv_text)
        return {
            "score": score,
            "match": bool(score >= self.threshold),
            "threshold": self.threshold,
            "model": "LogReg" if self.model is not None else "cosine-baseline",
            "encoder": str(self.encoder_path),
        }

# singleton para reutilizar pesos
_service_singleton: MatchModelService | None = None

def get_service() -> MatchModelService:
    global _service_singleton
    if _service_singleton is None:
        _service_singleton = MatchModelService()
    return _service_singleton
# === END FILE
    