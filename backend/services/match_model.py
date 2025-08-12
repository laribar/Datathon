# === BEGIN FILE: backend/services/match_model.py
"""
Adaptador retrocompatível para o serviço de match.
Permite que código antigo continue funcionando sem duplicar lógica.
"""

from __future__ import annotations
from typing import Dict
from backend.services.model import get_service

def predict_match(vaga_text: str, cv_text: str) -> Dict:
    """
    Compatível com a versão antiga do serviço.
    Retorna um dicionário com 'score' e 'classificacao' (compra/venda).
    """
    svc = get_service()
    result = svc.decision(vaga_text, cv_text)
    return {
        "score": result["score"],
        "classificacao": "match" if result["match"] else "no-match",
        "threshold": result["threshold"],
        "model": result["model"],
        "encoder": result["encoder"]
    }
# === END FILE
