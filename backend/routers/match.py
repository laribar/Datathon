# backend/routers/match.py
from fastapi import APIRouter
from pydantic import BaseModel
from backend.services import model as match_service

router = APIRouter(prefix="/match", tags=["match"])

# Modelo de entrada para requisição
class MatchRequest(BaseModel):
    vaga_text: str
    cv_text: str

# Endpoint para calcular o score de compatibilidade
@router.post("/")
def get_match_score(data: MatchRequest):
    """
    Recebe descrição da vaga e currículo em texto,
    retorna o score de compatibilidade e classificação.
    """
    return match_service.predict_match(data.vaga_text, data.cv_text)



