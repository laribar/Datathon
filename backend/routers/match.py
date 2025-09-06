from __future__ import annotations
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from pathlib import Path
import tempfile
import logging

# Serviço de match (novo ou antigo)
try:
    from backend.services.model import get_service
    _use_new_service = True
except Exception:
    from backend.services import model as match_service
    _use_new_service = False

# Leitor de PDF revisado
from extractor import extract_text

router = APIRouter(prefix="/match", tags=["match"])

logging.basicConfig(level=logging.INFO)

# ---------- Schemas ----------
class MatchRequest(BaseModel):
    vaga_text: str = Field(..., description="Descrição da vaga em texto")
    cv_text: str = Field(..., description="Texto do currículo/candidato")

class MatchResponse(BaseModel):
    score: float
    match: Optional[bool] = None
    threshold: Optional[float] = None
    model: Optional[str] = None
    encoder: Optional[str] = None
    classificacao: Optional[str] = None
    extra: Optional[dict] = None

# ---------- Util ----------
def _run_match(vaga_text: str, cv_text: str) -> MatchResponse:
    """Executa match usando o serviço novo (get_service) ou o antigo (predict_match)."""
    if _use_new_service:
        svc = get_service()
        result = svc.decision(vaga_text, cv_text)
        return MatchResponse(**result)
    else:
        result = match_service.predict_match(vaga_text, cv_text)
        return MatchResponse(score=result["score"], classificacao=result.get("classificacao"))

# ---------- Endpoints ----------
@router.post("", response_model=MatchResponse)
@router.post("/", response_model=MatchResponse)
def post_match(req: MatchRequest):
    return _run_match(req.vaga_text, req.cv_text)

@router.post("/pdf", response_model=MatchResponse)
async def post_match_pdf(
    vaga_text: str = Form(..., description="Descrição da vaga em texto"),
    cv_pdf: UploadFile = File(..., description="Arquivo PDF do currículo"),
):
    # Validação de tipo e tamanho
    if (cv_pdf.content_type or "").lower() != "application/pdf":
        raise HTTPException(status_code=400, detail="Envie um arquivo PDF válido.")

    file_bytes = await cv_pdf.read()
    max_size_mb = 15
    if len(file_bytes) > max_size_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Arquivo muito grande (> {max_size_mb} MB).")

    # Extrai texto do PDF
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = Path(tmp.name)
        cv_text = extract_text(str(tmp_path))
    except Exception as e:
        logging.error(f"Erro na extração do PDF {cv_pdf.filename}: {e}")
        raise HTTPException(status_code=400, detail=f"Erro ao extrair texto do PDF: {e}")
    finally:
        try:
            if 'tmp_path' in locals() and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    if not cv_text or len(cv_text.strip()) < 20:
        raise HTTPException(status_code=422, detail="Texto extraído é muito curto ou ilegível.")

    resp = _run_match(vaga_text, cv_text)
    resp.extra = {
        "chars_extraidos": len(cv_text),
        "arquivo": cv_pdf.filename,
    }
    return resp
