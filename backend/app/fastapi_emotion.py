# === FILE: backend/app/fastapi_emotion.py ===
import os
import io
import base64
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np
import cv2  # opencv-python
import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()

# -----------------------------
# Schemas
# -----------------------------
class EmotionRequest(BaseModel):
    image_base64: Optional[str] = Field(
        default=None,
        description="Imagem em base64 (sem prefixo data URI). Ex.: base64.b64encode(open('foto.jpg','rb').read()).decode()",
    )
    image_url: Optional[str] = Field(
        default=None,
        description="URL de uma imagem (http/https). Se informado, é usado se image_base64 não vier.",
    )
    top_n: int = Field(default=1, ge=1, le=5, description="Retornar as N emoções mais prováveis por face")
    detect_faces: bool = Field(default=True, description="Se False, tenta emoção em quadro inteiro (sem detecção)")

class EmotionScore(BaseModel):
    label: str
    score: float

class FaceEmotion(BaseModel):
    box: Optional[List[int]] = None  # [x, y, w, h]
    dominant: Optional[EmotionScore] = None
    scores: Optional[List[EmotionScore]] = None

from pydantic import Field

class EmotionResponse(BaseModel):
    dominant_overall: Optional[EmotionScore] = None
    faces: List[FaceEmotion] = Field(default_factory=list)
    num_faces: int = 0

# -----------------------------
# Lazy loaders
# -----------------------------
@lru_cache(maxsize=1)
def get_fer_detector():
    try:
        from fer import FER
    except Exception as e:
        # Fornece mensagem clara no 503 ao invés de 500 genérico
        from fastapi import HTTPException
        raise HTTPException(
            status_code=503,
            detail="Pacote 'fer' não está instalado no ambiente. Instale com: pip install fer opencv-python numpy requests"
        ) from e
    # mtcnn=False evita dependências extras
    detector = FER(mtcnn=False)
    return detector


# -----------------------------
# Utils
# -----------------------------
def _read_image_from_base64(b64_str: str) -> np.ndarray:
    try:
        data = base64.b64decode(b64_str, validate=True)
    except Exception:
        raise HTTPException(400, "image_base64 inválido (base64 decode falhou).")
    image = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(415, "Não foi possível decodificar a imagem (formato não suportado).")
    return img

def _read_image_from_url(url: str, max_bytes: int = 5_000_000) -> np.ndarray:
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
    except Exception:
        raise HTTPException(400, "Falha ao baixar image_url (URL inválida ou indisponível).")
    if len(r.content) > max_bytes:
        raise HTTPException(413, f"Imagem muito grande (> {max_bytes/1_000_000:.1f}MB).")
    image = np.frombuffer(r.content, dtype=np.uint8)
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(415, "Não foi possível decodificar a imagem baixada.")
    return img

def _prepare_image(req: EmotionRequest) -> np.ndarray:
    if req.image_base64:
        return _read_image_from_base64(req.image_base64)
    if req.image_url:
        return _read_image_from_url(req.image_url)
    raise HTTPException(400, "Informe image_base64 ou image_url.")

def _top_n_scores(score_dict: dict, n: int) -> List[EmotionScore]:
    # score_dict: {'happy':0.88, 'sad':0.01, ...}
    ordered = sorted(score_dict.items(), key=lambda kv: kv[1], reverse=True)
    return [EmotionScore(label=k, score=float(v)) for k, v in ordered[:n]]

# -----------------------------
# Endpoints
# -----------------------------
@router.get("/api/emotion/ping")
def ping():
    # teste rápido do subsistema (sem carregar o FER)
    return {"ok": True}

@router.post("/api/emotion", response_model=EmotionResponse)
def analyze_emotion(req: EmotionRequest):
    """
    Analisa emoções em uma imagem (base64/url).
    - Se detect_faces=True: roda detecção por face e retorna emoções por face.
    - Caso contrário: calcula emoções no quadro inteiro como fallback.
    """
    # Leitura/decodificação
    img = _prepare_image(req)

    # Restrições simples para não explodir memória em vídeos 4K:
    h, w = img.shape[:2]
    max_side = int(os.getenv("EMOTION_MAX_SIDE", "1280"))
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    detector = get_fer_detector()
    faces_out: List[FaceEmotion] = []

    if req.detect_faces:
        # detect_emotions retorna lista com dicts: {'box':[x,y,w,h], 'emotions':{...}}
        try:
            results = detector.detect_emotions(img) or []
        except Exception as e:
            # fallback sem detecção se der erro
            results = []
        if results:
            for r in results:
                box = None
                box_raw = r.get("box")
                if box_raw is not None:
                    arr = np.asarray(box_raw).reshape(-1)       # lida com numpy/list/tuple
                    if arr.size >= 4:
                        x, y, w, h = arr[:4].astype(int).tolist()
                        # (opcional) clamp dentro da imagem
                        h_img, w_img = img.shape[:2]
                        x = max(0, min(x, w_img - 1))
                        y = max(0, min(y, h_img - 1))
                        w = max(1, min(w, w_img - x))
                        h = max(1, min(h, h_img - y))
                        box = [x, y, w, h]

                scores_dict = r.get("emotions") or {}
                if not scores_dict:
                    continue  # sem emoções calculadas, ignora essa face

                scores = _top_n_scores(scores_dict, req.top_n)
                dominant = scores[0] if scores else None
                faces_out.append(FaceEmotion(box=box, dominant=dominant, scores=scores))

    # Se não achou faces (ou detect_faces=False), tenta quadro inteiro
    if not faces_out:
        try:
            # FER().top_emotion(img) -> ('happy', 0.83) ou (None, None)
            label, conf = detector.top_emotion(img)  # type: ignore
            if label is not None and conf is not None:
                scores_dict = detector.detect_emotions(img)
                # detect_emotions pode ser pesado; se None, cria um dict simples com a dominante
                if isinstance(scores_dict, list) and scores_dict:
                    # quando detect_emotions retorna com uma face “fake” no quadro inteiro,
                    # extraímos o dict de emoções
                    emotions_map = scores_dict[0].get("emotions", {}) or {}
                else:
                    emotions_map = {label: float(conf)}
                scores = _top_n_scores(emotions_map, req.top_n)
                faces_out.append(
                    FaceEmotion(box=None, dominant=EmotionScore(label=label, score=float(conf)), scores=scores)
                )
        except Exception:
            # retorna vazio se até o fallback falhar
            pass

    # Monta dominante geral pela maior confiança entre faces
    dominant_overall: Optional[EmotionScore] = None
    for f in faces_out:
        if f.dominant:
            if (dominant_overall is None) or (f.dominant.score > dominant_overall.score):
                dominant_overall = f.dominant

    return EmotionResponse(
        dominant_overall=dominant_overall,
        faces=faces_out,
        num_faces=len(faces_out),
    )
