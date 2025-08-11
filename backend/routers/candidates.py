# backend/app/routers/candidates.py
import json
import os
import random
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, Any

router = APIRouter(prefix="/candidates", tags=["candidates"])

# Caminho para o arquivo JSON de candidatos (applicants)
APPLICANTS_PATH = os.path.join("backend", "data", "applicants.json")

# Função para carregar e transformar candidatos
def _load_candidates():
    try:
        with open(APPLICANTS_PATH, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        raise HTTPException(404, f"Arquivo {APPLICANTS_PATH} não encontrado")
    except json.JSONDecodeError as e:
        raise HTTPException(500, f"Erro ao decodificar JSON: {e}")
    except Exception as e:
        raise HTTPException(500, f"Erro lendo {APPLICANTS_PATH}: {e}")

    candidates = []
    for cand_id, data in raw_data.items():
        infos = data.get("infos_basicas", {})
        prof = data.get("informacoes_profissionais", {})

        # Formatar data
        raw_date = infos.get("data_criacao")
        try:
            parsed_date = datetime.strptime(raw_date, "%d-%m-%Y %H:%M:%S")
            application_date = parsed_date.strftime("%d/%m/%Y")
        except Exception:
            application_date = ""

        candidates.append({
            "id": cand_id,
            "name": infos.get("nome", "").strip(),
            "email": infos.get("email", "").strip(),
            "position": prof.get("titulo_profissional", "").strip(),
            "department": prof.get("area_atuacao", "").strip(),
            "application_date": application_date,
            "compatibility": random.randint(60, 100),  # valor simulado
            "status": "novo",  # valor padrão
        })

    return candidates

# Função auxiliar para parsear datas
def parse_date(date_str: str):
    try:
        return datetime.strptime(date_str, "%d/%m/%Y")
    except (ValueError, TypeError):
        return None

@router.get("")
def list_candidates(
    search: Optional[str] = Query(None, description="Buscar por nome, email ou posição"),
    status: Optional[str] = Query(None, description="Filtrar por status"),
    position: Optional[str] = Query(None, description="Filtrar por posição"),
    min_score: Optional[int] = Query(None, description="Filtrar por compatibilidade mínima"),
    max_score: Optional[int] = Query(None, description="Filtrar por compatibilidade máxima"),
    start_date: Optional[str] = Query(None, description="Data inicial (dd/mm/yyyy)"),
    end_date: Optional[str] = Query(None, description="Data final (dd/mm/yyyy)"),
    order_by: Optional[str] = Query(None, description="Ordenar por: score ou date"),
    order_dir: Optional[str] = Query("desc", description="Direção de ordenação: asc ou desc")
) -> Dict[str, Any]:
    """
    Retorna a lista de candidatos filtrados, com estatísticas.
    """
    candidates = _load_candidates()

    # Busca por texto
    if search:
        search_lower = search.lower()
        candidates = [
            c for c in candidates
            if search_lower in c.get("name", "").lower()
            or search_lower in c.get("email", "").lower()
            or search_lower in c.get("position", "").lower()
        ]

    # Filtro por status
    if status:
        candidates = [c for c in candidates if c.get("status", "").lower() == status.lower()]

    # Filtro por posição
    if position:
        candidates = [c for c in candidates if position.lower() in c.get("position", "").lower()]

    # Filtro por score mínimo
    if min_score is not None:
        candidates = [c for c in candidates if c.get("compatibility", 0) >= min_score]

    # Filtro por score máximo
    if max_score is not None:
        candidates = [c for c in candidates if c.get("compatibility", 0) <= max_score]

    # Filtro por período de aplicação
    if start_date:
        start_dt = parse_date(start_date)
        if start_dt:
            candidates = [c for c in candidates if parse_date(c.get("application_date")) and parse_date(c.get("application_date")) >= start_dt]

    if end_date:
        end_dt = parse_date(end_date)
        if end_dt:
            candidates = [c for c in candidates if parse_date(c.get("application_date")) and parse_date(c.get("application_date")) <= end_dt]

    # Ordenação
    if order_by:
        reverse_order = order_dir.lower() == "desc"
        if order_by == "score":
            candidates.sort(key=lambda x: x.get("compatibility", 0), reverse=reverse_order)
        elif order_by == "date":
            candidates.sort(key=lambda x: parse_date(x.get("application_date")) or datetime.min, reverse=reverse_order)

    # Estatísticas
    stats = {
        "total": len(candidates),
        "new": sum(1 for c in candidates if c.get("status") == "novo"),
        "interviewing": sum(1 for c in candidates if c.get("status") == "entrevistando"),
        "approved": sum(1 for c in candidates if c.get("status") == "aprovado"),
        "rejected": sum(1 for c in candidates if c.get("status") == "rejeitado"),
        "highMatch": sum(1 for c in candidates if c.get("compatibility", 0) >= 80)
    }

    return {
        "stats": stats,
        "candidates": candidates
    }

@router.get("/{candidate_id}")
def get_candidate(candidate_id: str):
    """
    Retorna um candidato pelo ID.
    """
    for c in _load_candidates():
        if str(c.get("id")) == str(candidate_id):
            return c
    raise HTTPException(404, "Candidate not found")
