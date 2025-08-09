# backend/app/routers/candidates.py
import json
from fastapi import APIRouter, HTTPException, Query
from typing import List, Literal
from backend.app.models.catalog import Candidate


router = APIRouter(prefix="/candidates", tags=["candidates"])

APP_PATH = "backend/data/applicants.json"
PROS_PATH = "backend/data/prospects.json"

def _load(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except Exception as e:
        raise HTTPException(500, f"Erro lendo {path}: {e}")

def _merge_all():
    apps = [{**c, "id": f"app_{c.get('id', i)}", "source": "applicants"} for i, c in enumerate(_load(APP_PATH))]
    pros = [{**c, "id": f"pro_{c.get('id', i)}", "source": "prospects"} for i, c in enumerate(_load(PROS_PATH))]
    return apps + pros

@router.get("", response_model=List[Candidate])
def list_candidates(
    source: Literal["all","applicants","prospects"] = Query(default="all")
):
    if source == "all":
        return _merge_all()
    if source == "applicants":
        return [{**c, "id": f"app_{c.get('id', i)}", "source": "applicants"}
                for i, c in enumerate(_load(APP_PATH))]
    if source == "prospects":
        return [{**c, "id": f"pro_{c.get('id', i)}", "source": "prospects"}
                for i, c in enumerate(_load(PROS_PATH))]

@router.get("/{candidate_id}", response_model=Candidate)
def get_candidate(candidate_id: str):
    for c in _merge_all():
        if str(c["id"]) == str(candidate_id):
            return c
    raise HTTPException(404, "Candidate not found")


