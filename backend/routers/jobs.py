# backend/app/routers/jobs.py
import json
from fastapi import APIRouter, HTTPException
from typing import List
from backend.app.models.catalog import Job

router = APIRouter(prefix="/jobs", tags=["jobs"])

DATA_PATH = "backend/data/vagas.json"

def _load_jobs():
    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except Exception as e:
        raise HTTPException(500, f"Erro lendo {DATA_PATH}: {e}")

@router.get("", response_model=List[Job])
def list_jobs():
    return _load_jobs()

@router.get("/{job_id}", response_model=Job)
def get_job(job_id: str):
    for j in _load_jobs():
        if str(j.get("id")) == str(job_id):
            return j
    raise HTTPException(404, "Job not found")

