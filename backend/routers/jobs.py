from fastapi import APIRouter

router = APIRouter(prefix="/jobs", tags=["jobs"])

@router.get("/")
def list_jobs():
    return [{"id": 1, "title": "Data Engineer"}]