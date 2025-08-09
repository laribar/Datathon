from fastapi import APIRouter

router = APIRouter(prefix="/candidates", tags=["candidates"])

@router.get("/")
def list_candidates():
    return [{"id": 1, "name": "Maria"}]

