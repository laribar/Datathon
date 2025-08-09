from fastapi import APIRouter

router = APIRouter(prefix="/sessions", tags=["sessions"])

@router.get("/")
def list_sessions():
    return [{"id": 1, "status": "active"}]
