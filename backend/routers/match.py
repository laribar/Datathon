from fastapi import APIRouter

router = APIRouter(prefix="/match", tags=["match"])

@router.get("/")
def get_match():
    return {"match_score": 92}

