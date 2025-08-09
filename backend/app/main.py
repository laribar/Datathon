# backend/app/main.py
from fastapi import FastAPI
from  backend.routers import jobs, candidates, match, interview_ws, sessions  # <— ponto na frente

app = FastAPI()

# registre os routers
app.include_router(jobs.router, prefix="/jobs", tags=["jobs"])
app.include_router(candidates.router, prefix="/candidates", tags=["candidates"])
app.include_router(match.router, prefix="/match", tags=["match"])
app.include_router(sessions.router, prefix="/sessions", tags=["sessions"])
# se o interview_ws for WebSocket Router, inclua também se expõe rotas HTTP
# app.include_router(interview_ws.router, prefix="/interview", tags=["interview"])

# backend/routers/jobs.py
from fastapi import APIRouter

router = APIRouter(prefix="/jobs", tags=["jobs"])

# Rota raiz (home)
@app.get("/")
def home():
    return {"status": "API rodando com sucesso 🚀"}
def list_jobs():
    return [{"id": 1, "title": "Data Engineer"}]