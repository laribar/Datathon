from fastapi import FastAPI
from backend.routers import jobs, candidates, match, interview_ws, sessions

app = FastAPI()

@app.get("/")
def home():
    return {"status": "API rodando com sucesso 🚀"}

# Routers
app.include_router(jobs.router, prefix="/jobs", tags=["jobs"])
app.include_router(candidates.router, prefix="/candidates", tags=["candidates"])
app.include_router(match.router)  # <- sem prefix/tags aqui, pois já estão no router
app.include_router(sessions.router, prefix="/sessions", tags=["sessions"])
# app.include_router(interview_ws.router, prefix="/interview", tags=["interview"])

