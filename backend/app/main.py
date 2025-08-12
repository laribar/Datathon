from fastapi import FastAPI
from backend.routers import jobs, candidates, match, interview_ws, sessions, chat
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
import openai

app = FastAPI()

# Carrega variáveis do .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    print("⚠️ OPENAI_API_KEY não encontrada. Verifique o .env no Render.")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # durante MVP, depois restrinja para o Netlify
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "API rodando com sucesso 🚀"}

# Routers
app.include_router(jobs.router, prefix="/jobs", tags=["jobs"])
app.include_router(candidates.router, prefix="/candidates", tags=["candidates"])
app.include_router(match.router)  # já tem prefix/tags dentro do router
app.include_router(sessions.router, prefix="/sessions", tags=["sessions"])
# app.include_router(interview_ws.router, prefix="/interview", tags=["interview"])
app.include_router(chat.router, prefix="/api", tags=["chat"])
