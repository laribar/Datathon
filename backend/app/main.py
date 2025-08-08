# backend/main.py
from fastapi import FastAPI, WebSocket
from uuid import uuid4

app = FastAPI()

sessions = {}

@app.post("/start_session")
def start_session():
    session_id = str(uuid4())
    sessions[session_id] = {"status": "active", "history": []}
    return {"session_id": session_id, "link": f"https://seuapp.netlify.app/{session_id}"}

@app.websocket("/ws/{session_id}")
async def interview_ws(websocket: WebSocket, session_id: str):
    await websocket.accept()
    await websocket.send_text("Olá! Vamos começar a entrevista.")
    while True:
        data = await websocket.receive_text()
        # Aqui entraria STT, análise e próxima pergunta
        await websocket.send_text(f"Você respondeu: {data}")
