from fastapi import APIRouter, WebSocket

router = APIRouter(prefix="/interview", tags=["interview"])

@router.websocket("/ws")
async def interview_websocket(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text("Conexão WebSocket estabelecida.")
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"Você disse: {data}")
