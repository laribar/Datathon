# === BEGIN FILE: backend/routers/interview_ws.py
from __future__ import annotations
import json
import base64
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Any

from backend.services import orchestrator, stt, tts  # já preparado para integração futura

router = APIRouter(prefix="/interview", tags=["interview"])

# Estrutura de clientes conectados
active_connections: Dict[str, WebSocket] = {}

async def send_json(websocket: WebSocket, data: Dict[str, Any]):
    """Envia dados como JSON para o cliente."""
    await websocket.send_text(json.dumps(data, ensure_ascii=False))

@router.websocket("/ws/{session_id}")
async def interview_websocket(websocket: WebSocket, session_id: str):
    """
    WebSocket para gerenciar a entrevista em tempo real.
    session_id identifica a sessão da entrevista.
    """
    await websocket.accept()
    active_connections[session_id] = websocket

    await send_json(websocket, {
        "type": "status",
        "payload": f"Conexão WebSocket estabelecida para a sessão {session_id}."
    })

    try:
        while True:
            raw_data = await websocket.receive_text()

            # Garantir que a mensagem recebida é JSON
            try:
                msg = json.loads(raw_data)
                msg_type = msg.get("type")
                payload = msg.get("payload")
            except json.JSONDecodeError:
                await send_json(websocket, {
                    "type": "error",
                    "payload": "Mensagem inválida. Use JSON."
                })
                continue

            # --- Tratamento por tipo de mensagem ---
            if msg_type == "text_response":
                # Resposta em texto do candidato
                await handle_text_response(session_id, payload, websocket)

            elif msg_type == "audio_response":
                # Resposta de áudio em Base64 → Transcrição
                await handle_audio_response(session_id, payload, websocket)

            elif msg_type == "start_interview":
                # Início da entrevista
                await start_interview(session_id, websocket)

            elif msg_type == "end_interview":
                # Encerrar entrevista
                await end_interview(session_id, websocket)
                break

            else:
                await send_json(websocket, {
                    "type": "error",
                    "payload": f"Tipo de mensagem não suportado: {msg_type}"
                })

    except WebSocketDisconnect:
        print(f"🔌 Sessão {session_id} desconectada.")
    finally:
        active_connections.pop(session_id, None)


# ------------------------
# Handlers de cada evento
# ------------------------
async def start_interview(session_id: str, websocket: WebSocket):
    """Inicia a entrevista e envia a primeira pergunta."""
    # Aqui futuramente pode chamar orchestrator.get_first_question()
    question = "Bem-vindo à entrevista! Pode se apresentar?"
    await send_json(websocket, {"type": "question", "payload": question})


async def handle_text_response(session_id: str, text: str, websocket: WebSocket):
    """Processa uma resposta em texto."""
    print(f"📝 [{session_id}] Resposta recebida (texto): {text}")

    # Analisar tecnicamente e emocionalmente (integração futura com orchestrator)
    # next_question = orchestrator.get_next_question(text)
    from backend.services.orchestrator import get_orchestrator
    orch = get_orchestrator()
    result = orch.process_answer(session_id, text)
    await send_json(websocket, {"type": "analysis", "payload": result["analysis"]})
    if not result["finished"]:
        await send_json(websocket, {"type": "question", "payload": result["next_question"]})
    else:
        await send_json(websocket, {"type": "summary", "payload": result["summary"]})


        await send_json(websocket, {"type": "analysis", "payload": "Resposta registrada e analisada."})
        await send_json(websocket, {"type": "question", "payload": next_question})


async def handle_audio_response(session_id: str, audio_b64: str, websocket: WebSocket):
    """Processa uma resposta de áudio (Base64) → Texto."""
    try:
        audio_bytes = base64.b64decode(audio_b64)
        # Transcrição via STT
        transcript = stt.transcribe_audio_bytes(audio_bytes)  # Função no stt.py
        print(f"🎤 [{session_id}] Transcrição: {transcript}")

        await handle_text_response(session_id, transcript, websocket)
    except Exception as e:
        await send_json(websocket, {
            "type": "error",
            "payload": f"Falha ao processar áudio: {e}"
        })


async def end_interview(session_id: str, websocket: WebSocket):
    """Encerra a entrevista."""
    await send_json(websocket, {"type": "status", "payload": "Entrevista encerrada."})
    await websocket.close()
# === END FILE
