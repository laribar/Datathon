from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import openai
import os
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

router = APIRouter()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

@router.post("/chat", response_model=ChatResponse)
async def chat_with_ai(req: ChatRequest):
    """
    Endpoint que recebe a mensagem do usuário e retorna a resposta da IA.
    """
    if not openai.api_key:
        raise HTTPException(status_code=500, detail="Chave da OpenAI não configurada.")

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Você é um entrevistador técnico amigável."},
                {"role": "user", "content": req.message}
            ]
        )
        reply = response["choices"][0]["message"]["content"]
        return {"reply": reply}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar a requisição: {str(e)}")

