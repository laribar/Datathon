# === BEGIN FILE: backend/services/orchestrator.py
from __future__ import annotations
import os
from typing import Dict, Any, List, Optional
import openai

# Carrega chave da API OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

class InterviewOrchestrator:
    def __init__(self):
        # Estrutura interna de entrevistas por sessão
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def start_session(self, session_id: str, candidate_info: Optional[Dict[str, Any]] = None):
        """Inicia uma nova sessão de entrevista."""
        self.sessions[session_id] = {
            "candidate_info": candidate_info or {},
            "questions": [
                "Por favor, apresente-se brevemente.",
                "Qual foi seu último projeto e qual foi o maior desafio?",
                "Como você lida com prazos apertados?",
                "Pode citar uma tecnologia que domina e como a aplicou?",
            ],
            "answers": [],
            "current_index": 0
        }
        return self.sessions[session_id]["questions"][0]

    def process_answer(self, session_id: str, answer: str) -> Dict[str, Any]:
        """Armazena resposta e decide próxima pergunta ou encerra."""
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Sessão {session_id} não encontrada.")

        # Armazena a resposta
        session["answers"].append(answer)

        # Analisa resposta (técnico + emocional)
        analysis = self._analyze_answer(answer)

        # Avança para próxima pergunta
        session["current_index"] += 1
        if session["current_index"] < len(session["questions"]):
            next_q = session["questions"][session["current_index"]]
            return {
                "analysis": analysis,
                "next_question": next_q,
                "finished": False
            }
        else:
            return {
                "analysis": analysis,
                "next_question": None,
                "finished": True,
                "summary": self._generate_summary(session["answers"])
            }

    def _analyze_answer(self, answer: str) -> Dict[str, Any]:
        """Análise da resposta usando OpenAI."""
        try:
            prompt = f"""
            Avalie a resposta abaixo considerando:
            1. Clareza
            2. Conhecimento técnico
            3. Comunicação
            4. Emoção percebida

            Resposta: "{answer}"
            """
            resp = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": "Você é um recrutador experiente."},
                          {"role": "user", "content": prompt}],
                temperature=0.3
            )

            feedback = resp.choices[0].message["content"]
            return {"feedback": feedback}

        except Exception as e:
            return {"feedback": f"Erro na análise automática: {e}"}

    def _generate_summary(self, answers: List[str]) -> str:
        """Gera um resumo final da entrevista."""
        try:
            joined_answers = "\n".join([f"{i+1}. {a}" for i, a in enumerate(answers)])
            prompt = f"""
            Você é um recrutador. Gere um resumo final da entrevista com base nas respostas:

            {joined_answers}
            """
            resp = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": "Você é um recrutador especialista."},
                          {"role": "user", "content": prompt}],
                temperature=0.4
            )
            return resp.choices[0].message["content"]

        except Exception as e:
            return f"Erro ao gerar resumo: {e}"


# Singleton para reuso
_orchestrator_singleton: InterviewOrchestrator | None = None

def get_orchestrator() -> InterviewOrchestrator:
    global _orchestrator_singleton
    if _orchestrator_singleton is None:
        _orchestrator_singleton = InterviewOrchestrator()
    return _orchestrator_singleton
# === END FILE
