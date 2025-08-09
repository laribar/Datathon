import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_BASE_URL?.trim() || "http://127.0.0.1:8000";

export const API = axios.create({
  baseURL: BASE_URL,
  timeout: 20000,
});

API.interceptors.response.use(
  (res) => res,
  (err) => {
    const detail = err?.response?.data?.detail || err?.message || "Erro inesperado ao chamar a API.";
    return Promise.reject({ ...err, friendlyMessage: detail });
  }
);

// /match (texto puro) — retorna também duração em ms
export async function matchText({ vaga_text, cv_text }) {
  const t0 = performance.now();
  const { data } = await API.post("/match", { vaga_text, cv_text });
  const t1 = performance.now();
  return { ...data, _durationMs: t1 - t0 };
}

// /match/pdf (upload) — idem
export async function matchPdf({ vaga_text, file }) {
  const form = new FormData();
  form.append("vaga_text", vaga_text);
  form.append("cv_pdf", file);

  const t0 = performance.now();
  const { data } = await API.post("/match/pdf", form, {
    headers: { "Content-Type": "multipart/form-data" },
    timeout: 60000,
  });
  const t1 = performance.now();
  return { ...data, _durationMs: t1 - t0 };
}
// frontend/src/utils/api.js
export const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

export async function fetchJobs() {
  const r = await fetch(`${API_URL}/jobs`);
  if (!r.ok) throw new Error("Falha ao carregar vagas");
  return r.json();
}

export async function fetchCandidates(source = "all") {
  const r = await fetch(`${API_URL}/candidates?source=${source}`);
  if (!r.ok) throw new Error("Falha ao carregar candidatos");
  return r.json();
}

export async function postMatch(jobId, candidateId) {
  const r = await fetch(`${API_URL}/match`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ job_id: jobId, candidate_id: candidateId })
  });
  if (!r.ok) throw new Error("Falha ao calcular match");
  return r.json();
}
