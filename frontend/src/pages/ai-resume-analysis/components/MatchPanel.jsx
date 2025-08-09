import { useState, useMemo } from "react";
import { matchText, matchPdf } from "../../../utils/api";

export default function MatchPanel({ onResult }) {
  const [vagaText, setVagaText] = useState("");
  const [cvText, setCvText] = useState("");
  const [score, setScore] = useState(null);
  const [isMatch, setIsMatch] = useState(null);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");

  const scorePct = useMemo(() => (score != null ? Math.round(score * 100) : null), [score]);

  async function handleMatchText() {
    setErrorMsg("");
    setLoading(true);
    try {
      const res = await matchText({ vaga_text: vagaText, cv_text: cvText });
      const s = res.score;
      const m = typeof res.match === "boolean" ? res.match : s >= 0.5;
      setScore(s);
      setIsMatch(m);
      onResult?.({
        score: s,
        match: m,
        breakdown: res.breakdown,      // opcional, se o backend enviar
        durationMs: res._durationMs,   // tempo real medido
      });
    } catch (e) {
      setErrorMsg(e?.friendlyMessage || e?.response?.data?.detail || "Falha ao calcular o match (texto).");
    } finally {
      setLoading(false);
    }
  }

  async function handleMatchPdf(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    setErrorMsg("");
    setLoading(true);
    try {
      const res = await matchPdf({ vaga_text: vagaText, file });
      const s = res.score;
      const m = typeof res.match === "boolean" ? res.match : s >= 0.5;
      setScore(s);
      setIsMatch(m);
      onResult?.({
        score: s,
        match: m,
        breakdown: res.breakdown,      // opcional
        durationMs: res._durationMs,   // tempo real
      });
    } catch (e) {
      setErrorMsg(e?.friendlyMessage || e?.response?.data?.detail || "Falha ao calcular o match (PDF).");
    } finally {
      setLoading(false);
      e.target.value = "";
    }
  }

  return (
    <div className="w-full rounded-2xl border border-slate-800 bg-slate-900/50 p-4 md:p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg md:text-xl font-semibold text-white">Testar Match (Texto ou PDF)</h3>
        {scorePct != null && (
          <div className={`px-3 py-1 rounded-full text-sm ${isMatch ? "bg-emerald-600" : "bg-rose-600"} text-white`}>
            Score: {scorePct}%
          </div>
        )}
      </div>

      <div className="grid md:grid-cols-2 gap-4">
        <div className="flex flex-col gap-2">
          <label className="text-slate-300 text-sm">Descrição da Vaga</label>
          <textarea
            className="min-h-[120px] rounded-xl bg-slate-800 text-slate-100 p-3 outline-none border border-slate-700 focus:border-blue-500"
            placeholder="Ex.: Backend Python, FastAPI, Docker, AWS..."
            value={vagaText}
            onChange={(e) => setVagaText(e.target.value)}
          />
        </div>

        <div className="flex flex-col gap-2">
          <label className="text-slate-300 text-sm">Currículo (Texto)</label>
          <textarea
            className="min-h-[120px] rounded-xl bg-slate-800 text-slate-100 p-3 outline-none border border-slate-700 focus:border-blue-500"
            placeholder="Cole o conteúdo do CV aqui…"
            value={cvText}
            onChange={(e) => setCvText(e.target.value)}
          />
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-3 mt-4">
        <button
          onClick={handleMatchText}
          disabled={loading || !vagaText || !cvText}
          className="px-4 py-2 rounded-xl bg-blue-500 disabled:opacity-50 text-white hover:bg-blue-600 transition"
        >
          {loading ? "Calculando…" : "Calcular Match (Texto)"}
        </button>

        <label className="px-4 py-2 rounded-xl bg-slate-800 text-white border border-slate-700 cursor-pointer hover:bg-slate-700 transition">
          Enviar PDF do CV
          <input type="file" accept="application/pdf" className="hidden" onChange={handleMatchPdf} />
        </label>

        {errorMsg && <span className="text-rose-400 text-sm">{errorMsg}</span>}
      </div>

      {scorePct != null && (
        <div className="mt-4 text-slate-300 text-sm">
          {isMatch ? "✅ Match acima do threshold." : "⚠️ Abaixo do threshold."} (threshold padrão: 50%)
        </div>
      )}
    </div>
  );
}

