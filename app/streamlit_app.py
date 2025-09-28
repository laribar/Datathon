# streamlit_app.py — Streamlit puro (SBERT), deploy-ready p/ GitHub/Cloud
import os, re, json, hashlib, itertools
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ======================== CONFIG ========================
APP_NAME = "SkillAI Match"
APP_VERSION = "1.0.0"

DEFAULT_LIMIAR = float(os.getenv("SCORE_LIMIAR", "0.75"))

# Modelo: usa pasta local OU baixa do Hub se vazio (override por HF_MODEL_NAME)
MODEL_DIR = os.getenv("MODEL_DIR", "models/sbert_encoder")
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
MODEL_NAME = os.getenv("MODEL_NAME", f"local:{MODEL_DIR}")

# Dados: local ou URLs (ex.: raw GitHub)
BASE_CANDIDATOS_PATH = os.getenv("BASE_CANDIDATOS_PATH", "data/applicants_clean.csv")
BASE_VAGAS_PATH = os.getenv("BASE_VAGAS_PATH", "data/vagas_clean.csv")
CANDIDATOS_CSV_URL = os.getenv("CANDIDATOS_CSV_URL")  # opcional (https://raw.githubusercontent.com/...)
VAGAS_CSV_URL = os.getenv("VAGAS_CSV_URL")            # opcional

# Cache de embeddings
EMB_DIR = Path(os.getenv("EMB_DIR", "data/embeddings")); EMB_DIR.mkdir(parents=True, exist_ok=True)
CAND_EMB_PATH = EMB_DIR / "candidatos.npy"
CAND_META_PATH = EMB_DIR / "candidatos.meta.json"
VAGA_EMB_PATH = EMB_DIR / "vagas.npy"
VAGA_META_PATH = EMB_DIR / "vagas.meta.json"

# ======================== TEXT UTILS ========================
_whitespace_re = re.compile(r"\s+")
_SENT_SPLIT_RE = re.compile(r"(?<=[\.!?;:]|\n)\s+")

def clean_text(t: str) -> str:
    if t is None: return ""
    t = t.strip().lower()
    return _whitespace_re.sub(" ", t)

def proportional_score(sim: float, limiar: float) -> float:
    return 100.0 if sim >= limiar else max(0.0, (sim/limiar)*100.0)

def split_sentences(text: str, min_chars: int = 25) -> List[str]:
    if not text: return []
    t = clean_text(text)
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(t) if p and p.strip()]
    parts = [p for p in parts if len(p) >= min_chars]
    return parts if parts else [t]

def top_n_pairs_by_cosine(A: List[str], B: List[str], encoder, top_n: int = 3):
    if not A or not B: return []
    A_vecs = encoder.encode(A, normalize_embeddings=True, show_progress_bar=False)
    B_vecs = encoder.encode(B, normalize_embeddings=True, show_progress_bar=False)
    pairs = []
    for i, j in itertools.product(range(len(A)), range(len(B))):
        sim = float(np.dot(A_vecs[i], B_vecs[j]))
        pairs.append((i, j, sim))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:max(1, top_n)]

# ======================== EMB UTILS ========================
def _hash_dataframe(df: pd.DataFrame) -> str:
    buf = df.to_csv(index=False).encode("utf-8")
    return hashlib.md5(buf).hexdigest()

def _save_embeddings(npy_path: Path, meta_path: Path, embs: np.ndarray, meta: dict) -> None:
    np.save(npy_path, embs); meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

def _load_embeddings(npy_path: Path, meta_path: Path) -> Tuple[np.ndarray, dict]:
    arr = np.load(npy_path); meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return arr, meta

def get_or_build_embeddings(df: pd.DataFrame, text_col: str, npy_path: Path, meta_path: Path, model_dir: str) -> np.ndarray:
    sig = _hash_dataframe(df[[text_col]])
    meta_expected = {"model_dir": model_dir, "text_col": text_col, "signature": sig}
    if npy_path.exists() and meta_path.exists():
        try:
            embs, meta = _load_embeddings(npy_path, meta_path)
            if meta == meta_expected and embs.ndim == 2 and embs.shape[0] == len(df):
                return embs
        except Exception:
            pass
    texts = df[text_col].astype(str).tolist()
    embs = embed_texts(texts, model_dir)
    _save_embeddings(npy_path, meta_path, embs, meta_expected)
    return embs

# ======================== MODEL / ENCODER ========================
@st.cache_resource(show_spinner=False)
def load_model(model_dir: str):
    # import lazy (evita ModuleNotFoundError no boot do cloud)
    from sentence_transformers import SentenceTransformer
    if not os.path.exists(model_dir) or not os.listdir(model_dir):
        model = SentenceTransformer(HF_MODEL_NAME)
        os.makedirs(model_dir, exist_ok=True)
        model.save(model_dir)
        return model
    return SentenceTransformer(model_dir)

@st.cache_data(show_spinner=False)
def embed_texts(texts: List[str], model_dir: str) -> np.ndarray:
    model = load_model(model_dir)
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

@st.cache_data(show_spinner=False)
def embed_text(text: str, model_dir: str) -> np.ndarray:
    model = load_model(model_dir)
    return model.encode(text, normalize_embeddings=True, show_progress_bar=False)

# ======================== DATA LOADERS ========================
@st.cache_data(show_spinner=False)
def _concat_all_columns(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    out = df.copy()
    if target_col in out.columns: out = out.drop(columns=[target_col])
    s = out.fillna("").astype(str)
    out[target_col] = s.apply(lambda row: " ".join([v for v in row.values if v != ""]).strip(), axis=1)
    return out

@st.cache_data(show_spinner=False)
def _read_csv_local_or_url(local_path: str, url_env: str | None) -> pd.DataFrame | None:
    try:
        if os.path.exists(local_path): return pd.read_csv(local_path)
    except Exception: pass
    if url_env:
        try:
            return pd.read_csv(url_env)
        except Exception:
            pass
    return None

@st.cache_data(show_spinner=False)
def load_fixed_bases() -> Tuple[pd.DataFrame, pd.DataFrame, list]:
    logs = []
    cand = _read_csv_local_or_url(BASE_CANDIDATOS_PATH, CANDIDATOS_CSV_URL)
    vaga = _read_csv_local_or_url(BASE_VAGAS_PATH, VAGAS_CSV_URL)

    if cand is None:
        logs.append("⚠️ Não encontrei candidatos locais/URL. Usando amostra.")
        cand = pd.DataFrame({
            "nome": ["Ana"], "skills": ["python airflow spark"], "exp": ["3 anos dados"], "cidade": ["SP"]
        })
    if vaga is None:
        logs.append("⚠️ Não encontrei vagas locais/URL. Usando amostra.")
        vaga = pd.DataFrame({
            "titulo": ["Engenheira de Dados"], "requisitos": ["python spark airflow"], "descricao": ["time de dados cloud"]
        })

    cand = _concat_all_columns(cand, "cv_text")
    vaga = _concat_all_columns(vaga, "vaga_text")
    return cand, vaga, logs

# ======================== UI / APP ========================
st.set_page_config(page_title=APP_NAME, layout="wide")

with st.sidebar:
    st.markdown(f"### {APP_NAME}")
    st.caption(f"Versão {APP_VERSION}")
    st.write("**Encoder:**", MODEL_NAME)
    limiar = st.slider("Limiar de aprovação (cosine)", 0.50, 0.95, DEFAULT_LIMIAR, 0.01)

st.title("🔎 Match CV × Vaga (SBERT, sem FastAPI)")

candidatos_df, vagas_df, _logs = load_fixed_bases()
if _logs:
    for m in _logs: st.warning(m)

# guardar sessão
st.session_state["candidatos_df"] = candidatos_df.copy()
st.session_state["vagas_df"] = vagas_df.copy()

# Sidebar: pré-cálculo de embeddings em disco
with st.sidebar:
    st.divider()
    st.markdown("#### Cache de embeddings")
    if st.button("⚡ Pré-calcular e salvar cache"):
        with st.spinner("Gerando embeddings (candidatos e vagas)…"):
            _ = get_or_build_embeddings(st.session_state["candidatos_df"], "cv_text", CAND_EMB_PATH, CAND_META_PATH, MODEL_DIR)
            _ = get_or_build_embeddings(st.session_state["vagas_df"], "vaga_text", VAGA_EMB_PATH, VAGA_META_PATH, MODEL_DIR)
            st.success(f"Cache salvo em {EMB_DIR}")

# Tentar carregar do cache
try:
    st.session_state["cand_embs"], _ = _load_embeddings(CAND_EMB_PATH, CAND_META_PATH)
except Exception:
    st.session_state["cand_embs"] = None
try:
    st.session_state["vaga_embs"], _ = _load_embeddings(VAGA_EMB_PATH, VAGA_META_PATH)
except Exception:
    st.session_state["vaga_embs"] = None

# -------------------- TABS --------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1×1 (Texto)", "Batch (2 uploads)", "Batch (CSV pareado)",
    "Explain (trechos)", "Ranking por vaga", "Bases atuais",
])

with tab1:
    st.subheader("Match 1×1 — cole textos")
    colA, colB = st.columns(2)
    with colA: cv_text = st.text_area("CV — texto puro", height=220)
    with colB: vaga_text = st.text_area("Vaga — texto puro", height=220)
    clean = st.checkbox("Aplicar limpeza (lower + espaços)", True)
    if st.button("Calcular Similaridade"):
        if not cv_text or not vaga_text:
            st.warning("Informe CV e Vaga.")
        else:
            cv_raw = clean_text(cv_text) if clean else cv_text
            vaga_raw = clean_text(vaga_text) if clean else vaga_text
            cv_vec = embed_text(cv_raw, MODEL_DIR)
            vaga_vec = embed_text(vaga_raw, MODEL_DIR)
            sim = float(np.dot(cv_vec, vaga_vec))
            st.metric("Similaridade (cosine)", f"{sim:.4f}")
            st.metric("Score (%)", f"{proportional_score(sim, limiar):.2f}")
            st.write("**Aprovado?**", "✅ Sim" if sim >= limiar else "❌ Não")

with tab2:
    st.subheader("Batch — subir candidatos e vagas separadamente")
    st.caption("CSV de candidatos: qualquer esquema; CSV de vagas: idem. Concatenamos todas as colunas.")
    up_cand = st.file_uploader("CSV candidatos", type=["csv"], key="u1")
    up_vaga = st.file_uploader("CSV vagas", type=["csv"], key="u2")
    top_k = st.number_input("Top K por candidato", 1, 50, 5)
    if st.button("Processar Batch (2 uploads)"):
        if up_cand is not None: candidatos_df = pd.read_csv(up_cand)
        if up_vaga is not None: vagas_df = pd.read_csv(up_vaga)
        candidatos_df = _concat_all_columns(candidatos_df, "cv_text")
        vagas_df = _concat_all_columns(vagas_df, "vaga_text")
        st.session_state["candidatos_df"] = candidatos_df.copy()
        st.session_state["vagas_df"] = vagas_df.copy()
        emb_cvs = embed_texts(candidatos_df["cv_text"].astype(str).tolist(), MODEL_DIR)
        emb_vgs = embed_texts(vagas_df["vaga_text"].astype(str).tolist(), MODEL_DIR)
        sim_matrix = emb_cvs @ emb_vgs.T
        rows = []
        for i in range(len(candidatos_df)):
            sims = sim_matrix[i]; top_idx = np.argsort(-sims)[:top_k]
            for rank, j in enumerate(top_idx, start=1):
                rows.append({"cv_index": i, "vaga_index": int(j), "rank": rank,
                             "similaridade": float(sims[j]),
                             "score": round(proportional_score(float(sims[j]), limiar), 2),
                             "aprovado": bool(sims[j] >= limiar)})
        out_df = pd.DataFrame(rows)
        st.dataframe(out_df, use_container_width=True, hide_index=True)
        st.download_button("Baixar resultados (CSV)", out_df.to_csv(index=False).encode("utf-8"),
                           file_name="matches_topk.csv", mime="text/csv")

with tab3:
    st.subheader("Batch — CSV pareado (cv_text, vaga_text)")
    up = st.file_uploader("CSV pareado", type=["csv"], key="u3")
    if up is not None:
        df = pd.read_csv(up)
        if not {"cv_text", "vaga_text"}.issubset(df.columns):
            df = _concat_all_columns(df, "cv_text"); df = _concat_all_columns(df, "vaga_text")
        emb_cvs = embed_texts(df["cv_text"].astype(str).tolist(), MODEL_DIR)
        emb_vgs = embed_texts(df["vaga_text"].astype(str).tolist(), MODEL_DIR)
        sims = np.sum(emb_cvs * emb_vgs, axis=1)
        df_out = df.copy(); df_out["similaridade"] = sims
        df_out["score"] = [proportional_score(x, limiar) for x in sims]
        df_out["aprovado"] = df_out["similaridade"] >= limiar
        st.dataframe(df_out, use_container_width=True, hide_index=True)
        st.download_button("Baixar (CSV)", df_out.to_csv(index=False).encode("utf-8"),
                           file_name="matches_pareado.csv", mime="text/csv")

with tab4:
    st.subheader("Explain — Top trechos CV × Vaga")
    col1, col2 = st.columns(2)
    with col1: cv_t = st.text_area("CV", height=200)
    with col2: vg_t = st.text_area("Vaga", height=200)
    top_n = st.number_input("Top N pares", 1, 10, 3)
    min_chars = st.number_input("Mín. caracteres/sentença", 10, 500, 25)
    if st.button("Gerar Explain"):
        model = load_model(MODEL_DIR)
        pairs = top_n_pairs_by_cosine(split_sentences(cv_t, min_chars), split_sentences(vg_t, min_chars), model, int(top_n))
        rows = [{"rank": r+1, "similaridade": round(float(sim), 6),
                 "cv_index": i, "vaga_index": j,
                 "cv_snippet": split_sentences(cv_t, min_chars)[i],
                 "vaga_snippet": split_sentences(vg_t, min_chars)[j]} for r,(i,j,sim) in enumerate(pairs)]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with tab5:
    st.subheader("Ranking por vaga — escolha a vaga e veja os top N candidatos")
    vdf = st.session_state["vagas_df"].copy()
    cdf = st.session_state["candidatos_df"].copy()
    def _vaga_label(row: pd.Series) -> str:
        title = row.get("title") or row.get("vaga_titulo") or ""
        vid = row.get("vaga_id") or row.get("id") or ""
        if isinstance(title, str) and title.strip(): base_txt = title.strip()
        else:
            vt = str(row.get("vaga_text", "")); base_txt = vt[:80] + ("…" if len(vt) > 80 else "")
        return (f"[{vid}] " if isinstance(vid, str) and vid else "") + base_txt
    options = vdf.apply(_vaga_label, axis=1).tolist()
    idx = st.selectbox("Selecione a vaga", options=range(len(options)), format_func=lambda i: options[i])
    col_l, col_r = st.columns([1,1])
    with col_l: top_n = st.number_input("Top N candidatos", 1, 100, 10)
    with col_r: clean_rank = st.checkbox("Aplicar limpeza nos textos", True)
    with st.expander("Ver descrição completa da vaga"):
        vaga_text_sel = str(vdf.iloc[idx]["vaga_text"]); st.write(vaga_text_sel)

    if st.button("🔍 Gerar ranking"):
        with st.spinner("Preparando embeddings…"):
            if st.session_state.get("cand_embs") is None:
                cvs = cdf["cv_text"].astype(str).tolist()
                if clean_rank: cvs = [clean_text(x) for x in cvs]
                st.session_state["cand_embs"] = embed_texts(cvs, MODEL_DIR)
            emb_cvs = st.session_state["cand_embs"]
            if st.session_state.get("vaga_embs") is not None and len(st.session_state["vaga_embs"]) == len(vdf):
                emb_vaga = st.session_state["vaga_embs"][idx]
                if clean_rank: emb_vaga = embed_text(clean_text(vaga_text_sel), MODEL_DIR)
            else:
                emb_vaga = embed_text(clean_text(vaga_text_sel) if clean_rank else vaga_text_sel, MODEL_DIR)

        sims = emb_cvs @ emb_vaga
        order = np.argsort(-sims)[:int(top_n)]
        rows = []
        for rank, i in enumerate(order, start=1):
            sim = float(sims[i]); score = proportional_score(sim, limiar)
            row = {"rank": rank, "cand_index": int(i), "similaridade": round(sim, 6),
                   "score": round(score, 2), "aprovado": bool(sim >= limiar)}
            for col in ["candidate_id", "id", "nome", "name", "email", "telefone", "phone"]:
                if col in cdf.columns: row[col] = cdf.iloc[i][col]
            rows.append(row)
        res = pd.DataFrame(rows)
        st.dataframe(res, use_container_width=True, hide_index=True)
        st.download_button("Baixar ranking (CSV)", res.to_csv(index=False).encode("utf-8"),
                           file_name="ranking_por_vaga.csv", mime="text/csv")

with tab6:
    st.subheader("Bases atuais em memória")
    st.write("**Candidatos (cv_text)**"); st.dataframe(st.session_state["candidatos_df"], use_container_width=True, hide_index=True)
    st.write("**Vagas (vaga_text)**");    st.dataframe(st.session_state["vagas_df"], use_container_width=True, hide_index=True)

st.caption(f"Modelo: {MODEL_NAME} • Limiar: {limiar:.2f} • App: {APP_NAME} v{APP_VERSION} — Streamlit")
