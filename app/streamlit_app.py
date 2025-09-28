# streamlit_app.py — App Streamlit 100% sem FastAPI (match CV × Vaga com SBERT)
# ---------------------------------------------------------------
# Requisitos (requirements.txt)
# streamlit
# sentence-transformers
# numpy
# pandas
# scikit-learn  # (opcional; não usamos diretamente aqui)
# ---------------------------------------------------------------

import os
import io
import re
import itertools
from typing import List, Tuple
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

# ======================== CONFIG ========================
EMB_DIR = Path(os.getenv("EMB_DIR", "data/embeddings"))
EMB_DIR.mkdir(parents=True, exist_ok=True)
APP_NAME = "SkillAI Match"
APP_VERSION = "1.0.0"
DEFAULT_LIMIAR = float(os.getenv("SCORE_LIMIAR", "0.75"))
MODEL_DIR = os.getenv("MODEL_DIR", os.path.join("models", "sbert_encoder"))
MODEL_NAME = os.getenv("MODEL_NAME", f"local:{MODEL_DIR}")
CAND_EMB_PATH = EMB_DIR / "candidatos.npy"
CAND_META_PATH = EMB_DIR / "candidatos.meta.json"
VAGA_EMB_PATH = EMB_DIR / "vagas.npy"
VAGA_META_PATH = EMB_DIR / "vagas.meta.json"

# Caminhos para bases iniciais (se existirem, o app já sobe carregando-as)
BASE_CANDIDATOS_PATH = os.getenv("BASE_CANDIDATOS_PATH", "data/applicants_clean.csv")
BASE_VAGAS_PATH = os.getenv("BASE_VAGAS_PATH", "data/vagas_clean.csv")

# ======================== UTILS ========================
_whitespace_re = re.compile(r"\s+")
_SENT_SPLIT_RE = re.compile(r"(?<=[\.!?;:]|\n)\s+")


def clean_text(text: str) -> str:
    if text is None:
        return ""
    text = text.strip().lower()
    text = _whitespace_re.sub(" ", text)
    return text


def proportional_score(similarity: float, limiar: float) -> float:
    if similarity >= limiar:
        return 100.0
    return max(0.0, (similarity / limiar) * 100.0)


def split_sentences(text: str, min_chars: int = 25) -> List[str]:
    if not text:
        return []
    t = clean_text(text)
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(t) if p and p.strip()]
    parts = [p for p in parts if len(p) >= min_chars]
    return parts if parts else [t]


def top_n_pairs_by_cosine(A: List[str], B: List[str], encoder: "SentenceTransformer", top_n: int = 3) -> List[Tuple[int, int, float]]:
    if not A or not B:
        return []
    A_vecs = encoder.encode(A, normalize_embeddings=True, show_progress_bar=False)
    B_vecs = encoder.encode(B, normalize_embeddings=True, show_progress_bar=False)
    pairs = []
    for i, j in itertools.product(range(len(A)), range(len(B))):
        sim = float(np.dot(A_vecs[i], B_vecs[j]))  # dot == cosine por normalização
        pairs.append((i, j, sim))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[: max(1, top_n)]

# ---- Helpers para cache de embeddings em disco ----

def _hash_dataframe(df: pd.DataFrame) -> str:
    buf = df.to_csv(index=False).encode("utf-8")
    return hashlib.md5(buf).hexdigest()


def _save_embeddings(npy_path: Path, meta_path: Path, embs: np.ndarray, meta: dict) -> None:
    np.save(npy_path, embs)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_embeddings(npy_path: Path, meta_path: Path) -> Tuple[np.ndarray, dict]:
    arr = np.load(npy_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return arr, meta


def get_or_build_embeddings(df: pd.DataFrame, text_col: str, npy_path: Path, meta_path: Path, model_dir: str) -> np.ndarray:
    sig = _hash_dataframe(df[[text_col]])
    meta_expected = {"model_dir": model_dir, "text_col": text_col, "signature": sig}

    if npy_path.exists() and meta_path.exists():
        try:
            embs, meta = _load_embeddings(npy_path, meta_path)
            if meta == meta_expected and embs.shape[0] == len(df) and embs.ndim == 2:
                return embs
        except Exception:
            pass

    texts = df[text_col].astype(str).tolist()
    embs = embed_texts(texts, model_dir)
    _save_embeddings(npy_path, meta_path, embs, meta_expected)
    return embs


# ======================== CACHES ========================
@st.cache_resource(show_spinner=False)
def load_model(model_dir: str) -> SentenceTransformer:
    return SentenceTransformer(model_dir)


@st.cache_data(show_spinner=False)
def embed_texts(texts: List[str], model_dir: str) -> np.ndarray:
    model = load_model(model_dir)
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


@st.cache_data(show_spinner=False)
def embed_text(text: str, model_dir: str) -> np.ndarray:
    model = load_model(model_dir)
    return model.encode(text, normalize_embeddings=True, show_progress_bar=False)


# ======================== LAYOUT ========================
st.set_page_config(page_title=f"{APP_NAME}", layout="wide")

with st.sidebar:
    st.markdown(f"### {APP_NAME}")
    st.caption(f"Versão {APP_VERSION}")
    st.write("**Encoder:**", MODEL_NAME)
    limiar = st.slider("Limiar de aprovação (cosine)", 0.50, 0.95, DEFAULT_LIMIAR, 0.01)
    st.divider()
    st.markdown("#### Bases iniciais (auto-carregadas se existirem)")
    st.code(f"Candidatos: {BASE_CANDIDATOS_PATH}\nVagas: {BASE_VAGAS_PATH}")

st.title("🔎 Match CV × Vaga (SBERT, sem FastAPI)")

# ============ Carrega bases FIXAS e gera cv_text/vaga_text a partir de TODAS as colunas ============
@st.cache_data(show_spinner=False)
def _concat_all_columns(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    # Preserva o DF original e cria a coluna alvo concatenando **todas** as colunas como texto
    out = df.copy()
    if target_col in out.columns:
        out = out.drop(columns=[target_col])  # vamos recomputar
    # Converte NaN -> "" e tudo para string, mantendo ordem das colunas
    s = out.fillna("").astype(str)
    out[target_col] = s.apply(lambda row: " ".join([v for v in row.values if v != ""]).strip(), axis=1)
    return out

@st.cache_data(show_spinner=False)
def load_fixed_bases() -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Lê os CSVs obrigatórios e sempre constrói cv_text e vaga_text concatenando todas as colunas
    cand_raw = pd.read_csv(BASE_CANDIDATOS_PATH)
    vagas_raw = pd.read_csv(BASE_VAGAS_PATH)
    cand_df = _concat_all_columns(cand_raw, "cv_text")
    vagas_df = _concat_all_columns(vagas_raw, "vaga_text")
    return cand_df, vagas_df

candidatos_df, vagas_df = load_fixed_bases()

# Pré-calcula/recupera embeddings e guarda na sessão
st.session_state["candidatos_df"] = candidatos_df.copy()
st.session_state["vagas_df"] = vagas_df.copy()

with st.sidebar:
    if st.button("⚡ Pré-calcular embeddings e salvar cache"):
        with st.spinner("Gerando embeddings (candidatos e vagas)…"):
            cand_embs = get_or_build_embeddings(st.session_state["candidatos_df"], "cv_text", CAND_EMB_PATH, CAND_META_PATH, MODEL_DIR)
            vaga_embs = get_or_build_embeddings(st.session_state["vagas_df"], "vaga_text", VAGA_EMB_PATH, VAGA_META_PATH, MODEL_DIR)
            st.success(f"Cache salvo em {EMB_DIR}")

# Carrega do cache (se existir); caso contrário, calcula on-demand nas abas
try:
    st.session_state["cand_embs"], _ = _load_embeddings(CAND_EMB_PATH, CAND_META_PATH)
except Exception:
    st.session_state["cand_embs"] = None
try:
    st.session_state["vaga_embs"], _ = _load_embeddings(VAGA_EMB_PATH, VAGA_META_PATH)
except Exception:
    st.session_state["vaga_embs"] = None

# Sessão: mantém as bases atuais (podem ser trocadas por upload)
if "candidatos_df" not in st.session_state:
    st.session_state["candidatos_df"] = candidatos_df.copy()
if "vagas_df" not in st.session_state:
    st.session_state["vagas_df"] = vagas_df.copy()


# ======================== TABS ========================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1×1 (Texto)",
    "Batch (2 uploads)",
    "Batch (CSV pareado)",
    "Explain (trechos)",
    "Ranking por vaga",
    "Bases atuais",
])

# -------- TAB 1: Match 1×1 --------
with tab1:
    st.subheader("Match 1×1 — cole textos (sem FastAPI)")
    colA, colB = st.columns(2)
    with colA:
        cv_text = st.text_area("CV — texto puro", height=220, placeholder="Cole o CV aqui…")
    with colB:
        vaga_text = st.text_area("Vaga — texto puro", height=220, placeholder="Cole a descrição da vaga aqui…")

    clean = st.checkbox("Aplicar limpeza básica (lower + normalização de espaços)", value=True)
    if st.button("Calcular Similaridade"):
        if not cv_text or not vaga_text:
            st.warning("Informe CV e Vaga.")
        else:
            cv_raw = clean_text(cv_text) if clean else cv_text
            vaga_raw = clean_text(vaga_text) if clean else vaga_text
            cv_vec = embed_text(cv_raw, MODEL_DIR)
            vaga_vec = embed_text(vaga_raw, MODEL_DIR)
            sim = float(np.dot(cv_vec, vaga_vec))  # embeddings normalizados
            score = proportional_score(sim, limiar)
            passed = sim >= limiar

            st.metric("Similaridade (cosine)", f"{sim:.4f}")
            st.metric("Score (%)", f"{score:.2f}")
            st.write("**Aprovado pelo limiar?**", "✅ Sim" if passed else "❌ Não")

# -------- TAB 2: Batch (2 uploads) --------
with tab2:
    st.subheader("Batch — subir **candidatos** e **vagas** separadamente")
    st.caption("Arquivos CSV com colunas: **cv_text** (candidatos) e **vaga_text** (vagas). ")

    up_cand = st.file_uploader("CSV de candidatos (cv_text)", type=["csv"], key="u1")
    up_vaga = st.file_uploader("CSV de vagas (vaga_text)", type=["csv"], key="u2")
    top_k = st.number_input("Top K por candidato", 1, 50, 5)
    clean_batch = st.checkbox("Aplicar limpeza básica nos textos", True)

    if st.button("Processar Batch (2 uploads)"):
        if up_cand is not None:
            candidatos_df = pd.read_csv(up_cand)
        if up_vaga is not None:
            vagas_df = pd.read_csv(up_vaga)

        # Persistir na sessão
        st.session_state["candidatos_df"] = candidatos_df.copy()
        st.session_state["vagas_df"] = vagas_df.copy()

        if "cv_text" not in candidatos_df.columns or "vaga_text" not in vagas_df.columns:
            st.error("Colunas obrigatórias ausentes. Esperado: cv_text em candidatos, vaga_text em vagas.")
        else:
            cvs = [clean_text(x) if clean_batch else x for x in candidatos_df["cv_text"].astype(str).tolist()]
            vagas = [clean_text(x) if clean_batch else x for x in vagas_df["vaga_text"].astype(str).tolist()]

            # Embeddings
            emb_cvs = embed_texts(cvs, MODEL_DIR)
            emb_vagas = embed_texts(vagas, MODEL_DIR)

            # Matriz de similaridade via produto interno (cosine)
            sim_matrix = np.matmul(emb_cvs, emb_vagas.T)

            # Para cada CV, pegar Top K vagas
            rows = []
            for i, cv_row in enumerate(candidatos_df.itertuples(index=False)):
                sims = sim_matrix[i]
                top_idx = np.argsort(-sims)[:top_k]
                for rank, j in enumerate(top_idx, start=1):
                    sim = float(sims[j])
                    rows.append(
                        {
                            "cv_index": i,
                            "vaga_index": int(j),
                            "rank": rank,
                            "similaridade": round(sim, 6),
                            "score": round(proportional_score(sim, limiar), 2),
                            "aprovado": bool(sim >= limiar),
                        }
                    )
            out_df = pd.DataFrame(rows)
            st.dataframe(out_df, use_container_width=True, hide_index=True)

            # Download
            csv = out_df.to_csv(index=False).encode("utf-8")
            st.download_button("Baixar resultados (CSV)", data=csv, file_name="matches_topk.csv", mime="text/csv")

# -------- TAB 3: Batch (CSV pareado) --------
with tab3:
    st.subheader("Batch — um único CSV **pareado** (cv_text, vaga_text)")
    st.caption("Envie um CSV contendo **cv_text** e **vaga_text** por linha.")

    sample = pd.DataFrame({
        "cv_text": ["Engenheiro de dados com Spark e Airflow."],
        "vaga_text": ["Procuramos Engenheiro de Dados com Airflow e Spark."],
    })
    st.download_button(
        "Baixar template CSV",
        sample.to_csv(index=False).encode("utf-8"),
        file_name="template_batch.csv",
        mime="text/csv",
    )

    up_batch = st.file_uploader("Carregar CSV pareado", type=["csv"], key="u3")
    clean_paired = st.checkbox("Aplicar limpeza básica", True)

    if up_batch is not None:
        batch_df = pd.read_csv(up_batch)
        if not {"cv_text", "vaga_text"}.issubset(batch_df.columns):
            st.error("CSV deve conter colunas cv_text e vaga_text.")
        else:
            cvs = [clean_text(x) if clean_paired else x for x in batch_df["cv_text"].astype(str).tolist()]
            vagas = [clean_text(x) if clean_paired else x for x in batch_df["vaga_text"].astype(str).tolist()]

            emb_cvs = embed_texts(cvs, MODEL_DIR)
            emb_vagas = embed_texts(vagas, MODEL_DIR)
            sims = np.sum(emb_cvs * emb_vagas, axis=1)  # dot por linha

            batch_df_out = batch_df.copy()
            batch_df_out["similaridade"] = sims
            batch_df_out["score"] = [proportional_score(x, limiar) for x in sims]
            batch_df_out["aprovado"] = batch_df_out["similaridade"] >= limiar

            st.dataframe(batch_df_out, use_container_width=True, hide_index=True)
            st.download_button(
                "Baixar resultados (CSV)",
                data=batch_df_out.to_csv(index=False).encode("utf-8"),
                file_name="matches_pareado.csv",
                mime="text/csv",
            )

# -------- TAB 4: Explain --------
with tab4:
    st.subheader("Explain — Top trechos CV × Vaga")
    col1, col2 = st.columns(2)
    with col1:
        cv_t = st.text_area("CV — texto puro", height=220)
    with col2:
        vaga_t = st.text_area("Vaga — texto puro", height=220)

    top_n = st.number_input("Top N pares de trechos", 1, 10, 3)
    min_chars = st.number_input("Mínimo de caracteres por sentença", 10, 500, 25)
    clean_e = st.checkbox("Limpar textos", True)

    if st.button("Gerar Explain"):
        if not cv_t or not vaga_t:
            st.warning("Informe CV e Vaga.")
        else:
            cv_raw = clean_text(cv_t) if clean_e else cv_t
            vaga_raw = clean_text(vaga_t) if clean_e else vaga_t

            model = load_model(MODEL_DIR)
            pairs = top_n_pairs_by_cosine(
                split_sentences(cv_raw, min_chars=min_chars),
                split_sentences(vaga_raw, min_chars=min_chars),
                model,
                top_n=int(top_n),
            )
            if not pairs:
                st.info("Sem pares para explicar.")
            else:
                cv_sents = split_sentences(cv_raw, min_chars=min_chars)
                vaga_sents = split_sentences(vaga_raw, min_chars=min_chars)
                rows = []
                for rank, (i, j, sim) in enumerate(pairs, start=1):
                    rows.append(
                        {
                            "rank": rank,
                            "similaridade": round(float(sim), 6),
                            "cv_index": i,
                            "vaga_index": j,
                            "cv_snippet": cv_sents[i],
                            "vaga_snippet": vaga_sents[j],
                        }
                    )
                exp_df = pd.DataFrame(rows)
                st.dataframe(exp_df, use_container_width=True, hide_index=True)

# -------- TAB 5: Ranking por vaga --------
with tab5:
    st.subheader("Ranking por vaga — escolha a vaga e veja os top N candidatos")
    vdf = st.session_state["vagas_df"].copy()
    cdf = st.session_state["candidatos_df"].copy()

    if vdf.empty or cdf.empty or "vaga_text" not in vdf.columns or "cv_text" not in cdf.columns:
        st.info("Carregue as bases de **vagas (vaga_text)** e **candidatos (cv_text)** nas abas de Batch.")
    else:
        def _vaga_label(row: pd.Series) -> str:
            title = row.get("title") or row.get("vaga_titulo") or ""
            vid = row.get("vaga_id") or row.get("id") or ""
            base_txt = ""
            if isinstance(title, str) and title.strip():
                base_txt = title.strip()
            else:
                vt = str(row.get("vaga_text", ""))
                base_txt = (vt[:80] + ("…" if len(vt) > 80 else ""))
            prefix = f"[{vid}] " if isinstance(vid, str) and vid else ""
            return prefix + base_txt

        options = vdf.apply(_vaga_label, axis=1).tolist()
        idx = st.selectbox("Selecione a vaga", options=range(len(options)), format_func=lambda i: options[i])
        col_l, col_r = st.columns([1,1])
        with col_l:
            top_n = st.number_input("Top N candidatos", 1, 100, 10)
        with col_r:
            clean_rank = st.checkbox("Aplicar limpeza nos textos", True)

        with st.expander("Ver descrição completa da vaga"):
            vaga_text_sel = str(vdf.iloc[idx]["vaga_text"]) if "vaga_text" in vdf.columns else ""
            st.write(vaga_text_sel)

        # Botão explícito para gerar o ranking
        gerar = st.button("🔍 Gerar ranking")

        if gerar:
            cvs = cdf["cv_text"].astype(str).tolist()
            if clean_rank:
                cvs = [clean_text(x) for x in cvs]
                vtxt = clean_text(vaga_text_sel)
            else:
                vtxt = vaga_text_sel

            with st.spinner("Preparando embeddings dos candidatos…"):
                if st.session_state.get("cand_embs") is None:
                    cvs = cdf["cv_text"].astype(str).tolist()
                    if clean_rank:
                        cvs = [clean_text(x) for x in cvs]
                    st.session_state["cand_embs"] = embed_texts(cvs, MODEL_DIR)
                emb_cvs = st.session_state["cand_embs"]

            with st.spinner("Calculando embedding da vaga…"):
                if st.session_state.get("vaga_embs") is not None and len(st.session_state["vaga_embs"]) == len(vdf):
                    emb_vaga = st.session_state["vaga_embs"][idx]
                    if clean_rank:
                        vtxt = clean_text(vaga_text_sel)
                        emb_vaga = embed_text(vtxt, MODEL_DIR)
                else:
                    vtxt = clean_rank and clean_text(vaga_text_sel) or vaga_text_sel
                    emb_vaga = embed_text(vtxt, MODEL_DIR)

            sims = emb_cvs @ emb_vaga  # (N_candidatos,)

            order = np.argsort(-sims)[: int(top_n)]
            rows = []
            for rank, i in enumerate(order, start=1):
                sim = float(sims[i])
                score = proportional_score(sim, limiar)
                row_out = {
                    "rank": rank,
                    "cand_index": int(i),
                    "similaridade": round(sim, 6),
                    "score": round(score, 2),
                    "aprovado": bool(sim >= limiar),
                }
                for col in ["candidate_id", "id", "nome", "name", "email", "telefone", "phone"]:
                    if col in cdf.columns:
                        row_out[col] = cdf.iloc[i][col]
                rows.append(row_out)
            st.session_state["_last_ranking_df"] = pd.DataFrame(rows)

        # Mostrar resultado se existir
        if "_last_ranking_df" in st.session_state:
            st.dataframe(st.session_state["_last_ranking_df"], use_container_width=True, hide_index=True)
            st.download_button(
                "Baixar ranking (CSV)",
                data=st.session_state["_last_ranking_df"].to_csv(index=False).encode("utf-8"),
                file_name="ranking_por_vaga.csv",
                mime="text/csv",
            )
        else:
            st.info("Clique em **Gerar ranking** para calcular o Top N candidatos da vaga selecionada.")

# -------- TAB 6: Bases atuais --------
with tab6:
    st.subheader("Bases atuais em memória")
    st.write("**Candidatos (cv_text)**")
    st.dataframe(st.session_state["candidatos_df"], use_container_width=True, hide_index=True)
    st.write("**Vagas (vaga_text)**")
    st.dataframe(st.session_state["vagas_df"], use_container_width=True, hide_index=True)

    st.info(
        "Essas bases são carregadas automaticamente se existirem nos caminhos padrão. "
        "Você pode substituí-las via uploads na aba 'Batch (2 uploads)'."
    )

# ======================== FOOTER ========================
st.caption(
    f"Modelo: {MODEL_NAME} • Limiar atual: {limiar:.2f} • App: {APP_NAME} v{APP_VERSION} — Rodando 100% Streamlit"
)

with tab6:
    st.subheader("Bases atuais em memória")
    st.write("**Candidatos (cv_text)**")
    st.dataframe(st.session_state["candidatos_df"], use_container_width=True, hide_index=True)
    st.write("**Vagas (vaga_text)**")
    st.dataframe(st.session_state["vagas_df"], use_container_width=True, hide_index=True)

    st.info(
        "Essas bases são carregadas automaticamente se existirem nos caminhos padrão. "
        "Você pode substituí-las via uploads na aba 'Batch (2 uploads)'."
    )

# ======================== FOOTER ========================
st.caption(
    f"Modelo: {MODEL_NAME} • Limiar atual: {limiar:.2f} • App: {APP_NAME} v{APP_VERSION} — Rodando 100% Streamlit"
)
