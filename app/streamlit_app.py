# === STREAMLIT APP: Match CV × Vaga (FastAPI client) ===
# Requisitos: streamlit, requests, pandas
# Rodar: streamlit run app/streamlit_app.py

import os
import json
import time
import requests
import pandas as pd
import streamlit as st

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Match CV × Vaga", layout="wide")
DEFAULT_API_URL = os.getenv("MATCH_API_URL", "http://127.0.0.1:8000/")

# -----------------------------
# Helpers
# -----------------------------
def api_get(api_url: str, path: str, timeout=30):
    url = f"{api_url}{path}"
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()

def api_post(api_url: str, path: str, payload: dict, timeout=120):
    url = f"{api_url}{path}"
    headers = {"Content-Type": "application/json"}
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
    r.raise_for_status()
    return r.json()

@st.cache_data(show_spinner=False)
def check_health_cached(api_url: str):
    try:
        return api_get(api_url, "/health")
    except Exception as e:
        return {"status": "error", "error": str(e)}

def color_score(score: float) -> str:
    if score >= 90: return "✅"
    if score >= 75: return "🟨"
    return "❌"

def highlight_snippet(snippet: str) -> str:
    return f"<span style='background-color:#1f6feb22;padding:2px 4px;border-radius:4px'>{snippet}</span>"

def concat_cols(df: pd.DataFrame, cols: list) -> pd.Series:
    if not cols:
        return pd.Series([""] * len(df))
    return df[cols].astype(str).agg(" ".join, axis=1)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("⚙️ Configuração")
st.sidebar.write("Defina o endpoint da API (FastAPI).")

API_URL = st.sidebar.text_input(
    "MATCH_API_URL", value=DEFAULT_API_URL, help="Ex.: http://127.0.0.1:8000"
).strip()

st.sidebar.divider()
st.sidebar.caption("Dica: defina MATCH_API_URL como variável de ambiente para não precisar editar aqui.")

# -----------------------------
# Header
# -----------------------------
st.title("🧠 Match CV × Vaga — Streamlit UI")

health = check_health_cached(API_URL)
cols = st.columns(3)
with cols[0]:
    st.metric("API status", health.get("status", "desconhecido"))
with cols[1]:
    st.metric("Modelo", health.get("model_name", "—"))
with cols[2]:
    st.metric("Endpoint", API_URL)

if health.get("status") != "ok":
    st.error("Não consegui falar com a API. Verifique se o FastAPI está rodando (uvicorn) e a URL na barra lateral.")
    st.stop()

st.markdown("---")

# =========================================================
# Base compartilhada (upload persistente em sessão)
# =========================================================
st.header("📦 Base (Upload de CSVs)")
st.caption("Envie seus CSVs de **Currículos** e **Vagas** uma única vez. Eles ficarão disponíveis para o tab **🏅 Ranking por Vaga**.")

if "df_cvs" not in st.session_state: st.session_state.df_cvs = None
if "df_vagas" not in st.session_state: st.session_state.df_vagas = None
if "cv_cols_sel" not in st.session_state: st.session_state.cv_cols_sel = []
if "vaga_cols_sel" not in st.session_state: st.session_state.vaga_cols_sel = []

c_up1, c_up2 = st.columns(2)
with c_up1:
    cvs_file_base = st.file_uploader("Carregar CSV de Currículos (Base)", type=["csv"], key="cvs_base")
with c_up2:
    vagas_file_base = st.file_uploader("Carregar CSV de Vagas (Base)", type=["csv"], key="vagas_base")

if cvs_file_base is not None:
    try:
        st.session_state.df_cvs = pd.read_csv(cvs_file_base)
        st.success(f"CVs carregados: {st.session_state.df_cvs.shape[0]} linhas, {st.session_state.df_cvs.shape[1]} colunas.")
        st.dataframe(st.session_state.df_cvs.head(10), use_container_width=True)
    except Exception as e:
        st.error(f"Erro ao ler CSV de Currículos: {e}")

if vagas_file_base is not None:
    try:
        st.session_state.df_vagas = pd.read_csv(vagas_file_base)
        st.success(f"Vagas carregadas: {st.session_state.df_vagas.shape[0]} linhas, {st.session_state.df_vagas.shape[1]} colunas.")
        st.dataframe(st.session_state.df_vagas.head(10), use_container_width=True)
    except Exception as e:
        st.error(f"Erro ao ler CSV de Vagas: {e}")

if (st.session_state.df_cvs is not None) and (st.session_state.df_vagas is not None):
    st.write("### Configurar Colunas para Concatenação")
    c_conf1, c_conf2 = st.columns(2)
    with c_conf1:
        st.session_state.cv_cols_sel = st.multiselect(
            "Colunas do CSV de Currículos (texto do CV)",
            st.session_state.df_cvs.columns.tolist(),
            default=[c for c in st.session_state.df_cvs.columns if "curriculo" in c.lower() or "cv" in c.lower() or c.lower() == "curriculo_pt"],
            help="Essas colunas serão concatenadas para formar o texto do currículo."
        )
    with c_conf2:
        st.session_state.vaga_cols_sel = st.multiselect(
            "Colunas do CSV de Vagas (texto da Vaga)",
            st.session_state.df_vagas.columns.tolist(),
            default=[c for c in st.session_state.df_vagas.columns if ("vaga" in c.lower() or "descri" in c.lower() or "titulo" in c.lower())],
            help="Essas colunas serão concatenadas para formar o texto da vaga."
        )

st.markdown("---")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Match Único",
    "📚 Match em Lote (CSV)",
    "🔎 Explain (Top-N trechos)",
    "🏅 Ranking por Vaga"
])

# ========== TAB 1: SINGLE ==========
with tab1:
    st.subheader("🔍 Match Único")
    st.caption("Calcule o percentual de match entre um CV e uma Vaga. Ao calcular, exibimos automaticamente as sentenças mais próximas (Explain).")

    c1, c2 = st.columns(2)
    with c1:
        cv_text = st.text_area("Currículo (texto)", height=240, placeholder="Cole aqui o texto do currículo...")
    with c2:
        vaga_text = st.text_area("Vaga (texto)", height=240, placeholder="Cole aqui o texto da vaga...")

    clean = st.checkbox("Aplicar limpeza básica (lower + normalização de espaços)", value=True)

    with st.expander("Opções de Explain (automático no Match Único)"):
        top_n_single_exp = st.number_input("Top-N pares de sentenças", min_value=1, max_value=10, value=3, step=1, key="top_n_single_exp")
        min_chars_single_exp = st.number_input("Mínimo de caracteres por sentença", min_value=1, max_value=500, value=25, step=1, key="min_chars_single_exp")

    run_single = st.button("Calcular Match", type="primary")

    if run_single:
        if not cv_text.strip() or not vaga_text.strip():
            st.warning("Preencha ambos os campos: Currículo e Vaga.")
        else:
            # 1) Match
            with st.spinner("Calculando match..."):
                payload = {"cv_text": cv_text, "vaga_text": vaga_text, "clean": clean}
                try:
                    r = api_post(API_URL, "/match", payload)
                    s1, s2, s3, s4 = st.columns(4)
                    s1.metric("Similarity (0..1)", f"{r['similarity']:.4f}")
                    s2.metric("Score (%)", f"{r['score']:.2f} {color_score(r['score'])}")
                    s3.metric("Limiar", f"{r['limiar_usado']:.2f}")
                    s4.metric("Aprovou limiar?", "✅ Sim" if r["passed_threshold"] else "❌ Não")

                    st.success(f"Modelo: `{r['model_name']}`")
                    with st.expander("Resposta bruta da API (/match)"):
                        st.json(r)
                except Exception as e:
                    st.error(f"Erro na requisição /match: {e}")
                    st.stop()

            # 2) Explain automático
            try:
                with st.spinner("Gerando Explain (Top-N trechos semelhantes)..."):
                    exp_payload = {
                        "cv_text": cv_text,
                        "vaga_text": vaga_text,
                        "clean": clean,
                        "top_n": int(top_n_single_exp),
                        "min_chars": int(min_chars_single_exp),
                    }
                    rex = api_post(API_URL, "/match/explain", exp_payload)

                st.markdown("### 🔎 Explain — Top pares de sentenças")
                e1, e2 = st.columns(2)
                with e1:
                    st.metric("Overall Similarity", f"{rex['overall_similarity']:/.4f}")
                with e2:
                    st.metric("Overall Score (%)", f"{rex['overall_score']:.2f} {color_score(rex['overall_score'])}")

                exp_rows = []
                for item in rex["top_pairs"]:
                    exp_rows.append({
                        "rank": item["rank"],
                        "similarity": item["similarity"],
                        "cv_index": item["cv_index"],
                        "vaga_index": item["vaga_index"],
                        "cv_snippet": item["cv_snippet"],
                        "vaga_snippet": item["vaga_snippet"],
                    })
                df_exp_single = pd.DataFrame(exp_rows)
                st.dataframe(df_exp_single, use_container_width=True)

                st.write("#### Destaques (preview)")
                for item in rex["top_pairs"]:
                    st.markdown(
                        f"**#{item['rank']}** — sim: `{item['similarity']:.4f}`  \n"
                        f"CV: {highlight_snippet(item['cv_snippet'])}  \n"
                        f"Vaga: {highlight_snippet(item['vaga_snippet'])}",
                        unsafe_allow_html=True
                    )

                st.info(f"Explain gerado com `top_n={int(top_n_single_exp)}` e `min_chars={int(min_chars_single_exp)}` — Modelo: `{rex['model_name']}` | Limiar: `{rex['limiar_usado']}`")
            except Exception as e:
                st.warning(f"Não foi possível gerar o Explain automático: {e}")

# ========== TAB 2: BATCH CSV (CVs × Vagas com Top-N) ==========
with tab2:
    st.subheader("📚 Match em Lote (CVs × Vagas)")
    st.caption(
        "Carregue dois CSVs: um de Currículos e outro de Vagas. "
        "Escolha quais colunas compõem o texto de cada lado. "
        "O app gera o cruzamento e mostra o **Top-N** por CV (ou por Vaga)."
    )

    cvs_file = st.file_uploader("Carregar CSV de Currículos (Lote)", type=["csv"], key="cvs")
    vagas_file = st.file_uploader("Carregar CSV de Vagas (Lote)", type=["csv"], key="vagas")

    clean_batch = st.checkbox("Aplicar limpeza básica", value=True, key="clean_batch")

    c1, c2 = st.columns([1, 1])
    with c1:
        top_n = st.number_input("Top-N resultados", min_value=1, max_value=50, value=5, step=1)
    with c2:
        rank_mode = st.radio(
            "Modo de ranking",
            options=["Top-N por CV", "Top-N por Vaga"],
            index=0,
            horizontal=True,
        )

    if cvs_file is not None and vagas_file is not None:
        try:
            df_cvs = pd.read_csv(cvs_file)
            df_vagas = pd.read_csv(vagas_file)

            st.write("### Amostra de Currículos")
            st.dataframe(df_cvs.head(10), use_container_width=True)
            st.write("### Amostra de Vagas")
            st.dataframe(df_vagas.head(10), use_container_width=True)

            st.write("### Configuração das colunas para concatenação")
            cv_cols = st.multiselect(
                "Colunas do CSV de Currículos",
                df_cvs.columns.tolist(),
                default=[c for c in df_cvs.columns if "curriculo" in c.lower() or "cv" in c.lower() or c.lower() == "curriculo_pt"],
                help="Essas colunas serão concatenadas para formar o texto do currículo."
            )
            vaga_cols = st.multiselect(
                "Colunas do CSV de Vagas",
                df_vagas.columns.tolist(),
                default=[c for c in df_vagas.columns if ("vaga" in c.lower() or "descri" in c.lower() or "titulo" in c.lower())],
                help="Essas colunas serão concatenadas para formar o texto da vaga."
            )

            if st.button("Gerar Matches", type="primary"):
                if not cv_cols or not vaga_cols:
                    st.error("Selecione pelo menos uma coluna em **cada** dataset.")
                else:
                    with st.spinner("Calculando matches..."):
                        df_cvs["_texto_cv"] = concat_cols(df_cvs, cv_cols)
                        df_vagas["_texto_vaga"] = concat_cols(df_vagas, vaga_cols)

                        cv_id_col = "id" if "id" in df_cvs.columns else None
                        cv_nome_col = "nome" if "nome" in df_cvs.columns else None
                        vaga_id_col = "id" if "id" in df_vagas.columns else None
                        vaga_titulo_col = "titulo_vaga" if "titulo_vaga" in df_vagas.columns else None

                        pairs, index_map = [], []
                        for i, cv_row in df_cvs.iterrows():
                            for j, vaga_row in df_vagas.iterrows():
                                pairs.append({
                                    "cv_text": cv_row["_texto_cv"],
                                    "vaga_text": vaga_row["_texto_vaga"],
                                })
                                index_map.append((i, j))

                        payload = {"pairs": pairs, "clean": clean_batch}

                        t0 = time.time()
                        r = api_post(API_URL, "/match/batch", payload)
                        elapsed = time.time() - t0

                    records = []
                    for k, item in enumerate(r["results"]):
                        i, j = index_map[k]
                        cv_row = df_cvs.loc[i]
                        vaga_row = df_vagas.loc[j]
                        records.append({
                            "id_cv": cv_row[cv_id_col] if cv_id_col else i,
                            "nome_cv": cv_row[cv_nome_col] if cv_nome_col else "",
                            "id_vaga": vaga_row[vaga_id_col] if vaga_id_col else j,
                            "titulo_vaga": vaga_row[vaga_titulo_col] if vaga_titulo_col else "",
                            "similarity": item["similarity"],
                            "score": item["score"],
                            "passed_threshold": item["passed_threshold"],
                        })
                    df_all = pd.DataFrame(records)

                    if rank_mode == "Top-N por CV":
                        df_ranked = (
                            df_all.sort_values(["id_cv", "score"], ascending=[True, False])
                                 .groupby("id_cv", as_index=False).head(int(top_n))
                                 .sort_values(["id_cv", "score"], ascending=[True, False])
                        )
                        st.success(
                            f"Processado {len(df_all)} combinações em {elapsed:.2f}s — exibindo Top-{int(top_n)} **por CV**. "
                            f"Modelo: {r['model_name']}"
                        )
                    else:
                        df_ranked = (
                            df_all.sort_values(["id_vaga", "score"], ascending=[True, False])
                                 .groupby("id_vaga", as_index=False).head(int(top_n))
                                 .sort_values(["id_vaga", "score"], ascending=[True, False])
                        )
                        st.success(
                            f"Processado {len(df_all)} combinações em {elapsed:.2f}s — exibindo Top-{int(top_n)} **por Vaga**. "
                            f"Modelo: {r['model_name']}"
                        )

                    st.dataframe(df_ranked, use_container_width=True)
                    st.download_button(
                        "Baixar resultados (CSV)",
                        df_ranked.to_csv(index=False).encode("utf-8"),
                        file_name="resultados_match_topN.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            st.error(f"Erro ao processar os CSVs: {e}")

# ========== TAB 3: EXPLAIN ==========
with tab3:
    st.subheader("🔎 Explain (Top-N trechos)")
    st.caption("Mostra as sentenças de CV e Vaga mais semelhantes, além do score geral.")

    c1, c2 = st.columns(2)
    with c1:
        cv_text_ex = st.text_area("Currículo (texto)", height=220, key="cv_explain", placeholder="Cole aqui o texto do currículo...")
    with c2:
        vaga_text_ex = st.text_area("Vaga (texto)", height=220, key="vaga_explain", placeholder="Cole aqui o texto da vaga...")

    c3, c4, c5 = st.columns(3)
    with c3:
        top_n_explain_tab = st.number_input("Top-N pares de sentenças", min_value=1, max_value=10, value=3, step=1, key="top_n_explain_tab")
    with c4:
        min_chars_explain_tab = st.number_input("Mínimo de caracteres por sentença", min_value=1, max_value=500, value=25, step=1, key="min_chars_explain_tab")
    with c5:
        clean_explain = st.checkbox("Aplicar limpeza básica", value=True, key="clean_explain")

    if st.button("Gerar Explain", type="primary"):
        if not cv_text_ex.strip() or not vaga_text_ex.strip():
            st.warning("Preencha ambos os campos: Currículo e Vaga.")
        else:
            with st.spinner("Calculando explain..."):
                payload = {
                    "cv_text": cv_text_ex,
                    "vaga_text": vaga_text_ex,
                    "clean": clean_explain,
                    "top_n": int(top_n_explain_tab),
                    "min_chars": int(min_chars_explain_tab),
                }
                try:
                    r = api_post(API_URL, "/match/explain", payload)

                    s1, s2 = st.columns(2)
                    with s1:
                        st.metric("Overall Similarity", f"{r['overall_similarity']:.4f}")
                    with s2:
                        st.metric("Overall Score (%)", f"{r['overall_score']:.2f} {color_score(r['overall_score'])}")

                    st.write("### Top pares de sentenças")
                    exp_rows = []
                    for item in r["top_pairs"]:
                        exp_rows.append({
                            "rank": item["rank"],
                            "similarity": item["similarity"],
                            "cv_index": item["cv_index"],
                            "vaga_index": item["vaga_index"],
                            "cv_snippet": item["cv_snippet"],
                            "vaga_snippet": item["vaga_snippet"],
                        })
                    df_exp = pd.DataFrame(exp_rows)
                    st.dataframe(df_exp, use_container_width=True)

                    st.write("#### Destaques (preview)")
                    for item in r["top_pairs"]:
                        st.markdown(
                            f"**#{item['rank']}** — sim: `{item['similarity']:.4f}`  \n"
                            f"CV: {highlight_snippet(item['cv_snippet'])}  \n"
                            f"Vaga: {highlight_snippet(item['vaga_snippet'])}",
                            unsafe_allow_html=True
                        )

                    st.success(f"Modelo: `{r['model_name']}` — Limiar: `{r['limiar_usado']}`")

                except Exception as e:
                    st.error(f"Erro na requisição: {e}")

# ========== TAB 4: RANKING POR VAGA ==========
with tab4:
    st.subheader("🏅 Ranking por Vaga")
    st.caption("Use a **Base** carregada acima. Escolha **uma vaga** e gere o **ranking de candidatos** (Top-N) dessa vaga.")

    if (st.session_state.df_cvs is None) or (st.session_state.df_vagas is None):
        st.warning("Envie primeiro os CSVs na seção **📦 Base (Upload de CSVs)** acima.")
        st.stop()

    clean_rank = st.checkbox("Aplicar limpeza básica", value=True, key="clean_rank")
    top_n_rank = st.number_input("Top-N candidatos por vaga", min_value=1, max_value=200, value=20, step=1)

    # Concatena colunas conforme configurado na Base
    df_cvs_b = st.session_state.df_cvs.copy()
    df_vagas_b = st.session_state.df_vagas.copy()
    df_cvs_b["_texto_cv"] = concat_cols(df_cvs_b, st.session_state.cv_cols_sel)
    df_vagas_b["_texto_vaga"] = concat_cols(df_vagas_b, st.session_state.vaga_cols_sel)

    # Campos de identificação sugestivos
    cv_id_col = "id" if "id" in df_cvs_b.columns else None
    cv_nome_col = "nome" if "nome" in df_cvs_b.columns else None

    # Para selecionar a vaga: tenta criar um label amigável
    # prioridade: 'id' + 'titulo_vaga' + preview de _texto_vaga
    vaga_id_col = "id" if "id" in df_vagas_b.columns else None
    vaga_titulo_col = None
    for c in df_vagas_b.columns:
        cl = c.lower()
        if "titulo" in cl and "vaga" in cl:
            vaga_titulo_col = c
            break
        if cl == "titulo" or cl == "title":
            vaga_titulo_col = c
            break

    # Monta opções de seleção
    def _vaga_label(row):
        titulo = str(row[vaga_titulo_col]) if vaga_titulo_col else ""
        vid = str(row[vaga_id_col]) if vaga_id_col else ""
        preview = str(row["_texto_vaga"])[:80].replace("\n", " ")
        if titulo and vid:
            return f"[{vid}] {titulo} — {preview}..."
        if titulo:
            return f"{titulo} — {preview}..."
        if vid:
            return f"[{vid}] {preview}..."
        return preview + "..."

    vagas_options = df_vagas_b.apply(_vaga_label, axis=1).tolist()
    idx_vaga = st.selectbox("Escolha a vaga para rankear candidatos", options=list(range(len(vagas_options))),
                            format_func=lambda i: vagas_options[i])

    vaga_sel = df_vagas_b.iloc[idx_vaga]
    st.write("#### Prévia da Vaga Selecionada")
    st.text_area("Texto da Vaga (concat.)", value=vaga_sel["_texto_vaga"], height=160, disabled=True)

    # Botão para buscar ranking
    if st.button("🔎 Buscar na Base — Rankear Candidatos", type="primary"):
        with st.spinner("Calculando ranking de candidatos..."):
            # pares: cada CV contra a mesma vaga selecionada
            pairs = [{"cv_text": cv_text, "vaga_text": vaga_sel["_texto_vaga"]} for cv_text in df_cvs_b["_texto_cv"].tolist()]
            payload = {"pairs": pairs, "clean": clean_rank}
            try:
                r = api_post(API_URL, "/match/batch", payload)
            except Exception as e:
                st.error(f"Erro na requisição /match/batch: {e}")
                st.stop()

        # Monta ranking por CV
        recs = []
        for i, item in enumerate(r["results"]):
            cv_row = df_cvs_b.iloc[i]
            recs.append({
                "rank": None,  # vamos preencher após ordenar
                "id_cv": cv_row[cv_id_col] if cv_id_col else i,
                "nome_cv": cv_row[cv_nome_col] if cv_nome_col else "",
                "similarity": item["similarity"],
                "score": item["score"],
                "passed_threshold": item["passed_threshold"],
            })

        df_rank = pd.DataFrame(recs).sort_values("score", ascending=False)
        df_rank["rank"] = range(1, len(df_rank) + 1)
        df_rank = df_rank.head(int(top_n_rank))

        st.success(f"Ranking gerado — Modelo: {r.get('model_name','?')} | Vaga idx: {idx_vaga}")
        st.dataframe(df_rank, use_container_width=True)

        st.download_button(
            "Baixar Ranking (CSV)",
            df_rank.to_csv(index=False).encode("utf-8"),
            file_name="ranking_candidatos_por_vaga.csv",
            mime="text/csv"
        )
