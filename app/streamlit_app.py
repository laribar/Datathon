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
DEFAULT_API_URL = os.getenv("MATCH_API_URL", "http://127.0.0.1:8000")

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

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["🔍 Match Único", "📚 Match em Lote (CSV)", "🔎 Explain (Top-N trechos)"])

# ========== TAB 1: SINGLE ==========
with tab1:
    st.subheader("🔍 Match Único")
    st.caption("Calcule o percentual de match entre um CV e uma Vaga.")

    c1, c2 = st.columns(2)
    with c1:
        cv_text = st.text_area("Currículo (texto)", height=240, placeholder="Cole aqui o texto do currículo...")
    with c2:
        vaga_text = st.text_area("Vaga (texto)", height=240, placeholder="Cole aqui o texto da vaga...")

    clean = st.checkbox("Aplicar limpeza básica (lower + normalização de espaços)", value=True)
    run_single = st.button("Calcular Match", type="primary")

    if run_single:
        if not cv_text.strip() or not vaga_text.strip():
            st.warning("Preencha ambos os campos: Currículo e Vaga.")
        else:
            with st.spinner("Calculando..."):
                payload = {"cv_text": cv_text, "vaga_text": vaga_text, "clean": clean}
                try:
                    r = api_post(API_URL, "/match", payload)
                    s1, s2, s3, s4 = st.columns(4)
                    s1.metric("Similarity (0..1)", f"{r['similarity']:.4f}")
                    s2.metric("Score (%)", f"{r['score']:.2f} {color_score(r['score'])}")
                    s3.metric("Limiar", f"{r['limiar_usado']:.2f}")
                    s4.metric("Aprovou limiar?", "✅ Sim" if r["passed_threshold"] else "❌ Não")

                    st.success(f"Modelo: `{r['model_name']}`")
                    st.json(r)
                except Exception as e:
                    st.error(f"Erro na requisição: {e}")


# ========== TAB 2: BATCH CSV (CVs × Vagas com Top-N) ==========
with tab2:
    st.subheader("📚 Match em Lote (CVs × Vagas)")
    st.caption(
        "Carregue dois CSVs: um de Currículos e outro de Vagas. "
        "Escolha quais colunas compõem o texto de cada lado. "
        "O app gera o cruzamento e mostra o **Top-N** por CV (ou por Vaga)."
    )

    # Upload de arquivos
    cvs_file = st.file_uploader("Carregar CSV de Currículos", type=["csv"], key="cvs")
    vagas_file = st.file_uploader("Carregar CSV de Vagas", type=["csv"], key="vagas")

    clean_batch = st.checkbox("Aplicar limpeza básica", value=True, key="clean_batch")

    # Parâmetros de ranking
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

            # Escolha de colunas para concatenar
            st.write("### Configuração das colunas para concatenação")
            cv_cols = st.multiselect(
                "Colunas do CSV de Currículos",
                df_cvs.columns.tolist(),
                default=[c for c in df_cvs.columns if "curriculo" in c or "cv" in c or c == "curriculo_pt"],
                help="Essas colunas serão concatenadas para formar o texto do currículo."
            )
            vaga_cols = st.multiselect(
                "Colunas do CSV de Vagas",
                df_vagas.columns.tolist(),
                default=[c for c in df_vagas.columns if "vaga" in c or "descricao" in c or "titulo" in c],
                help="Essas colunas serão concatenadas para formar o texto da vaga."
            )

            if st.button("Gerar Matches", type="primary"):
                if not cv_cols or not vaga_cols:
                    st.error("Selecione pelo menos uma coluna em **cada** dataset.")
                else:
                    with st.spinner("Calculando matches..."):
                        # Concatena colunas selecionadas
                        df_cvs["_texto_cv"] = df_cvs[cv_cols].astype(str).agg(" ".join, axis=1)
                        df_vagas["_texto_vaga"] = df_vagas[vaga_cols].astype(str).agg(" ".join, axis=1)

                        # Campos de identificação (se existirem)
                        cv_id_col = "id" if "id" in df_cvs.columns else None
                        cv_nome_col = "nome" if "nome" in df_cvs.columns else None
                        vaga_id_col = "id" if "id" in df_vagas.columns else None
                        vaga_titulo_col = "titulo_vaga" if "titulo_vaga" in df_vagas.columns else None

                        # Monta pares cartesianos: cada CV × cada Vaga
                        pairs = []
                        index_map = []  # (i_cv, j_vaga)
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

                    # Reconstrói resultados com metadados
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

                    # Aplica ranking
                    if rank_mode == "Top-N por CV":
                        df_ranked = (
                            df_all.sort_values(["id_cv", "score"], ascending=[True, False])
                                 .groupby("id_cv", as_index=False)
                                 .head(int(top_n))
                        )
                        st.success(
                            f"Processado {len(df_all)} combinações em {elapsed:.2f}s — exibindo Top-{int(top_n)} **por CV**. "
                            f"Modelo: {r['model_name']}"
                        )
                        # Ordena para visualização
                        df_ranked = df_ranked.sort_values(["id_cv", "score"], ascending=[True, False])
                    else:
                        df_ranked = (
                            df_all.sort_values(["id_vaga", "score"], ascending=[True, False])
                                 .groupby("id_vaga", as_index=False)
                                 .head(int(top_n))
                        )
                        st.success(
                            f"Processado {len(df_all)} combinações em {elapsed:.2f}s — exibindo Top-{int(top_n)} **por Vaga**. "
                            f"Modelo: {r['model_name']}"
                        )
                        df_ranked = df_ranked.sort_values(["id_vaga", "score"], ascending=[True, False])

                    # Mostra tabela e oferece download
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
        top_n = st.number_input("Top-N pares de sentenças", min_value=1, max_value=10, value=3, step=1)
    with c4:
        min_chars = st.number_input("Mínimo de caracteres por sentença", min_value=1, max_value=500, value=25, step=1)
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
                    "top_n": int(top_n),
                    "min_chars": int(min_chars),
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
                    st.dataframe(df_exp)

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
