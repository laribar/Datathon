# streamlit_app.py — Código Final para Especialização (Modelo Local)
import os, re, json, hashlib
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import requests # Para ler URLs de dados

# ======================== CONFIG ========================
APP_NAME = "RECRUT.AI 🚀"
APP_VERSION = "1.3.0 (Especialização)"

DEFAULT_LIMIAR = float(os.getenv("SCORE_LIMIAR", "0.75"))

# 📌 ALTERAÇÃO CRÍTICA: Definimos o modelo como a pasta local DO SEU PROJETO
# O SentenceTransformer irá carregar o modelo salvo DENTRO desta pasta.
# Certifique-se de que a pasta 'models/sbert_encoder' exista no seu repositório.
MODEL_DIR = os.getenv("MODEL_DIR", "models/sbert_encoder")
MODEL_NAME = os.getenv("MODEL_NAME", f"Local_Custom:{MODEL_DIR}") 
# Usamos um fallback, mas a prioridade é o modelo local
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") 

# Dados: URLs RAW do GitHub têm prioridade (melhor para o Streamlit Cloud)
BASE_CANDIDATOS_PATH = os.getenv("BASE_CANDIDATOS_PATH", "data/applicants_clean.csv")
BASE_VAGAS_PATH = os.getenv("BASE_VAGAS_PATH", "data/vagas_clean.csv")
CANDIDATOS_CSV_URL = os.getenv("CANDIDATOS_CSV_URL") 
VAGAS_CSV_URL = os.getenv("VAGAS_CSV_URL") 

# Cache de embeddings (mantido, mas opcional/efêmero no Cloud)
EMB_DIR = Path(os.getenv("EMB_DIR", "data/embeddings")); EMB_DIR.mkdir(parents=True, exist_ok=True)
CAND_EMB_PATH = EMB_DIR / "candidatos.npy"
CAND_META_PATH = EMB_DIR / "candidatos.meta.json"
VAGA_EMB_PATH = EMB_DIR / "vagas.npy"
VAGA_META_PATH = EMB_DIR / "vagas.meta.json"

# ======================== TEXT UTILS (Sem Alteração) ========================
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

def top_n_pairs_by_cosine_optimized(A: List[str], B: List[str], encoder, top_n: int = 3):
    if not A or not B: return []
    A_vecs = encoder.encode(A, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
    B_vecs = encoder.encode(B, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
    similarity_matrix = A_vecs @ B_vecs.T
    flat_indices = np.argpartition(similarity_matrix.flatten(), -top_n)[-top_n:]
    pairs_indices = np.unravel_index(flat_indices, similarity_matrix.shape)
    pairs = []
    for i, j in zip(*pairs_indices):
        pairs.append((int(i), int(j), float(similarity_matrix[i, j])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs

# ======================== EMB UTILS (Pequenas Alterações no Log) ========================
def _hash_dataframe(df: pd.DataFrame) -> str:
    buf = df.to_csv(index=False).encode("utf-8")
    return hashlib.md5(buf).hexdigest()

def _save_embeddings(npy_path: Path, meta_path: Path, embs: np.ndarray, meta: dict) -> None:
    try:
        np.save(npy_path, embs)
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass # Falha silenciosamente se o disco for efêmero

def _load_embeddings(npy_path: Path, meta_path: Path) -> Tuple[np.ndarray | None, dict]:
    if not npy_path.exists() or not meta_path.exists():
        return None, {}
    try:
        arr = np.load(npy_path); meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return arr, meta
    except Exception:
        return None, {}

def get_or_build_embeddings(df: pd.DataFrame, text_col: str, npy_path: Path, meta_path: Path, model_dir: str) -> np.ndarray:
    """Tenta carregar do cache local ou constrói e salva os embeddings."""
    sig = _hash_dataframe(df[[text_col]])
    # Usamos MODEL_DIR como assinatura
    meta_expected = {"model_source": model_dir, "text_col": text_col, "signature": sig, "version": APP_VERSION}
    
    # 1. Tenta carregar do cache local
    embs, meta = _load_embeddings(npy_path, meta_path)
    if embs is not None and meta == meta_expected and embs.ndim == 2 and embs.shape[0] == len(df):
        st.info(f"💾 Cache de embeddings para '{text_col}' carregado do disco.")
        return embs
        
    # 2. Se falhar ou não existir, constrói (lento)
    st.warning(f"⏳ Reconstruindo embeddings para '{text_col}' (Não há cache válido).")
    texts = df[text_col].astype(str).tolist()
    embs = embed_texts(texts, model_dir) 
    _save_embeddings(npy_path, meta_path, embs, meta_expected)
    st.success(f"✅ Embeddings gerados. Salvo (se o disco não for efêmero).")
    return embs

# ======================== MODEL / ENCODER (Foco no Local) ========================
@st.cache_resource(show_spinner="Carregando Encoder (Priorizando Modelo Local)...", ttl=None)
def load_model(model_path: str):
    from sentence_transformers import SentenceTransformer
    
    # 📌 Prioriza o caminho local do projeto (seu modelo)
    st.info(f"Fazendo load do modelo a partir do caminho: **{model_path}**")
    try:
        model = SentenceTransformer(model_path) 
        st.success(f"✅ Modelo customizado carregado com sucesso de: **{model_path}**")
        return model
    except Exception as e:
        # Fallback para o Hub (caso o modelo não esteja na pasta esperada)
        st.error(f"❌ Falha ao carregar modelo de: {model_path}. Tentando fallback: {HF_MODEL_NAME}. Erro: {e}")
        try:
            model = SentenceTransformer(HF_MODEL_NAME)
            st.warning(f"⚠️ Modelo padrão do Hugging Face carregado como fallback.")
            return model
        except Exception:
            st.error("❌ Falha crítica: Não foi possível carregar o modelo de nenhuma fonte.")
            st.stop()
            

@st.cache_data(show_spinner="Gerando embeddings em lote...", ttl=3600, max_entries=5)
def embed_texts(texts: List[str], model_path: str) -> np.ndarray:
    model = load_model(model_path)
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)

@st.cache_data(show_spinner="Gerando embedding...", ttl=3600, max_entries=20)
def embed_text(text: str, model_path: str) -> np.ndarray:
    model = load_model(model_path)
    return model.encode(text, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)

# ======================== DATA LOADERS (Prioridade na URL) ========================
@st.cache_data(show_spinner=False)
def _concat_all_columns(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    out = df.copy()
    if target_col in out.columns: out = out.drop(columns=[target_col])
    s = out.fillna("").astype(str)
    out[target_col] = s.apply(lambda row: " ".join([v for v in row.values if v != ""]).strip(), axis=1)
    return out

@st.cache_data(show_spinner=False)
def _read_csv_local_or_url(local_path: str, url_env: str | None) -> pd.DataFrame | None:
    # 1. Tenta URL (Prioridade - Para deploy Cloud)
    if url_env:
        try:
            r = requests.head(url_env, timeout=5)
            if r.status_code == 200:
                st.info(f"🔗 Dados lidos da URL: {url_env}")
                return pd.read_csv(url_env)
        except Exception: 
            pass
            
    # 2. Tenta caminho local (Fallback - Para teste local)
    try:
        if os.path.exists(local_path): 
            st.info(f"📂 Dados lidos do disco local: {local_path}")
            return pd.read_csv(local_path)
    except Exception: 
        pass
        
    return None

@st.cache_data(show_spinner="Carregando bases de dados...", ttl=None)
def load_fixed_bases() -> Tuple[pd.DataFrame, pd.DataFrame, list]:
    logs = []
    cand = _read_csv_local_or_url(BASE_CANDIDATOS_PATH, CANDIDATOS_CSV_URL)
    vaga = _read_csv_local_or_url(BASE_VAGAS_PATH, VAGAS_CSV_URL)

    # Dados de Amostra (Fallback final)
    if cand is None or cand.empty:
        logs.append("⚠️ Não encontrei candidatos via URL ou local. Usando amostra.")
        cand = pd.DataFrame({
            "nome": ["Ana Silva", "Carlos Souza", "Maria Oliveira", "Pedro Rocha"],
            "skills": ["Python Airflow Spark", "Java Spring SQL", "Python Pandas Matplotlib", "C++ Linux Multithreading"],
            "experiencia": ["3 anos em dados", "5 anos em backend", "2 anos em análise", "8 anos em sistemas embarcados"],
            "cidade": ["São Paulo", "Rio de Janeiro", "Belo Horizonte", "Curitiba"]
        })
    if vaga is None or vaga.empty:
        logs.append("⚠️ Não encontrei vagas via URL ou local. Usando amostra.")
        vaga = pd.DataFrame({
            "titulo": ["Engenheira de Dados Senior", "Desenvolvedor Backend Java", "Analista de Dados Júnior"],
            "requisitos": ["Python, Spark, Airflow, AWS", "Java, Spring Boot, SQL, REST APIs", "Python, Pandas, Power BI, SQL"],
            "descricao": ["Projetos de dados em ambiente cloud. Criação de pipelines ETL.", "Desenvolvimento de microserviços de alta performance.", "Criação de relatórios e dashboards. Suporte a tomada de decisão."]
        })

    cand = _concat_all_columns(cand, "cv_text")
    vaga = _concat_all_columns(vaga, "vaga_text")
    
    logs.append(f"✅ Bases carregadas: {len(cand)} candidatos e {len(vaga)} vagas.")
    
    return cand, vaga, logs

# ======================== LÓGICA DE INICIALIZAÇÃO ========================

# 1. Configurações da Página
st.set_page_config(
    page_title=APP_NAME, 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# 2. Carrega o modelo (Priorizando o MODEL_DIR)
with st.spinner(f"Preparando o Encoder (Seu Modelo: {MODEL_DIR})..."):
    try:
        load_model(MODEL_DIR)
    except Exception:
        st.stop()


# 3. Carrega Bases Fixas (priorizando URL)
candidatos_df, vagas_df, _logs = load_fixed_bases()

# Guardar bases na session_state
if "candidatos_df" not in st.session_state:
    st.session_state["candidatos_df"] = candidatos_df.copy()
if "vagas_df" not in st.session_state:
    st.session_state["vagas_df"] = vagas_df.copy()

# 4. Tenta carregar embeddings de base OU recalcula se o cache estiver vazio
cand_embs_cache = None
vaga_embs_cache = None
cache_loaded = False
if not CAND_EMB_PATH.exists() or not VAGA_EMB_PATH.exists():
    with st.spinner("⚡ Preparando embeddings iniciais (base)..."):
        try:
            # Usamos MODEL_DIR como source para o get_or_build_embeddings
            cand_embs_cache = get_or_build_embeddings(st.session_state["candidatos_df"], "cv_text", CAND_EMB_PATH, CAND_META_PATH, MODEL_DIR)
            vaga_embs_cache = get_or_build_embeddings(st.session_state["vagas_df"], "vaga_text", VAGA_EMB_PATH, VAGA_META_PATH, MODEL_DIR)
            cache_loaded = True
        except Exception:
            st.warning("Não foi possível gerar os embeddings iniciais.")
else:
    try:
        cand_embs_cache, _ = _load_embeddings(CAND_EMB_PATH, CAND_META_PATH)
        vaga_embs_cache, _ = _load_embeddings(VAGA_EMB_PATH, VAGA_META_PATH)
        cache_loaded = True
    except Exception:
        pass

st.session_state["cache_loaded"] = cache_loaded
st.session_state["cand_embs_cache"] = cand_embs_cache
st.session_state["vaga_embs_cache"] = vaga_embs_cache


# ======================== SIDEBAR E UI PRINCIPAL ========================

with st.sidebar:
    st.markdown(f"## {APP_NAME} 🤖")
    st.caption(f"Versão {APP_VERSION}")
    st.write("---")
    st.markdown("#### Configurações Globais")
    st.write("**Encoder:**", MODEL_NAME)
    limiar = st.slider("Limiar de Aprovação (Cosine)", 0.50, 0.95, DEFAULT_LIMIAR, 0.01)

    st.divider()
    st.markdown("#### 🛠️ Cache de Embeddings")
    st.info(f"Status do Cache: {'✅ Disponível' if cache_loaded else '❌ Indisponível'}")

    if st.button("⚡ Gerar/Atualizar Cache de Base", help="Salva os embeddings das bases atuais no disco (Se o armazenamento não for efêmero)."):
        with st.spinner("Gerando e salvando cache (candidatos e vagas)…"):
            st.session_state["cand_embs_cache"] = get_or_build_embeddings(st.session_state["candidatos_df"], "cv_text", CAND_EMB_PATH, CAND_META_PATH, MODEL_DIR)
            st.session_state["vaga_embs_cache"] = get_or_build_embeddings(st.session_state["vagas_df"], "vaga_text", VAGA_EMB_PATH, VAGA_META_PATH, MODEL_DIR)
            st.session_state["cache_loaded"] = True
            st.success(f"Cache gerado. Recarregue a página para usar o cache.")


# Exibe logs de carregamento de base
if _logs:
    with st.expander("Logs de Carregamento de Bases"):
        for m in _logs: st.write(m)


st.title("🔎 Match CV × Vaga (SBERT) - Análise Semântica")
st.markdown(f"**Modelo em Uso:** {MODEL_DIR}")
st.markdown("Use as abas abaixo para realizar diferentes tipos de análise de similaridade.")

# -------------------- TABS --------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1×1 (Texto)", "Batch (Uploads)", "Explain (Trechos)", "Ranking Base", "Bases de Dados",
])

# ############## TAB 1: Match 1x1 ##############
with tab1:
    st.subheader("Match 1×1 — Cálculo de Similaridade")
    colA, colB = st.columns(2)
    with colA: 
        cv_text = st.text_area("Currículo (CV) — texto puro", height=250, placeholder="Cole o texto do currículo aqui...", key="t1_cv")
    with colB: 
        vaga_text = st.text_area("Descrição da Vaga — texto puro", height=250, placeholder="Cole a descrição da vaga aqui...", key="t1_vaga")
    
    clean = st.checkbox("Aplicar limpeza (lower + remover quebras)", True)
    
    if st.button("Calcular Similaridade", key="btn_1x1", use_container_width=True):
        if not cv_text or not vaga_text:
            st.warning("Informe o texto do CV e da Vaga para calcular.")
        else:
            with st.spinner("Calculando similaridade..."):
                cv_raw = clean_text(cv_text) if clean else cv_text
                vaga_raw = clean_text(vaga_text) if clean else vaga_text
                
                # Usa o MODEL_DIR
                cv_vec = embed_text(cv_raw, MODEL_DIR).flatten()
                vaga_vec = embed_text(vaga_raw, MODEL_DIR).flatten()
                
                sim = float(np.dot(cv_vec, vaga_vec))
                score = proportional_score(sim, limiar)
                aprovado = sim >= limiar
                
                st.write("---")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Similaridade (Cosine)", f"{sim:.4f}")
                with col2:
                    st.metric("Score (%)", f"{score:.2f}%", help="Proporcional ao Limiar")
                with col3:
                    status_emoji = "🎉 Aprovado" if aprovado else "❌ Não Aprovado"
                    st.metric("Status (Limiar > 0.75)", status_emoji)
                
                progress_val = min(sim / limiar, 1.0) if sim < limiar else 1.0
                st.progress(progress_val, f"Progresso para o Limiar ({limiar:.2f}): {progress_val*100:.1f}%")

# ############## TAB 2: Batch (Uploads) ##############
with tab2:
    st.subheader("Batch — Comparação de Bases Uploaded (N×M)")
    st.caption("Upload de duas bases separadas (Candidatos e Vagas). Calculamos o Top K de vagas para cada candidato.")
    
    col_up_l, col_up_r = st.columns(2)
    with col_up_l:
        up_cand = st.file_uploader("Upload CSV Candidatos", type=["csv"], key="u1_cand")
        if up_cand:
            candidatos_df_batch = _concat_all_columns(pd.read_csv(up_cand), "cv_text")
            st.info(f"{len(candidatos_df_batch)} candidatos carregados.")
            st.dataframe(candidatos_df_batch.head(3), use_container_width=True)
        else:
            candidatos_df_batch = pd.DataFrame()

    with col_up_r:
        up_vaga = st.file_uploader("Upload CSV Vagas", type=["csv"], key="u2_vaga")
        if up_vaga:
            vagas_df_batch = _concat_all_columns(pd.read_csv(up_vaga), "vaga_text")
            st.info(f"{len(vagas_df_batch)} vagas carregadas.")
            st.dataframe(vagas_df_batch.head(3), use_container_width=True)
        else:
            vagas_df_batch = pd.DataFrame()
            
    top_k = st.number_input("Top K de Vagas por Candidato", 1, 100, 5, key="topk_batch")
    
    if st.button("Processar Match N×M", key="btn_batch", use_container_width=True):
        if len(candidatos_df_batch) == 0 or len(vagas_df_batch) == 0:
            st.error("Por favor, faça upload de bases válidas para Candidatos e Vagas.")
        else:
            with st.spinner(f"Processando {len(candidatos_df_batch)} CVs × {len(vagas_df_batch)} Vagas..."):
                emb_cvs = embed_texts(candidatos_df_batch["cv_text"].astype(str).tolist(), MODEL_DIR)
                emb_vgs = embed_texts(vagas_df_batch["vaga_text"].astype(str).tolist(), MODEL_DIR)
                sim_matrix = emb_cvs @ emb_vgs.T
                
                rows = []
                for i in range(len(candidatos_df_batch)):
                    sims = sim_matrix[i]; top_idx = np.argsort(-sims)[:top_k]
                    for rank, j in enumerate(top_idx, start=1):
                        rows.append({
                            "Candidato (Idx)": i, 
                            "Vaga (Idx)": int(j), 
                            "Rank": rank,
                            "Similaridade": float(sims[j]),
                            "Score (%)": round(proportional_score(float(sims[j]), limiar), 2),
                            "Aprovado": bool(sims[j] >= limiar)
                        })
                
                out_df = pd.DataFrame(rows)
                st.success(f"🎉 Processado: {len(out_df)} matches no Top K.")
                
                st.dataframe(
                    out_df, 
                    use_container_width=True, 
                    hide_index=True,
                    column_config={
                        "Similaridade": st.column_config.ProgressColumn("Similaridade", format="%.4f", min_value=0.5, max_value=1.0),
                        "Score (%)": st.column_config.ProgressColumn("Score (%)", format="%f", min_value=0, max_value=100),
                        "Aprovado": st.column_config.CheckboxColumn("Aprovado?", disabled=True),
                    }
                )
                
                aprovados = out_df["Aprovado"].sum()
                st.info(f"**Estatísticas:** {aprovados} matches aprovados ({aprovados/len(out_df)*100:.1f}%) no Top {top_k}.")
                
                st.download_button(
                    "💾 Baixar resultados (CSV)", 
                    out_df.to_csv(index=False).encode("utf-8"),
                    file_name="matches_topk_batch.csv", 
                    mime="text/csv",
                    key="dl_batch"
                )

# ############## TAB 3: Explain (Trechos) ##############
with tab3:
    st.subheader("Explain — Top Trechos Mais Similares")
    st.caption("Quebra os textos em sentenças e encontra os pares de sentenças com maior similaridade (ótimo para debugging).")
    
    col_explain1, col_explain2 = st.columns(2)
    with col_explain1: 
        cv_t = st.text_area("CV Completo", height=200, placeholder="Cole o texto completo do CV...", key="cv_explain")
    with col_explain2: 
        vg_t = st.text_area("Vaga Completa", height=200, placeholder="Cole a descrição completa da vaga...", key="vaga_explain")
    
    col_config1, col_config2 = st.columns(2)
    with col_config1:
        top_n = st.number_input("Top N Pares", 1, 10, 3, key="topn_explain")
    with col_config2:
        min_chars = st.number_input("Mín. Caracteres/Sentença", 10, 500, 25, key="minchars_explain")
    
    if st.button("Gerar Explain", key="btn_explain", use_container_width=True):
        if not cv_t or not vg_t:
            st.error("Informe ambos os textos (CV e Vaga).")
        else:
            with st.spinner("Analisando trechos similares..."):
                model = load_model(MODEL_DIR)
                cv_sentences = split_sentences(cv_t, min_chars)
                vaga_sentences = split_sentences(vg_t, min_chars)
                
                if not cv_sentences or not vaga_sentences:
                    st.warning("Não foi possível extrair sentenças suficientes. Tente reduzir o 'Mín. Caracteres/Sentença'.")
                else:
                    st.info(f"CV: {len(cv_sentences)} sentenças. Vaga: {len(vaga_sentences)} sentenças.")
                    pairs = top_n_pairs_by_cosine_optimized(cv_sentences, vaga_sentences, model, int(top_n))
                    
                    if pairs:
                        rows = []
                        for r, (i, j, sim) in enumerate(pairs):
                            rows.append({
                                "Rank": r+1, 
                                "Similaridade": round(float(sim), 6),
                                "CV Snippet": cv_sentences[i][:150] + ("..." if len(cv_sentences[i]) > 150 else ""),
                                "Vaga Snippet": vaga_sentences[j][:150] + ("..." if len(vaga_sentences[j]) > 150 else "")
                            })
                        
                        df_pairs = pd.DataFrame(rows)
                        st.success(f"Top {len(pairs)} pares encontrados!")

                        st.dataframe(
                            df_pairs, 
                            use_container_width=True, 
                            hide_index=True,
                            column_config={
                                "Similaridade": st.column_config.ProgressColumn("Similaridade", format="%.6f", min_value=0.5, max_value=1.0)
                            }
                        )
                        
                        st.subheader("📋 Detalhes dos Trechos (Visualização Completa)")
                        for idx, row in df_pairs.iterrows():
                            with st.expander(f"**Par #{idx+1} | Similaridade: {row['Similaridade']:.4f}**"):
                                col_left, col_right = st.columns(2)
                                with col_left:
                                    st.write("**Trecho do CV:**")
                                    st.code(cv_sentences[int(pairs[idx][0])])
                                with col_right:
                                    st.write("**Trecho da Vaga:**")
                                    st.code(vaga_sentences[int(pairs[idx][1])])
                    else:
                        st.warning("Não foi possível encontrar pares similares.")


# ############## TAB 4: Ranking por Vaga (Base Fixa) ##############
with tab4:
    st.subheader("Ranking por Vaga — Base Fixa/Cache")
    
    vdf = st.session_state["vagas_df"]
    cdf = st.session_state["candidatos_df"]
    
    def _vaga_label(row: pd.Series) -> str:
        title = row.get("titulo") or ""
        vt = str(row.get("vaga_text", ""))
        base_txt = title.strip() or (vt[:80] + ("…" if len(vt) > 80 else ""))
        return f"({row.name}) {base_txt}"
    
    if len(vdf) > 0:
        options = vdf.apply(_vaga_label, axis=1).tolist()
        idx = st.selectbox("Selecione a vaga", options=range(len(options)), format_func=lambda i: options[i])
        
        col_l, col_r = st.columns([1,1])
        with col_l: 
            top_n = st.number_input("Top N Candidatos", 1, len(cdf), min(10, len(cdf)), key="topn_ranking")
        with col_r: 
            clean_rank = st.checkbox("Aplicar limpeza nos textos (desativa o cache)", False, key="clean_ranking")
            st.caption(f"Cache de embeddings: {'✅ Ativo' if st.session_state['cache_loaded'] and not clean_rank else '❌ Inativo/Ignorado'}")

        with st.expander("Ver descrição completa da vaga"):
            vaga_text_sel = str(vdf.iloc[idx]["vaga_text"])
            st.write(vaga_text_sel)

        if st.button("🔍 Gerar Ranking", key="btn_ranking", use_container_width=True):
            with st.spinner("Calculando ranking..."):
                
                # 1. Embedding da Vaga
                if st.session_state["cache_loaded"] and not clean_rank:
                    emb_vaga = st.session_state["vaga_embs_cache"][idx]
                elif clean_rank:
                    emb_vaga = embed_text(clean_text(vaga_text_sel), MODEL_DIR).flatten()
                else: 
                    emb_vaga = embed_text(vaga_text_sel, MODEL_DIR).flatten()

                # 2. Embeddings dos Candidatos
                if st.session_state["cache_loaded"] and not clean_rank:
                    emb_cvs = st.session_state["cand_embs_cache"]
                else: 
                    cvs = cdf["cv_text"].astype(str).tolist()
                    if clean_rank: 
                        cvs = [clean_text(x) for x in cvs]
                    emb_cvs = embed_texts(cvs, MODEL_DIR)

                # 3. Cálculo de similaridade e ordenação
                sims = emb_cvs @ emb_vaga.T
                order = np.argsort(-sims)[:int(top_n)]
                
                rows = []
                for rank, i in enumerate(order, start=1):
                    sim = float(sims[i])
                    score = proportional_score(sim, limiar)
                    row = {
                        "Rank": rank, 
                        "Similaridade": round(sim, 6),
                        "Score (%)": round(score, 2), 
                        "Aprovado": bool(sim >= limiar),
                    }
                    
                    for col in ["nome", "id", "titulo"]:
                        if col in cdf.columns: 
                            row[col] = cdf.iloc[i][col]
                    
                    rows.append(row)
                
                res = pd.DataFrame(rows)
                
                st.success(f"🎉 Ranking gerado: Top {len(res)} candidatos.")

                st.dataframe(
                    res, 
                    use_container_width=True, 
                    hide_index=True,
                    column_config={
                        "Similaridade": st.column_config.ProgressColumn("Similaridade", format="%.4f", min_value=0.5, max_value=1.0),
                        "Score (%)": st.column_config.ProgressColumn("Score (%)", format="%f", min_value=0, max_value=100),
                        "Aprovado": st.column_config.CheckboxColumn("Aprovado?", disabled=True),
                    }
                )
                
                aprovados = res["Aprovado"].sum()
                col1, col2, col3 = st.columns(3)
                with col1: st.metric("Candidatos Aprovados", aprovados)
                with col2: st.metric("Taxa de Aprovação", f"{aprovados/len(res)*100:.1f}%")
                with col3: st.metric("Melhor Similaridade", f"{res.iloc[0]['Similaridade']:.4f}")
                
                st.download_button(
                    "💾 Baixar Ranking (CSV)", 
                    res.to_csv(index=False).encode("utf-8"),
                    file_name="ranking_por_vaga.csv", 
                    mime="text/csv",
                    key="dl_ranking"
                )
    else:
        st.warning("Nenhuma vaga disponível na base de dados carregada para ranking.")


# ############## TAB 5: Bases de Dados ##############
with tab5:
    st.subheader("Bases de Dados Carregadas em Memória")

    def _preview_df(df: pd.DataFrame, text_cols: list[str], max_chars: int = 400, max_cols: int = 20) -> pd.DataFrame:
        df = df.copy()
        
        view_cols = [c for c in df.columns if c not in text_cols] + text_cols
        view_cols = view_cols[:max_cols] if view_cols else df.columns[:max_cols]
        dfv = df[view_cols].copy()

        for c in dfv.columns:
            if c in text_cols and dfv[c].dtype == object:
                dfv[c] = dfv[c].str.slice(0, max_chars) + (dfv[c].apply(lambda x: '...' if isinstance(x, str) and len(x) > max_chars else ''))
        return dfv

    # ---- Candidatos ----
    st.markdown("#### Candidatos (Preview)")
    cand_full = st.session_state["candidatos_df"]
    cand_prev = _preview_df(cand_full, text_cols=["cv_text"], max_chars=300, max_cols=10)
    st.data_editor(cand_prev, use_container_width=True, hide_index=True)
    
    col_dl_c, col_met_c = st.columns([1,2])
    with col_dl_c:
        st.download_button(
            "💾 Baixar CSV Candidatos (Completo)",
            data=cand_full.to_csv(index=False).encode("utf-8"),
            file_name="candidatos_completo.csv",
            mime="text/csv",
            key="dl_candidatos"
        )
    with col_met_c:
        st.metric("Total Candidatos", len(cand_full))

    st.divider()

    # ---- Vagas ----
    st.markdown("#### Vagas (Preview)")
    vagas_full = st.session_state["vagas_df"]
    vagas_prev = _preview_df(vagas_full, text_cols=["vaga_text"], max_chars=300, max_cols=10)
    st.data_editor(vagas_prev, use_container_width=True, hide_index=True)
    
    col_dl_v, col_met_v = st.columns([1,2])
    with col_dl_v:
        st.download_button(
            "💾 Baixar CSV Vagas (Completo)",
            data=vagas_full.to_csv(index=False).encode("utf-8"),
            file_name="vagas_completo.csv",
            mime="text/csv",
            key="dl_vagas"
        )
    with col_met_v:
        st.metric("Total Vagas", len(vagas_full))

# Footer
st.sidebar.divider()
st.sidebar.caption(f"""
    Desenvolvido para Especialização.
    **Modelo (SBERT):** Carregado da pasta **{MODEL_DIR}** no seu repositório.
    **Dados (CSV):** Prioriza URL (Variáveis de Ambiente) > Disco Local > Amostra.
""")