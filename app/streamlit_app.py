# streamlit_app.py — Código Final Especializado e Corrigido (v1.3.3 - Log Limpo)
import os, re, json, hashlib, io 
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import requests 
from scipy.spatial.distance import cosine # Importação não utilizada, mas mantida por consistência

# ======================== CONFIG ========================
APP_NAME = "RECRUT.AI 🚀"
APP_VERSION = "1.3.3 (Log Limpo)" # Versão atualizada

# Limiar padrão para "Aprovação" no ranking
DEFAULT_LIMIAR = float(os.getenv("SCORE_LIMIAR", "0.75"))

# Modelo: Pasta local do seu projeto (O seu sbert_encoder)
MODEL_DIR = os.getenv("MODEL_DIR", "models/sbert_encoder")
MODEL_NAME = os.getenv("MODEL_NAME", f"Local_Custom:{MODEL_DIR}") 
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") 

# Dados CSV (Local e URLs)
BASE_CANDIDATOS_PATH = os.getenv("BASE_CANDIDATOS_PATH", "data/applicants_clean.csv")
BASE_VAGAS_PATH = os.getenv("BASE_VAGAS_PATH", "data/vagas_clean.csv")
# Estas variáveis de ambiente devem ser configuradas para usar as URLs RAW do GitHub, por exemplo:
# CANDIDATOS_CSV_URL = "https://raw.githubusercontent.com/janbar/Datathon/main/data/applicants_clean.csv"
CANDIDATOS_CSV_URL = os.getenv("CANDIDATOS_CSV_URL", "https://raw.githubusercontent.com/janbar/Datathon/main/data/applicants_clean.csv") 
VAGAS_CSV_URL = os.getenv("VAGAS_CSV_URL", "https://raw.githubusercontent.com/janbar/Datathon/main/data/vagas_clean.csv") 

# Cache de embeddings: Remoto (URLs RAW do GitHub - PRIORIDADE MÁXIMA)
CAND_EMB_URL = os.getenv("CAND_EMB_URL", "https://raw.githubusercontent.com/janbar/Datathon/main/data/embeddings/candidatos.npy") 
CAND_META_URL = os.getenv("CAND_META_URL", "https://raw.githubusercontent.com/janbar/Datathon/main/data/embeddings/candidatos.meta.json")
VAGA_EMB_URL = os.getenv("VAGA_EMB_URL", "https://raw.githubusercontent.com/janbar/Datathon/main/data/embeddings/vagas.npy") 
VAGA_META_URL = os.getenv("VAGA_META_URL", "https://raw.githubusercontent.com/janbar/Datathon/main/data/embeddings/vagas.meta.json") 

# Cache de embeddings: Local (Fallback)
EMB_DIR = Path(os.getenv("EMB_DIR", "data/embeddings")); EMB_DIR.mkdir(parents=True, exist_ok=True)
CAND_EMB_PATH = EMB_DIR / "candidatos.npy"
CAND_META_PATH = EMB_DIR / "candidatos.meta.json"
VAGA_EMB_PATH = EMB_DIR / "vagas.npy"
VAGA_META_PATH = EMB_DIR / "vagas.meta.json"

# Colunas de metadados dos candidatos a exibir no ranking (AJUSTE CONFORME SEU CSV)
CAND_META_COLS = ["nome", "cidade", "experiencia", "skills", "id"] 

# ======================== TEXT UTILS ========================
_whitespace_re = re.compile(r"\s+")

def clean_text(t: str) -> str:
    """Limpa e normaliza texto (minúsculas e remoção de espaços extras)."""
    if t is None: return ""
    t = t.strip().lower()
    return _whitespace_re.sub(" ", t)

def proportional_score(sim: float, limiar: float) -> float:
    """Calcula o score de 0 a 100 baseado no limiar de aprovação."""
    # Garante que o score não seja negativo e escala até 100
    if limiar <= 0: return 0.0 # Evita divisão por zero
    if sim >= limiar: return 100.0
    return max(0.0, (sim / limiar) * 100.0)

# ======================== DATA LOADERS (MOVIDO PARA CIMA PARA CORRIGIR ERRO) ========================

@st.cache_data(show_spinner=False)
def _concat_all_columns(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Concatena todas as colunas de texto em uma única coluna para o SBERT."""
    out = df.copy()
    if target_col in out.columns: out = out.drop(columns=[target_col])
    # Tenta excluir colunas que são provavelmente IDs/índices
    cols_to_concat = [c for c in out.columns if c.lower() not in ["id", "index", "uid", "score", "rank"]]
    s = out[cols_to_concat].fillna("").astype(str)
    # Concatena os valores não vazios separados por espaço
    out[target_col] = s.apply(lambda row: " ".join([v for v in row.values if v != ""]).strip(), axis=1)
    return out

@st.cache_data(show_spinner=False)
def _read_csv_local_or_url(local_path: str, url_env: str | None) -> pd.DataFrame | None:
    """Carrega CSV da URL (Prioridade) ou do disco local (Fallback)."""
    # 1. Tenta URL (Prioridade - Para deploy Cloud)
    if url_env:
        try:
            r = requests.head(url_env, timeout=5)
            if r.status_code == 200:
                df = pd.read_csv(url_env)
                st.info(f"🔗 Dados lidos da URL: {url_env}")
                return df
        except Exception: 
            pass
            
    # 2. Tenta caminho local (Fallback - Para teste local)
    try:
        if os.path.exists(local_path): 
            df = pd.read_csv(local_path)
            st.info(f"📂 Dados lidos do disco local: {local_path}")
            return df
    except Exception: 
        pass
        
    return None
    
@st.cache_data(show_spinner="Carregando bases de dados...", ttl=None)
def load_fixed_bases() -> Tuple[pd.DataFrame, pd.DataFrame, list]:
    """Carrega as bases de candidatos e vagas e concatena colunas de texto."""
    logs = []
    
    # Chama a função que agora está definida!
    cand = _read_csv_local_or_url(BASE_CANDIDATOS_PATH, CANDIDATOS_CSV_URL) 
    vaga = _read_csv_local_or_url(BASE_VAGAS_PATH, VAGAS_CSV_URL)

    # Fallback para dados de Amostra (se nenhum arquivo for encontrado)
    if cand is None or cand.empty:
        logs.append("⚠️ Não encontrei candidatos via URL ou local. Usando amostra.")
        cand = pd.DataFrame({
            "nome": ["Ana Silva", "Carlos Souza"],
            "skills": ["Python Airflow Spark", "Java Spring SQL"],
            "experiencia": ["3 anos em dados", "5 anos em backend"],
            "cidade": ["São Paulo", "Rio de Janeiro"],
            "id": [1, 2]
        })
    if vaga is None or vaga.empty:
        logs.append("⚠️ Não encontrei vagas via URL ou local. Usando amostra.")
        vaga = pd.DataFrame({
            "titulo": ["Engenheira de Dados Senior", "Desenvolvedor Backend Java"],
            "requisitos": ["Python, Spark, Airflow, AWS", "Java, Spring Boot, SQL, REST APIs"],
            "descricao": ["Projetos de dados em ambiente cloud. Criação de pipelines ETL.", "Desenvolvimento de microserviços de alta performance."]
        })
        
    # Concatena colunas de texto para o SBERT
    cand = _concat_all_columns(cand, "cv_text")
    vaga = _concat_all_columns(vaga, "vaga_text")
    
    logs.append(f"✅ Bases carregadas: {len(cand)} candidatos e {len(vaga)} vagas.")
    
    return cand, vaga, logs

# ======================== EMBEDDING CACHE UTILS ========================
# NOTA: O restante das funções de utilidade (cache, modelo) é idêntico ao que você já tinha,
# mas foram movidas para baixo de 'DATA LOADERS' para garantir que tudo o que for chamado na 
# inicialização (main) já esteja definido.

@st.cache_data(show_spinner=False)
def _hash_dataframe(df: pd.DataFrame) -> str:
    """Gera um hash para o conteúdo de um DataFrame (usado para checagem de cache)."""
    buf = df.to_csv(index=False).encode("utf-8")
    return hashlib.md5(buf).hexdigest()

def _save_embeddings(npy_path: Path, meta_path: Path, embs: np.ndarray, meta: dict) -> None:
    """Tenta salvar embeddings e metadados no disco local."""
    try:
        np.save(npy_path, embs)
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        st.success(f"💾 Embeddings salvos no disco: {npy_path.name}")
    except Exception as e:
        # Silencia a falha em ambientes efêmeros (e.g., Streamlit Cloud)
        st.warning(f"⚠️ Falha ao salvar cache no disco ({npy_path.name}): {e}") 

def _load_embeddings_url(npy_url: str | None, meta_url: str | None) -> Tuple[np.ndarray | None, dict]:
    """Tenta carregar embeddings via URL RAW do GitHub (Maior Prioridade)."""
    if not npy_url or not meta_url: return None, {}
    try:
        # 1. Carrega o arquivo .meta.json
        meta_response = requests.get(meta_url, timeout=5)
        if meta_response.status_code != 200: return None, {}
        meta = json.loads(meta_response.text)
        
        # 2. Carrega o arquivo .npy (binário)
        npy_response = requests.get(npy_url, timeout=20)
        if npy_response.status_code != 200: return None, {}
        
        # Lê o array NumPy a partir dos bytes
        embs = np.load(io.BytesIO(npy_response.content))
        st.info(f"⚡ Cache de embeddings carregado via URL: {npy_url}")
        return embs, meta
    except Exception as e:
        return None, {}

def _load_embeddings_local(npy_path: Path, meta_path: Path) -> Tuple[np.ndarray | None, dict]:
    """Tenta carregar embeddings do disco local (Fallback)."""
    if not npy_path.exists() or not meta_path.exists():
        return None, {}
    try:
        arr = np.load(npy_path); meta = json.loads(meta_path.read_text(encoding="utf-8"))
        st.info(f"💾 Cache de embeddings carregado do disco: {npy_path.name}")
        return arr, meta
    except Exception:
        return None, {}

def get_or_build_embeddings(df: pd.DataFrame, text_col: str, model_dir: str) -> np.ndarray:
    """Carrega do cache (URL > Local) ou constrói (lento) os embeddings."""
    
    # Determina os caminhos e URLs
    if text_col == "cv_text":
        npy_path, meta_path, npy_url, meta_url = CAND_EMB_PATH, CAND_META_PATH, CAND_EMB_URL, CAND_META_URL
    else:
        npy_path, meta_path, npy_url, meta_url = VAGA_EMB_PATH, VAGA_META_PATH, VAGA_EMB_URL, VAGA_META_URL

    # Gera a assinatura e metadados esperados
    sig = _hash_dataframe(df[[text_col]])
    meta_expected = {"model_source": model_dir, "text_col": text_col, "signature": sig, "version": APP_VERSION}
    
    # Validação de cache (função auxiliar)
    def _is_valid(embs: np.ndarray | None, meta: dict, df: pd.DataFrame) -> bool:
        """Verifica se o cache é válido, comparando com o DataFrame atual, o modelo e a versão do app."""
        # Se FORCE_REBUILD estiver no model_dir, desativa o cache
        if model_dir == "FORCE_REBUILD":
             return False 
             
        # Checa se o conteúdo do meta.json e o tamanho do NPY batem
        return (
            embs is not None and 
            meta == meta_expected and 
            embs.ndim == 2 and 
            embs.shape[0] == len(df)
        )

    # 1. Tenta carregar do cache remoto (URL) - PRIORIDADE MÁXIMA
    embs, meta = _load_embeddings_url(npy_url, meta_url)
    if _is_valid(embs, meta, df): return embs

    # 2. Tenta carregar do cache local
    embs, meta = _load_embeddings_local(npy_path, meta_path)
    if _is_valid(embs, meta, df): return embs
        
    # 3. Se falhar ou não existir, constrói (Lento)
    st.warning(f"⏳ Reconstruindo embeddings para '{text_col}' (Não há cache válido). Isso pode levar alguns minutos...")
    texts = df[text_col].astype(str).tolist()
    
    # Garante que load_model seja chamada e que o modelo esteja pronto
    encoder_model = load_model(model_dir) 
    embs = embed_texts(texts, model_dir) 
    
    _save_embeddings(npy_path, meta_path, embs, meta_expected) # Tenta salvar localmente para próxima vez
    st.success(f"✅ Embeddings gerados para '{text_col}'.")
    return embs

# ======================== MODEL / ENCODER ========================
@st.cache_resource(show_spinner="Carregando Encoder (Priorizando Modelo Local)...", ttl=None)
def load_model(model_path: str):
    """Carrega o SentenceTransformer, priorizando o modelo local."""
    # Importação dentro da função para evitar erro se a lib não estiver instalada
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        st.error("❌ Falha crítica: A biblioteca 'sentence-transformers' não está instalada.")
        st.stop()
        
    try:
        model = SentenceTransformer(model_path) 
        return model
    except Exception as e:
        # Tenta fallback para o modelo Hugging Face
        try:
            model = SentenceTransformer(HF_MODEL_NAME)
            st.warning(f"⚠️ Modelo padrão do Hugging Face carregado como fallback.")
            return model
        except Exception:
            st.error("❌ Falha crítica: Não foi possível carregar o modelo de nenhuma fonte.")
            st.stop()
            

@st.cache_data(show_spinner="Gerando embeddings em lote...", ttl=3600, max_entries=5)
def embed_texts(texts: List[str], model_path: str) -> np.ndarray:
    """Gera embeddings para uma lista de textos."""
    model = load_model(model_path)
    # Aplica a limpeza antes de embedar
    texts = [clean_text(t) for t in texts] 
    # **IMPORTANTE**: Normaliza os embeddings para garantir que o produto escalar seja a similaridade do cosseno.
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)

@st.cache_data(show_spinner="Gerando embedding...", ttl=3600, max_entries=20)
def embed_text(text: str, model_path: str) -> np.ndarray:
    """Gera o embedding para um único texto, garantindo que o resultado seja 1D."""
    model = load_model(model_path)
    text_clean = clean_text(text)
    # **IMPORTANTE**: Normaliza os embeddings
    emb = model.encode(text_clean, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
    return emb.flatten()

# ======================== LÓGICA DE INICIALIZAÇÃO ========================

# 1. Configurações da Página 
st.set_page_config(
    page_title=APP_NAME, 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# 2. Carrega o modelo (e exibe o log de sucesso apenas UMA VEZ)
# Nota: load_model é @st.cache_resource e será executada apenas uma vez.
with st.spinner("Carregando modelo Sentence-BERT..."):
    load_model(MODEL_DIR)
    st.success(f"✅ Modelo customizado carregado com sucesso de: **{MODEL_DIR}**")

# 3. Carrega Bases Fixas (e concatena o texto)
candidatos_df, vagas_df, _logs = load_fixed_bases()

# Guarda bases na session_state
st.session_state["candidatos_df"] = candidatos_df.copy()
st.session_state["vagas_df"] = vagas_df.copy()

# 4. Tenta carregar embeddings de base OU recalcula
# Executado apenas na primeira vez
if "cache_loaded" not in st.session_state:
    st.session_state["cache_loaded"] = False # Flag inicial
    
    # Exibe logs de carregamento de base antes da lentidão do embedding
    if _logs:
        with st.expander("Logs de Carregamento de Bases", expanded=False):
            for m in _logs: st.caption(m)
            
    with st.spinner("⚡ Preparando e carregando embeddings iniciais (Cache ou Reconstrução)..."):
        try:
            # Tenta carregar ou construir os embeddings dos candidatos
            cand_embs_cache = get_or_build_embeddings(st.session_state["candidatos_df"], "cv_text", MODEL_DIR)
            # Tenta carregar ou construir os embeddings das vagas
            vaga_embs_cache = get_or_build_embeddings(st.session_state["vagas_df"], "vaga_text", MODEL_DIR)
            
            # Se ambos foram carregados/construídos com sucesso
            if cand_embs_cache is not None and vaga_embs_cache is not None:
                st.session_state["cache_loaded"] = True
                st.session_state["cand_embs_cache"] = cand_embs_cache
                st.session_state["vaga_embs_cache"] = vaga_embs_cache
            else:
                st.error("Não foi possível obter os embeddings para as bases fixas.")

        except Exception as e:
            st.error(f"Não foi possível carregar ou gerar os embeddings iniciais: {e}")
            st.session_state["cache_loaded"] = False
            st.session_state["cand_embs_cache"] = None
            st.session_state["vaga_embs_cache"] = None

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
    cache_loaded = st.session_state["cache_loaded"]
    st.info(f"Status do Cache: {'✅ Disponível' if cache_loaded else '❌ Indisponível'}")

    if st.button("⚡ Gerar/Atualizar Cache de Base", help="Força a reconstrução e tentativa de salvamento do cache local."):
        with st.spinner("Gerando e salvando cache (candidatos e vagas)…"):
            # Força a reconstrução (ignora caches existentes)
            # A função get_or_build_embeddings verifica o argumento "FORCE_REBUILD" e invalida o cache
            st.session_state["cand_embs_cache"] = get_or_build_embeddings(st.session_state["candidatos_df"], "cv_text", "FORCE_REBUILD")
            st.session_state["vaga_embs_cache"] = get_or_build_embeddings(st.session_state["vagas_df"], "vaga_text", "FORCE_REBUILD")
            
            # Força a atualização da flag de cache
            st.session_state["cache_loaded"] = True
            
            st.success(f"Cache gerado. Recarregue a página (F5) para usar o novo cache na inicialização.")

# Exibe logs de carregamento de base
# Nota: Os logs de carregamento de base foram movidos para a área de inicialização (Passo 4)
# para aparecerem antes dos warnings de reconstrução de embedding.

st.title("🔎 RECRUT.AI - Match Semântico (Especialização)")
st.markdown(f"Análise de similaridade entre Curricula e Vagas usando **Sentence-BERT** (Modelo: **{MODEL_DIR}**)")

# -------------------- TABS (Apenas Ranking e Bases) --------------------
tab_ranking, tab_bases = st.tabs(["📊 Ranking por Vaga", "📋 Bases de Dados"])

# ############## TAB 1: Ranking por Vaga (Base Fixa) ##############
with tab_ranking:
    st.header("Ranking de Candidatos por Vaga (N×1)")
    st.caption("Compara todos os candidatos da base fixa contra a vaga selecionada, utilizando embeddings.")
    
    vdf = st.session_state["vagas_df"]
    cdf = st.session_state["candidatos_df"]
    
    def _vaga_label(row: pd.Series) -> str:
        """Formata o rótulo da vaga para o selectbox."""
        title = row.get("titulo") or ""
        vt = str(row.get("vaga_text", ""))
        base_txt = title.strip() or (vt[:80] + ("…" if len(vt) > 80 else ""))
        return f"({row.name}) {base_txt}"
    
    if len(vdf) > 0 and len(cdf) > 0:
        col_sel, col_limpeza = st.columns([3, 1])
        with col_sel:
            options = vdf.apply(_vaga_label, axis=1).tolist()
            idx = st.selectbox("Selecione a vaga para ranquear", options=range(len(options)), format_func=lambda i: options[i], key="sel_vaga_ranking")
        
        vaga_text_sel = str(vdf.iloc[idx]["vaga_text"])

        with col_limpeza:
            # O clean_rank é usado para forçar o recálculo/aplicar clean_text no momento do ranking
            clean_rank = st.checkbox("Aplicar limpeza nos textos (Recalcula embeddings)", not cache_loaded, key="clean_ranking")
            # Mensagem de status do cache dentro do contexto da limpeza
            is_using_cache = cache_loaded and not clean_rank
            st.caption(f"Cache de Embeddings: {'✅ Ativo (Base)' if is_using_cache else '❌ Inativo (Recálculo)'}")

        with st.expander("Ver descrição completa da vaga"):
            st.write(vaga_text_sel)
        
        # --- Controles de Ranking ---
        col_controles = st.columns([1, 4])
        with col_controles[0]:
            top_n = st.number_input("Top N Candidatos", 1, len(cdf), min(10, len(cdf)), key="topn_ranking")
        with col_controles[1]:
            if st.button("🔍 GERAR RANKING", key="btn_ranking", use_container_width=True):
                with st.spinner("Calculando ranking..."):
                    
                    # 1. Embeddings: Prioriza cache, mas recalcula se a limpeza for ativada
                    
                    # Embeddings dos Candidatos (N)
                    if is_using_cache:
                        emb_cvs = st.session_state["cand_embs_cache"]
                    else: 
                        cvs = cdf["cv_text"].astype(str).tolist()
                        emb_cvs = embed_texts(cvs, MODEL_DIR) # embed_texts já aplica clean_text

                    # Embedding da Vaga (1)
                    if is_using_cache:
                        # Seleciona o embedding pré-calculado da vaga
                        emb_vaga = st.session_state["vaga_embs_cache"][idx]
                    else: 
                        # Recalcula o embedding da vaga (já é normalizado e flatten)
                        emb_vaga = embed_text(vaga_text_sel, MODEL_DIR) 

                    # 2. Cálculo de similaridade (Produto escalar, pois ambos estão normalizados)
                    # O produto escalar entre vetores unitários é a similaridade do cosseno
                    sims = emb_cvs @ emb_vaga.T
                    
                    # Ordenação e Top N
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
                        
                        # Adiciona metadados do candidato para exibição
                        for col in CAND_META_COLS:
                            if col in cdf.columns: 
                                val = cdf.iloc[i][col]
                                if col in ["experiencia", "skills"] and isinstance(val, str):
                                    # Limita o texto para visualização no dataframe
                                    row[col] = val[:100] + "..." if len(val) > 100 else val
                                else:
                                    row[col] = val
                        
                        rows.append(row)
                    
                    res = pd.DataFrame(rows)
                    
                    st.success(f"🎉 Ranking gerado: Top {len(res)} candidatos.")

                    # --- Visualização de Resultados ---
                    col_config = {
                        # A similaridade máxima é 1.0 (vetores idênticos e normalizados)
                        "Similaridade": st.column_config.ProgressColumn("Similaridade", format="%.4f", min_value=0.0, max_value=1.0),
                        "Score (%)": st.column_config.ProgressColumn("Score (%)", format="%f", min_value=0, max_value=100),
                        "Aprovado": st.column_config.CheckboxColumn("Aprovado?", disabled=True),
                        "nome": "Nome",
                        "cidade": "Cidade",
                        "experiencia": "Experiência (Resumo)",
                        "skills": "Skills (Resumo)",
                    }
                    
                    st.dataframe(res, use_container_width=True, hide_index=True, column_config=col_config)
                    
                    # Métricas de Resumo
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
        st.warning("Nenhuma vaga ou candidato disponível na base de dados carregada para ranking.")


# ############## TAB 2: Bases de Dados ##############
with tab_bases:
    st.header("Visualização das Bases de Dados Carregadas")

    def _preview_df(df: pd.DataFrame, text_cols: list[str], max_chars: int = 300) -> pd.DataFrame:
        """Prepara um dataframe para visualização, truncando colunas de texto longas."""
        df = df.copy()
        view_cols = [c for c in df.columns if c not in text_cols] + text_cols
        dfv = df[view_cols].copy()
        for c in dfv.columns:
            if c in text_cols and dfv[c].dtype == object:
                dfv[c] = dfv[c].str.slice(0, max_chars) + (dfv[c].apply(lambda x: '...' if isinstance(x, str) and len(x) > max_chars else ''))
        return dfv

    # ---- Candidatos ----
    st.subheader("Candidatos")
    cand_full = st.session_state["candidatos_df"]
    st.metric("Total de Candidatos", len(cand_full))
    st.caption(f"Colunas concatenadas para match: **cv_text**")
    
    col_dl_c, col_preview_c = st.columns([1, 3])
    with col_dl_c:
        st.download_button("💾 Baixar CSV (Completo)", data=cand_full.to_csv(index=False).encode("utf-8"), file_name="candidatos_completo.csv", mime="text/csv", key="dl_candidatos")
    with col_preview_c:
        st.dataframe(_preview_df(cand_full, text_cols=["cv_text"]), use_container_width=True, hide_index=True)

    st.divider()

    # ---- Vagas ----
    st.subheader("Vagas")
    vagas_full = st.session_state["vagas_df"]
    st.metric("Total de Vagas", len(vagas_full))
    st.caption(f"Colunas concatenadas para match: **vaga_text**")
    
    col_dl_v, col_preview_v = st.columns([1, 3])
    with col_dl_v:
        st.download_button("💾 Baixar CSV (Completo)", data=vagas_full.to_csv(index=False).encode("utf-8"), file_name="vagas_completo.csv", mime="text/csv", key="dl_vagas")
    with col_preview_v:
        st.dataframe(_preview_df(vagas_full, text_cols=["vaga_text"]), use_container_width=True, hide_index=True)

# Footer
st.sidebar.divider()
st.sidebar.caption(f"""
    Desenvolvido para Especialização.
    Modelo (SBERT): Carregado da pasta **{MODEL_DIR}**.
    Embeddings: Prioriza URL RAW > Disco Local.
""")