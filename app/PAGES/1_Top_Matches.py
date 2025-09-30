# pages/1_Top_Matches.py

# ==============================================================================
# 1. IMPORTS E CONFIGURAÇÕES INICIAIS (COPIADOS)
# ==============================================================================
import os
import io
import re
import time
import json
import shutil
import tempfile
import logging
import html
from datetime import datetime
from typing import Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
import s3fs
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings("ignore")

# Configuração da página Streamlit (IMPORTANTE!)
st.set_page_config(
    page_title="🥇 Top Matches", # Título ÚNICO para esta página
    layout="wide",
    initial_sidebar_state="expanded",
)

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================================================
# 2. VARIÁVEIS DE CONFIGURAÇÃO (COPIADAS)
# ==============================================================================
# ... (Copie todas as variáveis de configuração, caminhos S3, nomes de arquivos, e colunas do seu código original) ...
S3_BUCKET = "datathon-recrutai"
SBERT_MODEL_DIR = "sbert_encoder"
CANDIDATOS_FILE = "applicants_clean.csv"
VAGAS_FILE = "vagas_clean.csv"
EMBEDDINGS_FILE = "candidatos.npy"
VAGAS_EMBEDDINGS_FILE = "vagas.npy"
MODEL_FILE = "modelo_match_xgboost.pkl"
CV_TEXT_COL = "curriculo_pt"
VAGA_ID_COL = "id"
CANDIDATO_ID_COL = "id"
VAGA_TEXT_COL = "vaga_text"
CANDIDATO_METADATA_COLS = [
    "status", "nivel_hierarquico", "genero", "salario_atual", "cidade", "estado", 
    "pais", "idade", "escolaridade", "area_atuacao", "tempo_experiencia", 
    "ultima_experiencia", "email", "linkedin",
]
DEFAULT_SKILLS = [
    "python","java","javascript","typescript","c#","c++","sql","nosql","aws","gcp","azure",
    "docker","kubernetes","linux","git","rest","graphql","spark","hadoop","airflow","terraform",
    "react","vue","angular","node","django","flask",".net","spring","xgboost","pytorch","tensorflow",
    "etl","elt","modelagem de dados","ci/cd","devops","mlops","nlp","cv","agile","scrum","kanban"
]

# --- (Copie o código de helpers de explicabilidade, _norm, extract_skills, split_sentences, top_relevant_sentences e render_badges) ---
def _norm(text: str) -> str:
    return (text or "").lower()

def extract_skills(vaga_text: str, extra_skills: Optional[List[str]] = None) -> List[str]:
    """Extrai uma lista de skills-alvo a partir do texto da vaga (set simples)."""
    base = set(DEFAULT_SKILLS + (extra_skills or []))
    vt = _norm(vaga_text)
    found = [s for s in base if s in vt]
    return found[:20] if found else DEFAULT_SKILLS[:10]

def split_sentences(text: str) -> List[str]:
    """Split simples e robusto (sem novas deps) para sentenças do CV."""
    if not isinstance(text, str) or not text.strip():
        return []
    parts = re.split(r'(?<=[\.\!\?])\s+|\n+', text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 20][:60]

@st.cache_data(hash_funcs={SentenceTransformer: lambda _: None})
def top_relevant_sentences(
    candidate_id: Any, vaga_id: Any, cv_text: str, vaga_embedding: np.ndarray, 
    encoder: SentenceTransformer, k: int = 3
) -> List[str]:
    sents = split_sentences(cv_text)
    if not sents: return []
    # Usando o cache global, se disponível
    sent_emb = encoder.encode(sents, show_progress_bar=False, convert_to_numpy=True, batch_size=32).astype("float32")
    sent_emb = _l2_normalize(sent_emb)
    vaga_emb = vaga_embedding.astype("float32")
    scores = (sent_emb @ vaga_emb).reshape(-1)
    k = min(k, len(sents))
    top_idx = np.argpartition(scores, -k)[-k:]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return [sents[i] for i in top_idx]

def render_badges(items: list) -> str:
    return " ".join([f"<span style='display:inline-block;padding:2px 8px;border-radius:999px;background:#1f6feb20;border:1px solid #1f6feb55;margin:2px 6px 2px 0;font-size:12px'>{html.escape(i)}</span>" for i in items[:12]])
# --- 🎯 INJEÇÃO DE SECRETS DO STREAMLIT CLOUD (COPIADO) ---
if "aws" in st.secrets:
    try:
        access_key = st.secrets["aws"].get("AWS_ACCESS_KEY_ID") or st.secrets["aws"]["aws_access_key_id"]
        secret_key = st.secrets["aws"].get("AWS_SECRET_ACCESS_KEY") or st.secrets["aws"]["aws_secret_access_key"]
        aws_region = st.secrets["aws"].get("AWS_REGION") or st.secrets["aws"].get("AWS_DEFAULT_REGION") or st.secrets["aws"].get("region_name")

        os.environ["AWS_ACCESS_KEY_ID"] = access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = secret_key
        os.environ["AWS_DEFAULT_REGION"] = aws_region
    except KeyError:
        pass # Deixa o erro ser tratado pelo load_data/get_s3_fs

# ==============================================================================
# 3. FUNÇÕES UTILITÁRIAS DE HASH/NORMALIZAÇÃO E CACHE/LOAD (COPIADAS)
# ==============================================================================
# **COPIE AS FUNÇÕES ABAIXO DO SEU CÓDIGO ORIGINAL**
# _hash_df
# _l2_normalize
# get_s3_fs
# load_models
# load_encoder
# load_data
# get_or_create_embeddings
# (Certifique-se de que todas as funções sejam copiadas para que a nova página seja autossuficiente)

# Exemplo de cópia (para fins de demonstração, você precisa do corpo COMPLETO do seu código original):
def _hash_df(df: pd.DataFrame, cols: List[str], sample_rows: int = 0) -> str: 
    # COPIE O CORPO COMPLETO DA FUNÇÃO _hash_df
    if not set(cols).issubset(df.columns): return f"missing_cols_{hash(tuple(cols))}"
    if sample_rows and len(df) > sample_rows: df = df.sample(sample_rows, random_state=42)
    s = pd.util.hash_pandas_object(df[cols].astype(str), index=False).values
    return str(int(s[: min(2000, len(s))].sum())) if len(s) else "empty"

def _l2_normalize(M: np.ndarray) -> np.ndarray: 
    # COPIE O CORPO COMPLETO DA FUNÇÃO _l2_normalize
    n = np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    return M / n

@st.cache_resource(show_spinner=False)
def get_s3_fs(): 
    # COPIE O CORPO COMPLETO DA FUNÇÃO get_s3_fs
    try:
        fs = s3fs.S3FileSystem(anon=False)
        fs.ls(S3_BUCKET)
        return fs
    except Exception as e:
        raise RuntimeError(f"Erro de conexão com S3: {e}. Verifique as credenciais AWS.")

@st.cache_resource(show_spinner="Carregando modelo XGBoost do S3...")
def load_models() -> Any: 
    # COPIE O CORPO COMPLETO DA FUNÇÃO load_models
    fs = get_s3_fs()
    model_s3_path = f"{S3_BUCKET}/data/models/{MODEL_FILE}"
    if not fs.exists(model_s3_path): raise FileNotFoundError(f"Arquivo do modelo XGBoost não encontrado: s3://{model_s3_path}")
    with fs.open(model_s3_path, "rb") as f: bst = joblib.load(f)
    if bst is None: raise ValueError("Modelo XGBoost está vazio")
    return bst

@st.cache_resource(show_spinner="Carregando Sentence Transformer...")
def load_encoder(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    # COPIE O CORPO COMPLETO DA FUNÇÃO load_encoder
    temp_dir = None
    try:
        fs = get_s3_fs()
        sbert_s3_path = f"{S3_BUCKET}/{SBERT_MODEL_DIR}"
        test_file_path = f"{sbert_s3_path}/config.json"
        if not fs.exists(test_file_path): 
            encoder = SentenceTransformer(model_name)
            _ = encoder.encode(["probe"], convert_to_numpy=True)
            return encoder
        temp_dir = tempfile.mkdtemp()
        local_model_path = os.path.join(temp_dir, SBERT_MODEL_DIR)
        fs.get(sbert_s3_path, local_model_path, recursive=True)
        encoder = SentenceTransformer(local_model_path)
        _ = encoder.encode(["probe"], convert_to_numpy=True)
        return encoder
    except Exception as e: raise RuntimeError(f"Falha crítica ao carregar SBERT: {e}")
    finally:
        if temp_dir and os.path.exists(temp_dir): 
            try: shutil.rmtree(temp_dir)
            except: pass

@st.cache_data(show_spinner="Carregando dados dos candidatos e vagas do S3...", ttl=900)
def load_data(_max_rows: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    # COPIE O CORPO COMPLETO DA FUNÇÃO load_data
    log_messages: List[str] = []
    cdf = pd.DataFrame()
    vdf = pd.DataFrame()

    # --- Candidatos ---
    try:
        fs = get_s3_fs()
        candidatos_s3_path = f"{S3_BUCKET}/data/{CANDIDATOS_FILE}"
        with fs.open(candidatos_s3_path, "rb") as f:
            cdf = pd.read_csv(f, nrows=_max_rows, encoding="latin-1", engine="python", on_bad_lines="skip")
        
        critical_cols = [CANDIDATO_ID_COL, CV_TEXT_COL]
        missing_critical_cols = [col for col in critical_cols if col not in cdf.columns]
        if missing_critical_cols: raise KeyError(f"Erro ao carregar candidatos: colunas críticas ausentes: {missing_critical_cols}")
        cdf = cdf.dropna(subset=critical_cols)
        cdf[CV_TEXT_COL] = cdf[CV_TEXT_COL].astype(str)
        required_candidato_cols = [CANDIDATO_ID_COL, CV_TEXT_COL] + CANDIDATO_METADATA_COLS
        cols_to_keep = [col for col in required_candidato_cols if col in cdf.columns]
        cdf = cdf[cols_to_keep] 
        cdf = cdf.reset_index(drop=True) 
        log_messages.append(f"✅ Candidatos carregados: {len(cdf):,} registros")
    except Exception as e: log_messages.append(f"❌ Erro ao carregar candidatos: {str(e)}"); cdf = pd.DataFrame()

    # --- Vagas ---
    try:
        fs = get_s3_fs()
        vagas_s3_path = f"{S3_BUCKET}/data/{VAGAS_FILE}"
        with fs.open(vagas_s3_path, "rb") as f: vdf = pd.read_csv(f, encoding="latin-1")

        required_vaga_cols = [VAGA_ID_COL, "titulo_vaga"]
        missing_cols = [col for col in required_vaga_cols if col not in vdf.columns]
        if missing_cols: raise KeyError(f"Vagas sem colunas: {missing_cols}")

        text_cols_to_combine = ["titulo_vaga", "objetivo_vaga", "nivel_profissional", "principais_atividades", "competencias", "habilidades_comportamentais"]
        existing_text_cols = [col for col in text_cols_to_combine if col in vdf.columns]
        
        if existing_text_cols:
            vdf[VAGA_TEXT_COL] = vdf[existing_text_cols].fillna("").astype(str).agg(" ".join, axis=1)
        else: raise ValueError(f"Nenhuma coluna base encontrada para criar '{VAGA_TEXT_COL}'.")

        vdf = vdf.dropna(subset=[VAGA_TEXT_COL, VAGA_ID_COL]).reset_index(drop=True)
        vdf = vdf.reset_index(drop=True)
        log_messages.append(f"✅ Vagas carregadas: {len(vdf):,} registros")

    except Exception as e: log_messages.append(f"❌ Erro ao carregar vagas: {str(e)}"); vdf = pd.DataFrame()

    if cdf.empty or vdf.empty: log_messages.append("🚨 Crítico: Dados insuficientes para continuar.")
    return cdf, vdf, log_messages

@st.cache_data(show_spinner="Gerenciando cache de embeddings...", hash_funcs={SentenceTransformer: lambda _: None})
def get_or_create_embeddings(
    df: pd.DataFrame, text_col: str, filename: str, encoder: SentenceTransformer, _use_cache: bool = True
) -> np.ndarray:
    # COPIE O CORPO COMPLETO DA FUNÇÃO get_or_create_embeddings
    if df.empty: return np.zeros((0, 384), dtype="float32")
    content_hash = _hash_df(df, [text_col], sample_rows=20000)
    cache_key = f"emb_{filename}_{content_hash}"
    if _use_cache and cache_key in st.session_state: return st.session_state[cache_key]

    fs = get_s3_fs()
    s3_emb_path = f"{S3_BUCKET}/data/embeddings/{filename}"
    if _use_cache and fs.exists(s3_emb_path):
        with st.spinner(f"☁️ Carregando embeddings do S3: {filename}"):
            with fs.open(s3_emb_path, "rb") as f: embeddings = np.load(f)
        if embeddings.shape[0] == len(df):
            embeddings = _l2_normalize(embeddings.astype("float32"))
            st.session_state[cache_key] = embeddings
            return embeddings
    
    texts = df[text_col].astype(str).tolist()
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    batch_size = 64
    all_embeddings: List[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_embeddings = encoder.encode(batch_texts, show_progress_bar=False, convert_to_numpy=True, batch_size=min(batch_size, 32)).astype("float32")
        all_embeddings.append(batch_embeddings)
        progress = min((i + batch_size) / len(texts), 1.0)
        progress_bar.progress(progress)
        status_text.text(f"Processando: {min(i + batch_size, len(texts)):,} / {len(texts):,}")

    embeddings = _l2_normalize(np.vstack(all_embeddings).astype("float32"))
    progress_bar.empty()
    status_text.empty()

    try:
        with st.spinner("💾 Salvando embeddings no S3..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as tmp:
                np.save(tmp.name, embeddings)
                tmp_path = tmp.name
            fs.put(tmp_path, s3_emb_path)
            os.unlink(tmp_path)
    except Exception as e: logger.warning(f"Falha ao salvar embeddings no S3 ({filename}): {e}")

    st.session_state[cache_key] = embeddings
    return embeddings


# ==============================================================================
# 4. FUNÇÕES DE PREDIÇÃO (COPIADAS)
# ==============================================================================

def predict_match_and_rank(
    vaga_embedding: np.ndarray, all_candidate_embeddings: np.ndarray, cdf: pd.DataFrame, 
    bst: Any, top_k: int = 1000
) -> pd.DataFrame:
    # COPIE O CORPO COMPLETO DA FUNÇÃO predict_match_and_rank (removendo a dependência 'le')
    if cdf.empty or all_candidate_embeddings.size == 0: return pd.DataFrame()
    n_cand = min(len(cdf), all_candidate_embeddings.shape[0])
    if n_cand == 0: return pd.DataFrame()
    cand_emb = all_candidate_embeddings[:n_cand]
    cdf_safe = cdf.iloc[:n_cand].reset_index(drop=True)
    sims = cand_emb @ vaga_embedding.astype("float32")
    k = min(top_k, n_cand)
    top_idx = np.argpartition(sims, -k)[-k:]
    top_idx = top_idx[np.argsort(-sims[top_idx])]
    X_left = all_candidate_embeddings[top_idx]
    X_right = np.broadcast_to(vaga_embedding, X_left.shape) 
    X_predict_768 = np.hstack([X_left, X_right]).astype(np.float32, copy=False)
    X_predict = np.hstack([X_predict_768, X_predict_768]).astype(np.float32, copy=False)
    
    predictions: np.ndarray
    try:
        if isinstance(bst, xgb.Booster):
            dtest = xgb.DMatrix(X_predict)
            predictions = bst.predict(dtest)
        else: # sklearn.XGBClassifier ou similar
            if hasattr(bst, "predict_proba"):
                proba = bst.predict_proba(X_predict)
                if proba.ndim == 2 and proba.shape[1] > 1: predictions = proba[:, 1]
                else: predictions = proba.ravel()
            else:
                pred = bst.predict(X_predict)
                predictions = pred.astype(np.float32, copy=False) if isinstance(pred, np.ndarray) else np.array(pred, dtype=np.float32)
    except Exception as e: return pd.DataFrame()

    results_df = cdf_safe.iloc[top_idx].copy()
    results_df["probabilidade_match"] = predictions.astype(np.float32, copy=False)
    results_df = results_df.sort_values("probabilidade_match", ascending=False).reset_index(drop=True)
    results_df["rank"] = np.arange(1, len(results_df) + 1, dtype=int)
    return results_df


# ==============================================================================
# 5. FUNÇÕES AUXILIARES PARA UI (COPIADAS)
# ==============================================================================
# COPIE format_currency e display_load_logs (display_candidate_card não é usada nesta tela)
def format_currency(value: Any) -> str:
    try:
        val = float(value) if pd.notna(value) else 0.0
        return f"R$ {val:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "R$ 0,00"

def display_load_logs(log_messages: List[str]) -> bool:
    with st.container():
        st.subheader("📊 Status do Carregamento de Dados")
        data_loaded_ok = True
        for msg in log_messages:
            if "✅" in msg: st.success(msg)
            elif "❌" in msg or "🚨" in msg:
                st.error(msg)
                data_loaded_ok = False
            else: st.info(msg)
        return data_loaded_ok

# ==============================================================================
# 6. EXECUÇÃO PRINCIPAL DA NOVA TELA
# ==============================================================================

def top_matches_page():
    st.title("🥇 Top Matches: Análise Geral do Pool de Talentos")
    st.info("Aqui, identificamos o **candidato com maior probabilidade de Match** para cada vaga no sistema, ranqueando as vagas pelo seu melhor resultado possível.")
    
    # --- Sidebar: Configurações e Carregamento ---
    with st.sidebar:
        st.header("Análise Geral")
        max_vagas = st.slider("Top N Vagas para Analisar", 5, 200, 50)
        max_candidates = st.slider("Máx. Candidatos a Carregar (Pool)", 100, 10000, 5000)
        top_k_for_xgboost = st.slider("Top K para Predição XGBoost", 100, 5000, 1000)
        use_cache = st.checkbox("Usar Cache de Embeddings", value=True)
        st.markdown("---")

        try:
            cdf, vdf, log_messages = load_data(max_candidates)
            data_ok = display_load_logs(log_messages)
            if not data_ok or cdf.empty or vdf.empty: return

            bst = load_models()
            encoder = load_encoder()
            candidate_embeddings = get_or_create_embeddings(cdf, CV_TEXT_COL, EMBEDDINGS_FILE, encoder, use_cache)
            vaga_embeddings = get_or_create_embeddings(vdf, VAGA_TEXT_COL, VAGAS_EMBEDDINGS_FILE, encoder, use_cache)

        except Exception as e:
            st.error(f"❌ Erro Crítico de Inicialização/Conexão: {e}")
            return
        
        st.info(f"Pool: {len(cdf):,} candidatos e {len(vdf):,} vagas carregadas.")

    # --- Processamento (Loop nas Vagas) ---
    all_top_results = []
    
    st.subheader(f"Calculando Top Matches para as {min(len(vdf), max_vagas)} vagas...")
    progress_bar = st.progress(0.0)
    start_time = time.time()

    vagas_a_analisar = vdf.head(max_vagas)
    
    for i, vaga_row in vagas_a_analisar.iterrows():
        vaga_id = vaga_row[VAGA_ID_COL]
        vaga_index = vaga_row.name 
        vaga_embedding = vaga_embeddings[vaga_index]
        
        results_df = predict_match_and_rank(
            vaga_embedding, candidate_embeddings, cdf, bst, top_k=top_k_for_xgboost
        )

        if not results_df.empty:
            top_candidate = results_df.iloc[0].copy()
            
            top_results = {
                'vaga_id': vaga_id,
                'titulo_vaga': vaga_row['titulo_vaga'],
                'melhor_match_prob': float(top_candidate['probabilidade_match']),
                'id_candidato_top': top_candidate[CANDIDATO_ID_COL],
                'status_candidato_top': top_candidate.get('status', 'N/A'),
                'nivel_hierarquico': top_candidate.get('nivel_hierarquico', 'N/A'),
                'salario_atual_top': top_candidate.get('salario_atual', 0),
            }
            all_top_results.append(top_results)
            
        progress_bar.progress((i + 1) / len(vagas_a_analisar))
        
    processing_time = time.time() - start_time
    progress_bar.empty()
    st.caption(f"⏱️ Tempo de Processamento: {processing_time:.2f} s")
    
    # --- Exibição de Resultados ---
    if not all_top_results:
        st.warning("Nenhum match encontrado para as vagas analisadas.")
        return

    final_df = pd.DataFrame(all_top_results)
    final_df = final_df.sort_values('melhor_match_prob', ascending=False).reset_index(drop=True)
    
    st.header(f"Tabela de Top Matches (Ranqueada)")
    
    df_display = final_df.copy()
    df_display['melhor_match_prob'] = df_display['melhor_match_prob'].map(lambda x: f"{x:.1%}")
    df_display['salario_atual_top'] = df_display['salario_atual_top'].apply(format_currency)
    
    df_display.columns = [
        'ID Vaga', 'Título da Vaga', 'Melhor Match', 'ID Candidato Top', 
        'Status do Candidato', 'Nível Candidato', 'Salário Atual do Candidato',
    ]

    st.dataframe(df_display, use_container_width=True, height=min(len(df_display) * 35 + 38, 700))
    
    # Download
    st.markdown("---")
    csv = df_display.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Baixar Top Matches (CSV)",
        data=csv,
        file_name='top_matches_summary.csv',
        mime='text/csv',
    )


if __name__ == "__main__":
    top_matches_page()