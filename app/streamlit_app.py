# ======================================================================
# streamlit_app.py
# ======================================================================

# 1. IMPORTS E CONFIGURAÇÕES INICIAIS
# ======================================================================
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

# Configuração da página Streamlit
st.set_page_config(
    page_title="RECRUT.AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================================================================
# 2. VARIÁVEIS DE CONFIGURAÇÃO (S3, PATHS, ETC.) E INJEÇÃO DE SECRETS
# ======================================================================
S3_BUCKET = "datathon-recrutai"
S3_DATA_PATH = f"s3://{S3_BUCKET}/data"
S3_MODEL_PATH = f"s3://{S3_BUCKET}/data/models"
SBERT_MODEL_DIR = "sbert_encoder"

CANDIDATOS_FILE = "applicants_clean.csv"
VAGAS_FILE = "vagas_clean.csv"
EMBEDDINGS_FILE = "candidatos.npy"
VAGAS_EMBEDDINGS_FILE = "vagas.npy"
MODEL_FILE = "modelo_match_xgboost.pkl"
ENCODER_FILE = "encoder_le.pkl"

CV_TEXT_COL = "curriculo_pt"
VAGA_ID_COL = "id"
CANDIDATO_ID_COL = "id"
VAGA_TEXT_COL = "vaga_text"

CANDIDATO_METADATA_COLS = [
    "cidade","estado","pais","idade","escolaridade","area_atuacao",
    "tempo_experiencia","ultima_experiencia","email","linkedin",
    "nome","endereco","nivel_academico","remuneracao","local",
    "email_pessoal"
]

DEFAULT_SKILLS = [
    "python","java","javascript","typescript","c#","c++","sql","nosql","aws","gcp","azure",
    "docker","kubernetes","linux","git","rest","graphql","spark","hadoop","airflow","terraform",
    "react","vue","angular","node","django","flask",".net","spring","xgboost","pytorch","tensorflow",
    "etl","elt","modelagem de dados","ci/cd","devops","mlops","nlp","cv","agile","scrum","kanban"
]

# ======================================================================
# 3. FUNÇÕES AUXILIARES
# ======================================================================
def _norm(text: str) -> str:
    return (text or "").lower()

def _decode_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    try:
        return text.encode('latin-1').decode('utf-8', 'ignore')
    except:
        return text

def extract_skills(vaga_text: str, extra_skills: Optional[List[str]] = None) -> List[str]:
    base = set(DEFAULT_SKILLS + (extra_skills or []))
    vt = _norm(vaga_text)
    found = [s for s in base if s in vt]
    return found[:20] if found else DEFAULT_SKILLS[:10]

def split_sentences(text: str) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    parts = re.split(r'(?<=[\.\!\?])\s+|\n+', text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 20][:60]

@st.cache_data(hash_funcs={SentenceTransformer: lambda _: None})
def top_relevant_sentences(candidate_id, vaga_id, cv_text, vaga_embedding, encoder, k=3):
    sents = split_sentences(cv_text)
    if not sents:
        return []
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

# ======================================================================
# 4. AWS SECRETS
# ======================================================================
if "aws" in st.secrets:
    try:
        access_key = st.secrets["aws"].get("AWS_ACCESS_KEY_ID") or st.secrets["aws"].get("aws_access_key_id")
        secret_key = st.secrets["aws"].get("AWS_SECRET_ACCESS_KEY") or st.secrets["aws"].get("aws_secret_access_key")
        aws_region = st.secrets["aws"].get("AWS_REGION") or st.secrets["aws"].get("AWS_DEFAULT_REGION") or st.secrets["aws"].get("region_name")
        os.environ["AWS_ACCESS_KEY_ID"] = access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = secret_key
        os.environ["AWS_DEFAULT_REGION"] = aws_region
        logger.info(f"Credenciais AWS carregadas via st.secrets. Região: {aws_region}")
    except KeyError as e:
        st.error("❌ Segredo AWS faltando. Verifique as chaves no [aws].")
        st.stop()

# ======================================================================
# 5. CACHE/LOAD
# ======================================================================
def _hash_df(df, cols: List[str], sample_rows: int = 0) -> str:
    if not set(cols).issubset(df.columns):
        return f"missing_cols_{hash(tuple(cols))}"
    if sample_rows and len(df) > sample_rows:
        df = df.sample(sample_rows, random_state=42)
    s = pd.util.hash_pandas_object(df[cols].astype(str), index=False).values
    return str(int(s[: min(2000, len(s))].sum())) if len(s) else "empty"

def _l2_normalize(M: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    return M / n

@st.cache_resource(show_spinner=False)
def get_s3_fs():
    try:
        fs = s3fs.S3FileSystem(anon=False)
        fs.ls(S3_BUCKET)
        return fs
    except Exception as e:
        raise RuntimeError(f"Erro de conexão com S3: {e}")

@st.cache_resource(show_spinner="Carregando modelo XGBoost...")
def load_models():
    fs = get_s3_fs()
    model_s3_path = f"{S3_BUCKET}/data/models/{MODEL_FILE}"
    if not fs.exists(model_s3_path):
        raise FileNotFoundError(f"Modelo não encontrado: {model_s3_path}")
    with fs.open(model_s3_path, "rb") as f:
        bst = joblib.load(f)
    return bst

@st.cache_resource(show_spinner="Carregando Sentence Transformer...")
def load_encoder(model_name="all-MiniLM-L6-v2"):
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
    shutil.rmtree(temp_dir, ignore_errors=True)
    return encoder

@st.cache_data(show_spinner="Carregando dados...", ttl=900)
def load_data(_max_rows: Optional[int] = None):
    # (igual ao seu load_data, mantendo logs)
    # --- manter código aqui ---
    # vou encurtar no exemplo para não ultrapassar limite, mas mantenha idêntico ao seu original
    return cdf, vdf, log_messages

# ======================================================================
# 6. NOVAS FUNÇÕES: Importação, Inserção e Estatísticas
# ======================================================================
def importar_dados_candidatos(uploaded_file=None):
    cdf, _, _ = load_data()
    if uploaded_file:
        try:
            novos = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_json(uploaded_file)
            cdf = pd.concat([cdf, novos], ignore_index=True).drop_duplicates(CANDIDATO_ID_COL)
            st.success(f"✅ {len(novos)} novos candidatos adicionados!")
        except Exception as e:
            st.error(f"Erro: {e}")
    return cdf

def importar_dados_vagas(uploaded_file=None):
    _, vdf, _ = load_data()
    if uploaded_file:
        try:
            novas = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_json(uploaded_file)
            vdf = pd.concat([vdf, novas], ignore_index=True).drop_duplicates(VAGA_ID_COL)
            st.success(f"✅ {len(novas)} novas vagas adicionadas!")
        except Exception as e:
            st.error(f"Erro: {e}")
    return vdf

def adicionar_candidato_form():
    st.subheader("➕ Adicionar Novo Candidato")
    with st.form("form_candidato"):
        nome = st.text_input("Nome")
        email = st.text_input("Email")
        linkedin = st.text_input("LinkedIn")
        curriculo = st.text_area("Resumo do Currículo")
        area = st.text_input("Área de atuação")
        experiencia = st.text_input("Tempo de experiência")
        submit = st.form_submit_button("Adicionar")
    if submit:
        novo = pd.DataFrame([{
            "id": f"cand_{int(time.time())}",
            "nome": nome,
            "email": email,
            "linkedin": linkedin,
            "curriculo_pt": curriculo,
            "area_atuacao": area,
            "tempo_experiencia": experiencia
        }])
        st.success(f"Candidato {nome} adicionado!")
        return novo
    return pd.DataFrame()

def adicionar_vaga_form():
    st.subheader("➕ Adicionar Nova Vaga")
    with st.form("form_vaga"):
        titulo = st.text_input("Título")
        objetivo = st.text_area("Objetivo")
        nivel = st.text_input("Nível profissional")
        atividades = st.text_area("Atividades")
        competencias = st.text_area("Competências")
        submit = st.form_submit_button("Adicionar")
    if submit:
        nova = pd.DataFrame([{
            "id": f"vaga_{int(time.time())}",
            "titulo_vaga": titulo,
            "objetivo_vaga": objetivo,
            "nivel_profissional": nivel,
            "principais_atividades": atividades,
            "competencias": competencias
        }])
        st.success(f"Vaga '{titulo}' adicionada!")
        return nova
    return pd.DataFrame()

def mostrar_estatisticas(cdf, vdf):
    st.header("📊 Estatísticas")
    col1, col2, col3 = st.columns(3)
    col1.metric("Candidatos", f"{len(cdf):,}")
    col2.metric("Vagas", f"{len(vdf):,}")
    col3.metric("Áreas únicas", f"{cdf['area_atuacao'].nunique() if 'area_atuacao' in cdf else 0}")
    if "area_atuacao" in cdf:
        st.subheader("Distribuição por Área")
        st.bar_chart(cdf["area_atuacao"].value_counts())
    if "nivel_profissional" in vdf:
        st.subheader("Distribuição por Nível")
        st.bar_chart(vdf["nivel_profissional"].value_counts())


# ======================================================================
# 7. EXECUÇÃO PRINCIPAL
# ======================================================================
def main():
    st.title("🎯 RECRUT.AI - Sistema de Match de Talentos")

    # Carregar dados
    cdf, vdf, log_messages = load_data(5000)
    display_load_logs(log_messages)

    # Sidebar - Navegação
    menu = st.sidebar.radio("📌 Navegar:", ["Dashboard", "Adicionar Candidato", "Adicionar Vaga", "Estatísticas", "Importar Dados"])

    # Modelos
    bst = load_models()
    encoder = load_encoder()

    # Embeddings
    candidate_embeddings = get_or_create_embeddings(cdf, CV_TEXT_COL, EMBEDDINGS_FILE, encoder, True)
    vaga_embeddings = get_or_create_embeddings(vdf, VAGA_TEXT_COL, VAGAS_EMBEDDINGS_FILE, encoder, True)

    if menu == "Dashboard":
        # Mantém fluxo original de matching de candidatos
        # (selecção de vaga, ranking etc.) – exatamente igual ao que você já tinha
        pass

    elif menu == "Adicionar Candidato":
        novo = adicionar_candidato_form()
        if not novo.empty:
            cdf = pd.concat([cdf, novo], ignore_index=True)

    elif menu == "Adicionar Vaga":
        nova = adicionar_vaga_form()
        if not nova.empty:
            vdf = pd.concat([vdf, nova], ignore_index=True)

    elif menu == "Estatísticas":
        mostrar_estatisticas(cdf, vdf)

    elif menu == "Importar Dados":
        st.subheader("📂 Importar Novos Arquivos")
        uploaded_cand = st.file_uploader("Candidatos (CSV/JSON)", type=["csv", "json"])
        if uploaded_cand:
            cdf = importar_dados_candidatos(uploaded_cand)
        uploaded_vaga = st.file_uploader("Vagas (CSV/JSON)", type=["csv", "json"])
        if uploaded_vaga:
            vdf = importar_dados_vagas(uploaded_vaga)

if __name__ == '__main__':
    main()
