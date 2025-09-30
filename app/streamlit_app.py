# ==============================================================================
# 1. IMPORTS E CONFIGURAÇÕES INICIAIS
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

# Configuração da página Streamlit
st.set_page_config(
    page_title="RECRUT.AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================================================
# 2. VARIÁVEIS DE CONFIGURAÇÃO (S3, PATHS, ETC.) E INJEÇÃO DE SECRETS
# ==============================================================================
# Configurações do S3
S3_BUCKET = "datathon-recrutai"
S3_DATA_PATH = f"s3://{S3_BUCKET}/data"
S3_MODEL_PATH = f"s3://{S3_BUCKET}/data/models"

# Caminho do SBERT no S3 (pasta)
SBERT_MODEL_DIR = "sbert_encoder"

# Nomes dos arquivos
CANDIDATOS_FILE = "applicants_clean.csv"
VAGAS_FILE = "vagas_clean.csv"
EMBEDDINGS_FILE = "candidatos.npy"
VAGAS_EMBEDDINGS_FILE = "vagas.npy"
MODEL_FILE = "modelo_match_xgboost.pkl"
ENCODER_FILE = "encoder_le.pkl"

# Colunas para o embedding e IDs
CV_TEXT_COL = "curriculo_pt"
VAGA_ID_COL = "id"
CANDIDATO_ID_COL = "id"

# Coluna para o texto combinado da vaga
VAGA_TEXT_COL = "vaga_text"

# NOVO: Colunas de metadados dos candidatos para exibição na UI
CANDIDATO_METADATA_COLS = [
    # ... colunas que já estavam ...
    "cidade",
    "estado",
    "pais",
    "idade",
    "escolaridade",
    "area_atuacao",
    "tempo_experiencia",
    "ultima_experiencia",
    "email",
    "linkedin",
    "nome",               # Novo
    "endereco",           # Novo
    "nivel_academico",    # Novo
    "remuneracao",        # Novo
    "local",              # Novo (para endereço)
    "email_pessoal",      # Novo (para contatos)
    # ...
]
# ---------------------------------------------
# 🔎 Helpers de explicabilidade
# ---------------------------------------------
DEFAULT_SKILLS = [
    # tech
    "python","java","javascript","typescript","c#","c++","sql","nosql","aws","gcp","azure",
    "docker","kubernetes","linux","git","rest","graphql","spark","hadoop","airflow","terraform",
    "react","vue","angular","node","django","flask",".net","spring","xgboost","pytorch","tensorflow",
    # dados / soft
    "etl","elt","modelagem de dados","ci/cd","devops","mlops","nlp","cv","agile","scrum","kanban"
]

def _norm(text: str) -> str:
    return (text or "").lower()

def _decode_text(text: str) -> str:
    """Tenta corrigir a codificação de texto lido erroneamente (ex: latin-1 lendo utf-8)."""
    if not isinstance(text, str):
        return ""
    try:
        # Tenta decodificar o texto que parece ser "latin-1 que deveria ser utf-8"
        return text.encode('latin-1').decode('utf-8', 'ignore')
    except:
        # Se falhar, retorna o texto original
        return text

def extract_skills(vaga_text: str, extra_skills: Optional[List[str]] = None) -> List[str]:
    """Extrai uma lista de skills-alvo a partir do texto da vaga (set simples)."""
    base = set(DEFAULT_SKILLS + (extra_skills or []))
    vt = _norm(vaga_text)
    found = [s for s in base if s in vt]
    # se achar menos de 5, ainda retorna um conjunto mínimo útil (top 10 do default)
    return found[:20] if found else DEFAULT_SKILLS[:10]

def split_sentences(text: str) -> List[str]:
    """Split simples e robusto (sem novas deps) para sentenças do CV."""
    if not isinstance(text, str) or not text.strip():
        return []
    # quebra por . ! ? e quebras de linha
    parts = re.split(r'(?<=[\.\!\?])\s+|\n+', text.strip())
    # remove lixo e sentenças muito curtas
    return [p.strip() for p in parts if len(p.strip()) > 20][:60]  # no máx 60 p/ velocidade

@st.cache_data(hash_funcs={SentenceTransformer: lambda _: None})
def top_relevant_sentences(
    candidate_id: Any,
    vaga_id: Any,
    cv_text: str,
    vaga_embedding: np.ndarray,
    encoder: SentenceTransformer,
    k: int = 3
) -> List[str]:
    """
    Retorna as k sentenças do CV mais próximas do embedding da vaga (explicabilidade).
    Cacheia por (candidate_id, vaga_id) p/ desempenho.
    """
    sents = split_sentences(cv_text)
    if not sents:
        return []
    # embed só as sentenças (rápido, limitei a 60 acima)
    sent_emb = encoder.encode(sents, show_progress_bar=False, convert_to_numpy=True, batch_size=32).astype("float32")
    # normaliza para cosine ~ produto interno
    sent_emb = _l2_normalize(sent_emb)
    vaga_emb = vaga_embedding.astype("float32")
    scores = (sent_emb @ vaga_emb).reshape(-1)
    k = min(k, len(sents))
    top_idx = np.argpartition(scores, -k)[-k:]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return [sents[i] for i in top_idx]

def render_badges(items: list) -> str:
    """Renders a list of items as a series of stylized HTML badges."""
    return " ".join([f"<span style='display:inline-block;padding:2px 8px;border-radius:999px;background:#1f6feb20;border:1px solid #1f6feb55;margin:2px 6px 2px 0;font-size:12px'>{html.escape(i)}</span>" for i in items[:12]])
# --- 🎯 INJEÇÃO DE SECRETS DO STREAMLIT CLOUD ---
if "aws" in st.secrets:
    try:
        # tenta maiúsculas
        access_key = st.secrets["aws"].get("AWS_ACCESS_KEY_ID")
        secret_key = st.secrets["aws"].get("AWS_SECRET_ACCESS_KEY")
        aws_region = st.secrets["aws"].get("AWS_REGION") or st.secrets["aws"].get("AWS_DEFAULT_REGION")

        # fallback minúsculas
        if not access_key:
            access_key = st.secrets["aws"]["aws_access_key_id"]
            secret_key = st.secrets["aws"]["aws_secret_access_key"]
            aws_region = st.secrets["aws"].get("region_name")

        # exporta p/ env
        os.environ["AWS_ACCESS_KEY_ID"] = access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = secret_key
        os.environ["AWS_DEFAULT_REGION"] = aws_region

        logger.info(f"Credenciais AWS carregadas via st.secrets. Região: {aws_region}")
    except KeyError as e:
        logger.error(f"Erro: Segredo AWS faltando. Chave não encontrada: {e}.")
        st.error("❌ Segredo AWS faltando. Verifique se as chaves (ID, SECRET, REGION) estão no formato [aws] correto.")
        #st.stop() # Comentado para permitir testes locais sem secrets

# ==============================================================================
# 3. FUNÇÕES UTILITÁRIAS DE HASH/NORMALIZAÇÃO E CACHE/LOAD
# ==============================================================================

def _hash_df(df: pd.DataFrame, cols: List[str], sample_rows: int = 0) -> str:
    """
    Gera um hash leve e estável baseado no conteúdo das colunas indicadas.
    Usa amostra (opcional) para performance em bases gigantes.
    """
    if not set(cols).issubset(df.columns):
        return f"missing_cols_{hash(tuple(cols))}"
    if sample_rows and len(df) > sample_rows:
        df = df.sample(sample_rows, random_state=42)

    # hash_pandas_object já é estável; soma parcial para obter um valor curto
    s = pd.util.hash_pandas_object(df[cols].astype(str), index=False).values
    # usa apenas primeiros 2000 itens para performance
    return str(int(s[: min(2000, len(s))].sum())) if len(s) else "empty"

def _l2_normalize(M: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    return M / n

@st.cache_resource(show_spinner=False)
def get_s3_fs():
    """Retorna o filesystem do S3 com configuração correta."""
    try:
        # Se as variáveis de ambiente não existirem, s3fs falhará
        if not os.environ.get("AWS_ACCESS_KEY_ID"):
             raise RuntimeError("Credenciais AWS não configuradas. Verifique st.secrets.")

        fs = s3fs.S3FileSystem(anon=False)
        # valida acesso ao bucket
        # fs.ls(S3_BUCKET) # Comentado para evitar erro se a região estiver errada, mas as credenciais OK
        return fs
    except Exception as e:
        raise RuntimeError(f"Erro de conexão com S3: {e}. Verifique as credenciais AWS.")

@st.cache_resource(show_spinner="Carregando modelo XGBoost do S3...")
def load_models() -> Any:
    """Carrega o modelo XGBoost do S3."""
    fs = get_s3_fs()
    model_s3_path = f"{S3_BUCKET}/data/models/{MODEL_FILE}"

    if not fs.exists(model_s3_path):
        raise FileNotFoundError(f"Arquivo do modelo XGBoost não encontrado: s3://{model_s3_path}")

    with fs.open(model_s3_path, "rb") as f:
        bst = joblib.load(f)

    if bst is None:
        raise ValueError("Modelo XGBoost está vazio")

    return bst

@st.cache_resource(show_spinner="Carregando Sentence Transformer...")
def load_encoder(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Carrega o modelo SBERT. Tenta S3; se não existir, baixa da internet."""
    temp_dir = None
    try:
        fs = get_s3_fs()
        sbert_s3_path = f"{S3_BUCKET}/{SBERT_MODEL_DIR}"
        test_file_path = f"{sbert_s3_path}/config.json"

        if not fs.exists(test_file_path):
            logger.warning(f"Modelo SBERT não encontrado em S3 ({sbert_s3_path}). Baixando {model_name}...")
            encoder = SentenceTransformer(model_name)
            _ = encoder.encode(["probe"], convert_to_numpy=True)
            return encoder

        temp_dir = tempfile.mkdtemp()
        local_model_path = os.path.join(temp_dir, SBERT_MODEL_DIR)
        fs.get(sbert_s3_path, local_model_path, recursive=True)

        encoder = SentenceTransformer(local_model_path)
        _ = encoder.encode(["probe"], convert_to_numpy=True)
        return encoder

    except Exception as e:
        logger.error(f"Falha ao carregar SBERT do S3, tentando download local: {e}")
        try:
            encoder = SentenceTransformer(model_name)
            _ = encoder.encode(["probe"], convert_to_numpy=True)
            return encoder
        except Exception as e_local:
            raise RuntimeError(f"Falha crítica ao carregar SBERT: {e_local}")
            
    finally:
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"Diretório temporário limpo: {temp_dir}")
            except Exception as e:
                logger.error(f"Erro ao limpar diretório temporário: {e}")

@st.cache_data(show_spinner="Carregando dados dos candidatos e vagas do S3...", ttl=900)
def load_data(_max_rows: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Carrega os DataFrames de candidatos e vagas do S3.
    Inclui colunas de metadados dos candidatos e garante a preservação delas.
    """
    log_messages: List[str] = []
    cdf = pd.DataFrame()
    vdf = pd.DataFrame()

    # --- Candidatos ---
    try:
        fs = get_s3_fs()
        candidatos_s3_path = f"{S3_BUCKET}/data/{CANDIDATOS_FILE}"
        log_messages.append(f"📁 Buscando candidatos em: s3://{candidatos_s3_path}")

        if not fs.exists(candidatos_s3_path):
            raise FileNotFoundError(f"Arquivo de candidatos não encontrado: s3://{candidatos_s3_path}")

        with fs.open(candidatos_s3_path, "rb") as f:
            cdf = pd.read_csv(
                f,
                nrows=_max_rows,
                encoding="latin-1", # Mantido latin-1 pois é a sua configuração
                engine="python",
                on_bad_lines="skip",
            )

        # AJUSTE: Incluir metadados nas colunas obrigatórias
        required_candidato_cols = [CANDIDATO_ID_COL, CV_TEXT_COL] + CANDIDATO_METADATA_COLS
        
        # 1. Checagem de colunas CRÍTICAS (ID e CV)
        critical_cols = [CANDIDATO_ID_COL, CV_TEXT_COL]
        missing_critical_cols = [col for col in critical_cols if col not in cdf.columns]
        if missing_critical_cols:
            raise KeyError(f"Erro ao carregar candidatos: colunas críticas ausentes: {missing_critical_cols}")

        # 2. Drop NaNs apenas nas colunas críticas e garante tipo str para o texto
        cdf = cdf.dropna(subset=critical_cols)
        cdf[CV_TEXT_COL] = cdf[CV_TEXT_COL].astype(str)
        
        # 🟢 CORREÇÃO: Aplicar decodificação para tentar resolver problemas de acentuação
        cdf[CV_TEXT_COL] = cdf[CV_TEXT_COL].apply(_decode_text)

        # 3. Preservar APENAS as colunas que existem no CSV e que são necessárias (ID, CV, Metadados)
        cols_to_keep = [col for col in required_candidato_cols if col in cdf.columns]
        cdf = cdf[cols_to_keep] 
        
        # 4. Garante índice posicional contínuo
        cdf = cdf.reset_index(drop=True) 

        log_messages.append(f"✅ Candidatos carregados: {len(cdf):,} registros")

    except Exception as e:
        log_messages.append(f"❌ Erro ao carregar candidatos: {str(e)}")
        cdf = pd.DataFrame()

    # --- Vagas ---
    try:
        fs = get_s3_fs()
        vagas_s3_path = f"{S3_BUCKET}/data/{VAGAS_FILE}"
        log_messages.append(f"📁 Buscando vagas em: s3://{vagas_s3_path}")

        if not fs.exists(vagas_s3_path):
            raise FileNotFoundError(f"Arquivo de vagas não encontrado: s3://{vagas_s3_path}")

        with fs.open(vagas_s3_path, "rb") as f:
            vdf = pd.read_csv(f, encoding="latin-1")

        required_vaga_cols = [VAGA_ID_COL, "titulo_vaga"]
        missing_cols = [col for col in required_vaga_cols if col not in vdf.columns]
        if missing_cols:
            raise KeyError(f"Vagas sem colunas: {missing_cols}")

        text_cols_to_combine = [
            "titulo_vaga",
            "objetivo_vaga",
            "nivel_profissional",
            "principais_atividades",
            "competencias",
            "habilidades_comportamentais",
        ]
        existing_text_cols = [col for col in text_cols_to_combine if col in vdf.columns]
        
        # 🟢 CORREÇÃO: Aplicar decodificação nas colunas de texto da vaga antes de combinar
        for col in existing_text_cols:
            vdf[col] = vdf[col].apply(_decode_text)

        if existing_text_cols:
            vdf[VAGA_TEXT_COL] = vdf[existing_text_cols].fillna("").astype(str).agg(" ".join, axis=1)
        else:
            raise ValueError(f"Nenhuma coluna base encontrada para criar '{VAGA_TEXT_COL}'.")

        vdf = vdf.dropna(subset=[VAGA_TEXT_COL, VAGA_ID_COL]).reset_index(drop=True)
        
        # Garante índice posicional contínuo
        vdf = vdf.reset_index(drop=True)

        log_messages.append(f"✅ Vagas carregadas: {len(vdf):,} registros")

    except Exception as e:
        log_messages.append(f"❌ Erro ao carregar vagas: {str(e)}")
        vdf = pd.DataFrame()

    if cdf.empty or vdf.empty:
        log_messages.append("🚨 Crítico: Dados insuficientes para continuar.")

    return cdf, vdf, log_messages

@st.cache_data(
    show_spinner="Gerenciando cache de embeddings...",
    hash_funcs={SentenceTransformer: lambda _: None}, 
)
def get_or_create_embeddings(
    df: pd.DataFrame,
    text_col: str,
    filename: str,
    encoder: SentenceTransformer,
    _use_cache: bool = True,
) -> np.ndarray:
    """
    Gerencia cache de embeddings, com:
    - chave robusta por hash de conteúdo
    - normalização L2
    - dtype float32 (metade da RAM)
    - gravação/ leitura em S3
    """
    if df.empty:
        # Assumindo dimensão 384 para o all-MiniLM-L6-v2
        return np.zeros((0, 384), dtype="float32")

    content_hash = _hash_df(df, [text_col], sample_rows=20000)
    cache_key = f"emb_{filename}_{content_hash}"

    if _use_cache and cache_key in st.session_state:
        return st.session_state[cache_key]

    fs = get_s3_fs()
    s3_emb_path = f"{S3_BUCKET}/data/embeddings/{filename}"

    # 1) Tentar carregar do S3
    if _use_cache and fs.exists(s3_emb_path):
        with st.spinner(f"☁️ Carregando embeddings do S3: {filename}"):
            with fs.open(s3_emb_path, "rb") as f:
                embeddings = np.load(f)
        # Se o shape não bate com o df atual, vamos ignorar o cache e recomputar
        if embeddings.shape[0] == len(df):
            embeddings = _l2_normalize(embeddings.astype("float32"))
            st.session_state[cache_key] = embeddings
            return embeddings
        else:
            st.warning(f"Cache de embeddings desatualizado para {filename}: {embeddings.shape[0]} ≠ {len(df)}. Recalculando...")


    # 2) Gerar novos embeddings
    texts = df[text_col].astype(str).tolist()
    progress_bar = st.progress(0.0)
    status_text = st.empty()

    batch_size = 64
    all_embeddings: List[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_embeddings = encoder.encode(
            batch_texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=min(batch_size, 32),
        ).astype("float32")
        all_embeddings.append(batch_embeddings)

        progress = min((i + batch_size) / len(texts), 1.0)
        progress_bar.progress(progress)
        status_text.text(f"Processando: {min(i + batch_size, len(texts)):,} / {len(texts):,}")

    embeddings = _l2_normalize(np.vstack(all_embeddings).astype("float32"))
    progress_bar.empty()
    status_text.empty()

    # 3) Salvar no S3
    try:
        with st.spinner("💾 Salvando embeddings no S3..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as tmp:
                np.save(tmp.name, embeddings)
                tmp_path = tmp.name
            fs.put(tmp_path, s3_emb_path)
            os.unlink(tmp_path)
    except Exception as e:
        logger.warning(f"Falha ao salvar embeddings no S3 ({filename}): {e}")

    st.session_state[cache_key] = embeddings
    return embeddings

# ==============================================================================
# 4. FUNÇÕES DE PREDIÇÃO
# ==============================================================================

def predict_match_and_rank(
    vaga_embedding: np.ndarray,
    all_candidate_embeddings: np.ndarray,
    cdf: pd.DataFrame,
    bst: Any, 
    le: Any = None, 
    top_k: int = 1000,
) -> pd.DataFrame:
    """
    Calcula matching e ranking de forma otimizada e segura contra desalinhamentos.
    """
    # Checagens básicas
    if cdf.empty or all_candidate_embeddings.size == 0:
        return pd.DataFrame()

    # Garante que não vamos indexar além do limite
    n_cand = min(len(cdf), all_candidate_embeddings.shape[0])
    if n_cand == 0:
        return pd.DataFrame()

    cand_emb = all_candidate_embeddings[:n_cand]
    cdf_safe = cdf.iloc[:n_cand].reset_index(drop=True)

    # similaridade por produto interno (assumindo normalização L2 prévia)
    sims = cand_emb @ vaga_embedding.astype("float32")
    k = min(top_k, n_cand)
    top_idx = np.argpartition(sims, -k)[-k:]
    top_idx = top_idx[np.argsort(-sims[top_idx])]

    # 2. Construção da Matriz de Predição (X_predict)
    X_left = all_candidate_embeddings[top_idx]
    # Cria a matriz de embeddings da vaga replicada
    X_right = np.broadcast_to(vaga_embedding, X_left.shape) 
    
    # Concatenação Simples (768 features)
    X_predict_768 = np.hstack([X_left, X_right]).astype(np.float32, copy=False)

    # CORREÇÃO CRÍTICA: Duplicar as features para 1536 (768 + 768)
    X_predict = np.hstack([X_predict_768, X_predict_768]).astype(np.float32, copy=False)
    
    # 3. Predição compatível com Booster OU sklearn
    predictions: np.ndarray
    try:
        if isinstance(bst, xgb.Booster):
            dtest = xgb.DMatrix(X_predict)
            predictions = bst.predict(dtest)

        else: # sklearn.XGBClassifier ou similar
            if hasattr(bst, "predict_proba"):
                proba = bst.predict_proba(X_predict)
                if proba.ndim == 2 and proba.shape[1] > 1:
                    predictions = proba[:, 1]
                else:
                    predictions = proba.ravel()
            else:
                pred = bst.predict(X_predict)
                predictions = pred.astype(np.float32, copy=False) if isinstance(pred, np.ndarray) else np.array(pred, dtype=np.float32)

    except Exception as e:
        # Se algo der errado, deixe traço claro na interface
        st.error(f"Erro ao gerar predições com o modelo XGBoost: {e}")
        return pd.DataFrame()

    # 4. Formatação e Ranking
    # ✅ CORREÇÃO GARANTIDA: O cdf agora tem índice alinhado ao top_idx
    results_df = cdf_safe.iloc[top_idx].copy()
    results_df["probabilidade_match"] = predictions.astype(np.float32, copy=False)
    results_df = results_df.sort_values("probabilidade_match", ascending=False).reset_index(drop=True)
    results_df["rank"] = np.arange(1, len(results_df) + 1, dtype=int)
    
    return results_df

# ==============================================================================
# 5. NOVAS FUNÇÕES DE PERSISTÊNCIA E INSERÇÃO DE DADOS
# ==============================================================================

def save_dataframe_to_s3(df: pd.DataFrame, filename: str, s3_dir: str = S3_DATA_PATH) -> bool:
    """Salva um DataFrame no S3, sobrescrevendo o arquivo existente."""
    try:
        fs = get_s3_fs()
        s3_path = f"{s3_dir.rstrip('/')}/{filename}"

        csv_buffer = io.StringIO()
        # Usamos encoding 'latin-1' de forma consistente com o load_data
        df.to_csv(csv_buffer, index=False, encoding='latin-1')
        
        with fs.open(s3_path, 'w', encoding='latin-1') as f:
            f.write(csv_buffer.getvalue())

        logger.info(f"✅ DataFrame salvo com sucesso no S3: {s3_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Erro ao salvar DataFrame no S3 ({filename}): {e}")
        st.error(f"❌ Erro ao salvar no S3: {e}")
        return False

def add_new_data_point(
    df: pd.DataFrame, 
    new_data: dict, 
    id_col: str, 
    text_col: str,
    id_prefix: str = "custom"
) -> pd.DataFrame:
    """Adiciona um novo ponto de dado (candidato ou vaga) ao DataFrame existente."""
    
    # Gera um ID único e baseado no tempo
    new_id = f"{id_prefix}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    
    # Cria uma nova linha como Series
    new_row_series = pd.Series(new_data)
    new_row_series[id_col] = new_id
    
    # Cria um DF temporário para garantir que todas as colunas sejam mantidas (incluindo as NaNs)
    df_temp = pd.DataFrame(columns=df.columns)
    # Reindexa a nova linha para ter a ordem e o conjunto de colunas corretos do DF original
    df_temp.loc[0] = new_row_series.reindex(df.columns)
    
    # Aplica a limpeza de texto (importante para o embedding)
    df_temp[text_col] = df_temp[text_col].apply(_decode_text)

    # Concatena a nova linha
    updated_df = pd.concat([df, df_temp], ignore_index=True)
    
    # Invalida o cache de dados e embeddings para forçar o Streamlit a recarregar
    st.cache_data.clear() 
    st.session_state.clear()
    
    return updated_df

# ==============================================================================
# 6. FUNÇÕES AUXILIARES PARA UI
# ==============================================================================

def format_currency(value: Any) -> str:
    """Formata valor para o padrão monetário brasileiro, tratando NaNs/tipos."""
    try:
        # Tenta converter para float. Se for None, NaN ou string vazia, usa 0.0
        val = float(value) if pd.notna(value) else 0.0
        # Formato brasileiro R$ X.XXX,XX
        return f"R$ {val:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "R$ 0,00"
    
def safe_display(value, default_str="N/A"):
    """Retorna o valor ou o default se for nulo/vazio."""
    if pd.isna(value) or value is None or str(value).lower() in ('nan', 'n/a', ''):
        return default_str
    # Garante que o número não seja exibido com 20 casas decimais (caso da remuneração)
    if isinstance(value, (int, float)):
        return str(int(value)) if int(value) == value else f"{value:.2f}"
    return value

def display_candidate_card(candidate_data: pd.Series, rank: int, vaga_row: pd.Series, vaga_embedding: np.ndarray, encoder: SentenceTransformer):
    with st.container(border=True):
        col1, col2, col3 = st.columns([3, 2, 1])

        # === Coluna 1 (Nome e Metadados) ===
        with col1:
            # 🟢 NOVO NOME PRINCIPAL
            nome_display = safe_display(candidate_data.get("nome", candidate_data.get(CANDIDATO_ID_COL, 'Candidato N/A')))
            st.subheader(f"#{rank} - {nome_display}")
            
            # 🟢 NOVO ENDEREÇO
            endereco_display = safe_display(candidate_data.get("endereco", "Localidade N/A"))
            st.write(f"**Endereço:** {endereco_display}")

            linha1 = []
            if pd.notna(candidate_data.get("local", None)):
                 linha1.append(str(candidate_data.get("local")))
            elif pd.notna(candidate_data.get("cidade", None)):
                linha1.append(str(candidate_data.get("cidade")))
            st.caption(" | ".join(linha1) if linha1 else "Local/Cidade: N/A")

            # 🟢 NOVO NÍVEL ACADÊMICO
            st.write(f"**Escolaridade:** {safe_display(candidate_data.get('nivel_academico', 'Nível Acadêmico N/A'))}")
            if pd.notna(candidate_data.get("tempo_experiencia", None)):
                 st.write(f"**Experiência:** {candidate_data.get('tempo_experiencia')}")

        # === Coluna 2 (Remuneração e Contatos) ===
        with col2:
            st.write(f"**Gênero:** {safe_display(candidate_data.get('genero', 'N/A'))}")
            st.write(f"**Área:** {safe_display(candidate_data.get('area_atuacao', 'N/A'))}")
            st.write(f"**Última experiência:** {safe_display(candidate_data.get('ultima_experiencia', 'N/A'))}")
            
            # 🟢 NOVA REMUNERAÇÃO (Salário) - Usando a sua função format_currency
            salary = candidate_data.get("remuneracao", 0) 
            st.write(f"**Salário/Remuneração:** {format_currency(salary)}")

            # contatos
            links = []
            if pd.notna(candidate_data.get("email", None)):
                links.append(f"📧 [Email Profissional]({'mailto:' + str(candidate_data.get('email'))})")
            if pd.notna(candidate_data.get("email_pessoal", None)): # Coluna adicional
                links.append(f"📧 [Email Pessoal]({'mailto:' + str(candidate_data.get('email_pessoal'))})")
            if pd.notna(candidate_data.get("linkedin", None)):
                 links.append(f"🔗 [LinkedIn]({candidate_data.get('linkedin')})")
            if links:
                 st.markdown(" | ".join(links), unsafe_allow_html=True) # Usamos markdown para links

        with col3:
            prob = float(candidate_data.get("probabilidade_match", 0.0))
            st.metric(label="Match", value=f"{prob:.1%}", delta=f"Rank #{rank}" if rank <= 3 else None)

        # Skills: extrai da vaga e destaca as que aparecem no CV
        vaga_text = str(vaga_row.get(VAGA_TEXT_COL, vaga_row.get("titulo_vaga", "")))
        target_skills = extract_skills(vaga_text)
        cv_text_norm = _norm(str(candidate_data.get(CV_TEXT_COL, "")))
        skills_hit = [s for s in target_skills if s in cv_text_norm]

        st.markdown("**Skills que batem com a vaga:**", help="Detectadas no texto da vaga e encontradas no CV")
        st.markdown(render_badges(skills_hit), unsafe_allow_html=True)

        # Explicabilidade: frases mais relevantes do CV para a vaga
        with st.expander("🔎 Por que este candidato apareceu aqui? (trechos mais relevantes do CV)"):
            # A chave de cache deve incluir o ID da vaga e do candidato, mas o top_relevant_sentences já faz isso!
            highlights = top_relevant_sentences(
                candidate_data.get(CANDIDATO_ID_COL, f"cand_{rank}"),
                vaga_row.get(VAGA_ID_COL, "vaga"),
                str(candidate_data.get(CV_TEXT_COL, "")),
                vaga_embedding,
                encoder,
                k=3
            )
            if highlights:
                for h in highlights:
                    st.write(f"• {h}")
            else:
                st.caption("Não foi possível extrair trechos relevantes.")

        # CV resumido
        with st.expander("📄 Ver CV Resumido"):
            cv_text = str(candidate_data.get(CV_TEXT_COL, ""))
            preview = cv_text[:1200] + ("..." if len(cv_text) > 1200 else "")
            st.text(preview)

def display_load_logs(log_messages: List[str]) -> bool:
    """Exibe os logs de carregamento de dados na sidebar."""
    with st.expander("Status de Carregamento de Dados", expanded=False):
        data_loaded_ok = True
        for msg in log_messages:
            if "✅" in msg:
                st.success(msg)
            elif "❌" in msg or "🚨" in msg:
                st.error(msg)
                data_loaded_ok = False
            else:
                st.info(msg)
        return data_loaded_ok

# ==============================================================================
# 7. FUNÇÕES DE PÁGINAS (PAINEL, ADMIN E MATCHING)
# ==============================================================================

# --- PÁGINA DE ADMIN ---
def page_admin(cdf: pd.DataFrame, vdf: pd.DataFrame):
    """Página para importação e adição manual de candidatos/vagas."""
    st.header("🛠️ Administração de Dados")
    st.markdown("Use esta seção para importar novos arquivos ou adicionar registros manualmente.")

    tab_upload, tab_manual = st.tabs(["📁 Importar Arquivo CSV", "✏️ Adicionar Manualmente"])

    # -----------------------------------------------------
    # TAB 1: UPLOAD DE ARQUIVO
    # -----------------------------------------------------
    with tab_upload:
        st.subheader("Substituir Bases de Dados no S3")
        st.warning("⚠️ **ATENÇÃO:** O upload substituirá os arquivos no S3 e forçará o recálculo dos embeddings.")

        data_type = st.radio("Tipo de Dados para Upload:", ["Candidatos", "Vagas"])

        uploaded_file = st.file_uploader(f"Selecione o arquivo CSV de {data_type.lower()}", type=["csv"])

        if uploaded_file and st.button(f"📥 Substituir {data_type} no S3", key="upload_s3"):
            try:
                # Carrega o novo DF com o encoding correto (latin-1)
                new_df = pd.read_csv(uploaded_file, encoding='latin-1', engine="python", on_bad_lines="skip")
                
                filename = CANDIDATOS_FILE if data_type == "Candidatos" else VAGAS_FILE

                if save_dataframe_to_s3(new_df, filename):
                    st.success(f"✅ Arquivo de {data_type} substituído no S3 com sucesso!")
                    st.info("🔄 Recarregando a aplicação para carregar a nova base de dados...")
                    time.sleep(1)
                    st.rerun()
                
            except Exception as e:
                st.error(f"❌ Erro ao processar o arquivo: {e}")
                st.info("Certifique-se de que o CSV possui as colunas essenciais e a codificação está correta (latin-1).")

    # -----------------------------------------------------
    # TAB 2: ADIÇÃO MANUAL
    # -----------------------------------------------------
    with tab_manual:
        st.subheader("Adicionar um único registro")
        add_type = st.radio("Tipo de Registro para Adicionar:", ["Candidato", "Vaga"])
        
        if add_type == "Candidato":
            st.markdown("Preencha os dados do novo candidato:")
            with st.form("new_candidate_form"):
                col_name, col_email = st.columns(2)
                nome = col_name.text_input("Nome Completo", key="new_cand_nome")
                email = col_email.text_input("Email Principal", key="new_cand_email")
                cv_text = st.text_area("Texto Completo do Currículo (CRÍTICO para o Match!)", height=250, key="new_cand_cv")
                
                col_meta1, col_meta2, col_meta3 = st.columns(3)
                remuneracao = col_meta1.number_input("Remuneração Almejada (R$)", min_value=0.0, step=100.0, key="new_cand_rem")
                escolaridade = col_meta2.selectbox("Escolaridade", cdf["escolaridade"].unique() if not cdf.empty and "escolaridade" in cdf.columns else ["Superior Completo", "Mestrado"], key="new_cand_esc")
                area_atuacao = col_meta3.text_input("Área de Atuação", key="new_cand_area")
                
                submitted = st.form_submit_button("➕ Adicionar Novo Candidato e Recarregar")

                if submitted:
                    if len(cv_text) < 50:
                        st.error("O texto do currículo deve ter no mínimo 50 caracteres.")
                    elif not nome or not cv_text:
                         st.error("Nome e Texto do Currículo são obrigatórios.")
                    else:
                        new_cand_data = {
                            "nome": nome,
                            "email": email,
                            CV_TEXT_COL: cv_text, # Coluna CV_TEXT_COL é crítica
                            "remuneracao": remuneracao,
                            "escolaridade": escolaridade,
                            "area_atuacao": area_atuacao,
                            CANDIDATO_ID_COL: "TEMP_ID",
                        }
                        
                        updated_cdf = add_new_data_point(cdf, new_cand_data, CANDIDATO_ID_COL, CV_TEXT_COL, id_prefix="cand")
                        
                        if save_dataframe_to_s3(updated_cdf, CANDIDATOS_FILE):
                            st.success(f"✅ Candidato **{nome}** adicionado e base atualizada no S3!")
                            st.info("🔄 Recarregando a aplicação para gerar os novos embeddings...")
                            time.sleep(1)
                            st.rerun()
        
        elif add_type == "Vaga":
            st.markdown("Preencha os dados da nova vaga:")
            with st.form("new_vaga_form"):
                titulo = st.text_input("Título da Vaga", key="new_vaga_titulo")
                objetivo = st.text_area("Objetivo / Descrição Curta", key="new_vaga_objetivo")
                atividades = st.text_area("Principais Atividades e Requisitos (CRÍTICO para o Match!)", height=250, key="new_vaga_atividades")
                
                col_vaga1, col_vaga2 = st.columns(2)
                nivel = col_vaga1.text_input("Nível Profissional (Ex: Senior)", key="new_vaga_nivel")
                competencias = col_vaga2.text_input("Competências/Habilidades Chave (separar por vírgula)", key="new_vaga_comp")

                submitted = st.form_submit_button("➕ Adicionar Nova Vaga e Recarregar")

                if submitted:
                    if not titulo or len(atividades) < 50:
                        st.error("Título e o campo 'Principais Atividades' (mín. 50 caracteres) são obrigatórios.")
                    else:
                        new_vaga_data = {
                            "titulo_vaga": titulo,
                            "objetivo_vaga": objetivo,
                            "principais_atividades": atividades,
                            "nivel_profissional": nivel,
                            "competencias": competencias,
                            VAGA_ID_COL: "TEMP_ID",
                        }
                        
                        updated_vdf = add_new_data_point(vdf, new_vaga_data, VAGA_ID_COL, VAGA_TEXT_COL, id_prefix="vaga")
                        
                        if save_dataframe_to_s3(updated_vdf, VAGAS_FILE):
                            st.success(f"✅ Vaga **{titulo}** adicionada e base atualizada no S3!")
                            st.info("🔄 Recarregando a aplicação para gerar os novos embeddings...")
                            time.sleep(1)
                            st.rerun()


# --- PÁGINA DE PAINEL ---
def page_dashboard(cdf: pd.DataFrame, vdf: pd.DataFrame):
    """Página de painel com estatísticas gerais sobre os dados."""
    st.header("📊 Painel de Estatísticas de Dados")
    
    col_c, col_v = st.columns(2)

    # CARD 1: Candidatos
    with col_c:
        st.metric("Total de Candidatos", f"{len(cdf):,}")
        
        # Distribuição de Escolaridade
        st.markdown("##### Top 7 Escolaridades")
        if not cdf.empty and "escolaridade" in cdf.columns:
            esc_counts = cdf["escolaridade"].value_counts().nlargest(7)
            esc_df = esc_counts.reset_index()
            esc_df.columns = ["Escolaridade", "Contagem"]
            st.bar_chart(esc_df, x="Escolaridade", y="Contagem")
        else:
            st.caption("Dados de escolaridade indisponíveis.")

    # CARD 2: Vagas
    with col_v:
        st.metric("Total de Vagas", f"{len(vdf):,}")

        # Top 5 Áreas de Atuação (Vagas)
        st.markdown("##### Top 5 Títulos de Vaga")
        if not vdf.empty and "titulo_vaga" in vdf.columns:
            title_counts = vdf["titulo_vaga"].value_counts().nlargest(5)
            title_df = title_counts.reset_index()
            title_df.columns = ["Título da Vaga", "Contagem"]
            st.bar_chart(title_df, x="Título da Vaga", y="Contagem")
        else:
             st.caption("Dados de vagas indisponíveis.")
             
    st.markdown("---")
    
    # Distribuição Geográfica
    st.subheader("Distribuição Geográfica de Candidatos (Top 10 Estados)")
    if "estado" in cdf.columns and "pais" in cdf.columns:
        br_cands = cdf[cdf["pais"].str.lower() == "brasil"]
        if not br_cands.empty:
            state_counts = br_cands["estado"].value_counts().nlargest(10).reset_index()
            state_counts.columns = ["Estado", "Candidatos"]
            st.dataframe(state_counts, use_container_width=True)
            
        else:
             st.caption("Dados geográficos insuficientes ou não-Brasil.")


# --- PÁGINA DE MATCHING (LÓGICA PRINCIPAL) ---
def page_matching(cdf, vdf, encoder, bst):
    """Lógica principal do Matching de Talentos (Seu código original)."""
    
    # --- Sidebar: Configurações Iniciais ---
    with st.sidebar:
        st.header("⚙️ Configurações do Match")
        
        # Controles de performance (repassados do main() original, mas agora aqui)
        max_candidates = st.slider("Nº Máximo de Candidatos a Carregar", 100, 100000, 5000, 100)
        top_k_for_xgboost = st.slider("Top K Candidatos para Predição XGBoost", 100, 5000, 1000, 100)
        use_cache = st.checkbox("Usar Cache de Embeddings", value=True)
        top_n = st.slider("Resultados por página", 5, 50, 15)
        
        vaga_options = vdf.set_index(VAGA_ID_COL)["titulo_vaga"].to_dict()
        selected_vaga_id = st.selectbox(
            "Selecione a Vaga:",
            options=list(vaga_options.keys()),
            format_func=lambda x: f"{x} - {vaga_options[x]}",
        )
        
        st.markdown("---")
        st.info(
            f"""
        **Estatísticas de Carga:**
        - 📊 {len(cdf):,} candidatos carregados
        - 💼 {len(vdf):,} vagas disponíveis
        - 🎯 exibindo {top_n} por página
        """
        )

    # --- Carregar/Gerar Embeddings ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Embeddings de Candidatos")
        candidate_embeddings = get_or_create_embeddings(cdf, CV_TEXT_COL, EMBEDDINGS_FILE, encoder, use_cache)
        
        content_hash_c = _hash_df(cdf, [CV_TEXT_COL], sample_rows=20000)
        cache_key_c = f"emb_{EMBEDDINGS_FILE}_{content_hash_c}"
        if use_cache and cache_key_c in st.session_state:
             st.success(f"✅ Embeddings Candidatos: {candidate_embeddings.shape} (Cache)")
        else:
             st.warning(f"🔄 Embeddings Candidatos gerados: {candidate_embeddings.shape} (Novo cálculo)")

    with col2:
        st.subheader("Embeddings de Vagas")
        vaga_embeddings = get_or_create_embeddings(vdf, VAGA_TEXT_COL, VAGAS_EMBEDDINGS_FILE, encoder, use_cache)
        
        content_hash_v = _hash_df(vdf, [VAGA_TEXT_COL], sample_rows=20000)
        cache_key_v = f"emb_{VAGAS_EMBEDDINGS_FILE}_{content_hash_v}"
        if use_cache and cache_key_v in st.session_state:
             st.success(f"✅ Embeddings Vagas: {vaga_embeddings.shape} (Cache)")
        else:
             st.warning(f"🔄 Embeddings Vagas gerados: {vaga_embeddings.shape} (Novo cálculo)")

    st.markdown("---")

    # --- Processar Matching ---
    if selected_vaga_id:
        vaga_row = vdf[vdf[VAGA_ID_COL] == selected_vaga_id].iloc[0]
        vaga_index = vdf[vdf[VAGA_ID_COL] == selected_vaga_id].index[0]
        vaga_embedding = vaga_embeddings[vaga_index]

        st.subheader(f"💼 Vaga Selecionada: {vaga_row['titulo_vaga']}")
        st.caption(vaga_row[VAGA_TEXT_COL][:200] + "...")

        # ⏳ Cálculo e Ranking
        with st.spinner(f"Calculando Match (Top {top_k_for_xgboost} candidatos)..."):
            results_df = predict_match_and_rank(
                vaga_embedding, 
                candidate_embeddings, 
                cdf, 
                bst, 
                top_k=top_k_for_xgboost
            )

        if results_df.empty:
            st.error("❌ Não foi possível gerar o ranking de candidatos.")
            return

        st.success(f"✅ Match calculado para {len(results_df)} candidatos.")

        # --- Paginação e Exibição ---
        total_results = len(results_df)
        total_pages = int(np.ceil(total_results / top_n))
        
        st.markdown(f"**Top {total_results} Resultados**")
        
        if total_pages > 1:
            page_cols = st.columns([1, 4, 1])
            current_page = page_cols[1].slider("Página", 1, total_pages, 1)
        else:
            current_page = 1

        start_idx = (current_page - 1) * top_n
        end_idx = min(start_idx + top_n, total_results)
        
        paginated_df = results_df.iloc[start_idx:end_idx]

        st.markdown(f"Exibindo **{start_idx + 1}** a **{end_idx}** (Total: {total_results})")

        # Exibir Cards de Candidatos
        for _, candidate_data in paginated_df.iterrows():
            display_candidate_card(
                candidate_data, 
                int(candidate_data['rank']), 
                vaga_row, 
                vaga_embedding, 
                encoder
            )


# ==============================================================================
# 8. EXECUÇÃO PRINCIPAL DO APLICATIVO
# ==============================================================================

def main():
    st.title("🎯 RECRUT.AI - Sistema de Match de Talentos")
    
    # Carregar dados e modelos (globalmente)
    try:
        # Load inicial com um limite para manter a velocidade no início
        cdf, vdf, log_messages = load_data(_max_rows=5000) 
        
        # Exibe logs de carregamento na sidebar
        with st.sidebar:
            st.header("🔐 Configurações e Status")
            st.info(f"**Bucket:** {S3_BUCKET}")
            st.info(f"**Região:** {os.environ.get('AWS_DEFAULT_REGION', 'N/A')}")
            data_ok = display_load_logs(log_messages)

        if not data_ok or cdf.empty or vdf.empty:
            st.error("🚨 Crítico: Falha no carregamento dos dados iniciais. Verifique os logs acima.")
            # Permite que o usuário acesse a página de Admin para upload, mas sem os dados
            # Apenas um DF vazio para não travar a função page_admin
            page_admin(pd.DataFrame(columns=[CANDIDATO_ID_COL, CV_TEXT_COL, "nome", "escolaridade"]), pd.DataFrame(columns=[VAGA_ID_COL, "titulo_vaga", VAGA_TEXT_COL]))
            return

        with st.spinner("🚀 Inicializando modelos de IA..."):
            bst = load_models()
            encoder = load_encoder()
            st.toast("✅ Modelos de IA carregados com sucesso!", icon="✅")

    except Exception as e:
        st.error(f"❌ Erro Crítico: Falha na inicialização da aplicação: {e}")
        return


    # --- Estrutura de Navegação (Tabs) ---
    tab_match, tab_dashboard, tab_admin = st.tabs(
        [
            "🔎 Match de Talentos", 
            "📊 Painel de Estatísticas", 
            "🛠️ Administração de Dados"
        ]
    )

    with tab_match:
        page_matching(cdf, vdf, encoder, bst)

    with tab_dashboard:
        page_dashboard(cdf, vdf)

    with tab_admin:
        # Passamos os DFs carregados para que o Admin possa fazer o append
        page_admin(cdf, vdf)


if __name__ == "__main__":
    main()
