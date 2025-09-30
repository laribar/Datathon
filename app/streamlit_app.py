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
import plotly.express as px
from wordcloud import WordCloud # Pode precisar de 'pip install wordcloud'
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

# Colunas de metadados dos candidatos para exibição na UI
CANDIDATO_METADATA_COLS = [
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
    "nome",
    "endereco",
    "nivel_academico",
    "remuneracao",
    "local",
    "email_pessoal",
    "genero"
]

# Limite de linhas a carregar na inicialização para evitar timeouts no Streamlit Cloud
MAX_ROWS_INITIAL_LOAD = 10000
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
        # A codificação mais comum de erro em CSV brasileiro é latin-1 lendo utf-8
        return text.encode('latin-1').decode('utf-8', 'ignore')
    except:
        return text

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
    # Divide por ponto, exclamação, interrogação seguidos de espaço ou por quebra de linha
    parts = re.split(r'(?<=[\.\!\?])\s+|\n+', text.strip())
    # Filtra partes muito curtas para evitar ruído
    return [p.strip() for p in parts if len(p.strip()) > 20][:60]

@st.cache_data(hash_funcs={SentenceTransformer: lambda _: None}, show_spinner=False)
def top_relevant_sentences(
    candidate_id: Any,
    vaga_id: Any,
    cv_text: str,
    vaga_embedding: np.ndarray,
    encoder: SentenceTransformer,
    k: int = 3
) -> List[str]:
    """Retorna as k sentenças do CV mais próximas do embedding da vaga (explicabilidade)."""
    sents = split_sentences(cv_text)
    if not sents:
        return []
    
    # Reduzindo o batch_size e garantindo dtype float32
    sent_emb = encoder.encode(sents, show_progress_bar=False, convert_to_numpy=True, batch_size=32).astype("float32")
    sent_emb = _l2_normalize(sent_emb)
    
    # Produto escalar entre a matriz de sentenças e o vetor da vaga
    scores = (sent_emb @ vaga_embedding.reshape(-1)).reshape(-1) 
    k = min(k, len(sents))
    # Encontra os índices dos top K scores
    top_idx = np.argpartition(scores, -k)[-k:]
    # Ordena os top K em ordem decrescente
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return [sents[i] for i in top_idx]

def render_badges(items: list) -> str:
    """Renders a list of items as a series of stylized HTML badges."""
    return " ".join([f"<span style='display:inline-block;padding:2px 8px;border-radius:999px;background:#1f6feb20;border:1px solid #1f6feb55;margin:2px 6px 2px 0;font-size:12px'>{html.escape(i)}</span>" for i in items[:12]])

# --- 🎯 INJEÇÃO DE SECRETS DO STREAMLIT CLOUD ---
if "aws" in st.secrets:
    try:
        access_key = st.secrets["aws"].get("AWS_ACCESS_KEY_ID") or st.secrets["aws"]["aws_access_key_id"]
        secret_key = st.secrets["aws"].get("AWS_SECRET_ACCESS_KEY") or st.secrets["aws"]["aws_secret_access_key"]
        aws_region = st.secrets["aws"].get("AWS_REGION") or st.secrets["aws"].get("AWS_DEFAULT_REGION") or st.secrets["aws"].get("region_name")

        os.environ["AWS_ACCESS_KEY_ID"] = access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = secret_key
        os.environ["AWS_DEFAULT_REGION"] = aws_region

        logger.info(f"Credenciais AWS carregadas via st.secrets. Região: {aws_region}")
    except KeyError as e:
        logger.error(f"Erro: Segredo AWS faltando. Chave não encontrada: {e}.")
        st.error("❌ Segredo AWS faltando. Verifique se as chaves (ID, SECRET, REGION) estão no formato [aws] correto.")

# ==============================================================================
# 3. FUNÇÕES UTILITÁRIAS DE HASH/NORMALIZAÇÃO E CACHE/LOAD
# ==============================================================================

def _hash_df(df: pd.DataFrame, cols: List[str], sample_rows: int = 0) -> str:
    """Gera um hash leve e estável baseado no conteúdo das colunas indicadas."""
    if not set(cols).issubset(df.columns):
        return f"missing_cols_{hash(tuple(cols))}"
    if sample_rows and len(df) > sample_rows:
        df = df.sample(sample_rows, random_state=42)
    s = pd.util.hash_pandas_object(df[cols].astype(str).fillna(""), index=False).values
    return str(int(s[: min(2000, len(s))].sum())) if len(s) else "empty"

def _l2_normalize(M: np.ndarray) -> np.ndarray:
    """Normalização L2 (unitária) de cada vetor na matriz."""
    n = np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    return M / n

@st.cache_resource(show_spinner=False)
def get_s3_fs():
    """Retorna o filesystem do S3 com configuração correta."""
    try:
        if not os.environ.get("AWS_ACCESS_KEY_ID"):
             raise RuntimeError("Credenciais AWS não configuradas. Verifique st.secrets.")

        fs = s3fs.S3FileSystem(anon=False)
        return fs
    except Exception as e:
        raise RuntimeError(f"Erro de conexão com S3: {e}. Verifique as credenciais AWS.")

@st.cache_resource(show_spinner="Carregando modelo XGBoost do S3...")
def load_models() -> Any:
    """Carrega o modelo XGBoost do S3."""
    fs = get_s3_fs()
    model_s3_path = f"{S3_BUCKET}/data/models/{MODEL_FILE}"

    if not fs.exists(model_s3_path):
        raise FileNotFoundError(f"Arquivo do modelo XGBoost não encontrado: s3://{S3_BUCKET}/data/models/{MODEL_FILE}")

    with fs.open(model_s3_path, "rb") as f:
        bst = joblib.load(f)

    if bst is None:
        raise ValueError("Modelo XGBoost está vazio")

    return bst

@st.cache_resource(show_spinner="Carregando Sentence Transformer...")
def load_encoder(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Carrega o modelo SBERT. Tenta S3; se não existir, tenta cache local; se falhar, baixa da internet."""
    
    # 1. Tentar S3
    try:
        fs = get_s3_fs()
        sbert_s3_path = f"{S3_BUCKET}/{SBERT_MODEL_DIR}"
        test_file_path = f"{sbert_s3_path}/config.json"
        local_cache_path = os.path.join(tempfile.gettempdir(), SBERT_MODEL_DIR)

        if fs.exists(test_file_path):
            if not os.path.exists(local_cache_path): 
                logger.info("Modelo SBERT encontrado em S3. Baixando para cache local...")
                fs.get(sbert_s3_path, local_cache_path, recursive=True)
            
            encoder = SentenceTransformer(local_cache_path)
            _ = encoder.encode(["probe"], convert_to_numpy=True)
            return encoder
        else:
            logger.warning(f"Modelo SBERT não encontrado em S3. Baixando {model_name}...")
            encoder = SentenceTransformer(model_name)
            _ = encoder.encode(["probe"], convert_to_numpy=True)
            return encoder
            
    except Exception as e:
        logger.error(f"Falha ao carregar SBERT (S3 ou cache): {e}. Tentando baixar de huggingface...")
        try:
            encoder = SentenceTransformer(model_name)
            _ = encoder.encode(["probe"], convert_to_numpy=True)
            return encoder
        except Exception as e_local:
            raise RuntimeError(f"Falha crítica ao carregar SBERT de todas as fontes: {e_local}")
            

@st.cache_data(show_spinner="Carregando dados dos candidatos e vagas do S3...", ttl=900)
def load_data(_max_rows: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Carrega os DataFrames de candidatos e vagas do S3. LIMITA as linhas APENAS se _max_rows for setado (SÓ PARA CANDIDATOS)."""
    log_messages: List[str] = []
    cdf = pd.DataFrame()
    vdf = pd.DataFrame()

    # --- Candidatos (Aplicar limite de linhas aqui) ---
    try:
        fs = get_s3_fs()
        candidatos_s3_path = f"{S3_BUCKET}/data/{CANDIDATOS_FILE}"
        
        with fs.open(candidatos_s3_path, "rb") as f:
            cdf = pd.read_csv(f, nrows=_max_rows, encoding="latin-1", engine="python", on_bad_lines="skip")

        critical_cols = [CANDIDATO_ID_COL, CV_TEXT_COL]
        cdf = cdf.dropna(subset=critical_cols)
        cdf[CV_TEXT_COL] = cdf[CV_TEXT_COL].astype(str).apply(_decode_text)
        
        cols_to_keep = [col for col in ([CANDIDATO_ID_COL, CV_TEXT_COL] + CANDIDATO_METADATA_COLS) if col in cdf.columns]
        cdf = cdf[cols_to_keep].reset_index(drop=True) 

        log_messages.append(f"✅ Candidatos carregados: {len(cdf):,} registros ({'total' if _max_rows is None else f'máx {_max_rows}'})")

    except Exception as e:
        log_messages.append(f"❌ Erro ao carregar candidatos: {str(e)}")
        cdf = pd.DataFrame()

    # --- Vagas (NUNCA APLICAR LIMITE aqui para garantir que todas as vagas sejam sempre carregadas) ---
    try:
        fs = get_s3_fs()
        vagas_s3_path = f"{S3_BUCKET}/data/{VAGAS_FILE}"

        with fs.open(vagas_s3_path, "rb") as f:
            vdf = pd.read_csv(f, encoding="latin-1")

        text_cols_to_combine = ["titulo_vaga", "objetivo_vaga", "nivel_profissional", "principais_atividades", "competencias", "habilidades_comportamentais"]
        existing_text_cols = [col for col in text_cols_to_combine if col in vdf.columns]
        
        for col in existing_text_cols:
            vdf[col] = vdf[col].apply(_decode_text)

        # CRÍTICO: Cria a coluna combinada para ser usada no embedding
        vdf[VAGA_TEXT_COL] = vdf[existing_text_cols].fillna("").astype(str).agg(" ".join, axis=1)

        # Garante que o ID da vaga é uma string limpa e sem espaços
        if VAGA_ID_COL in vdf.columns:
            vdf[VAGA_ID_COL] = vdf[VAGA_ID_COL].astype(str).str.strip()
        
        vdf = vdf.dropna(subset=[VAGA_TEXT_COL, VAGA_ID_COL]).reset_index(drop=True)
        log_messages.append(f"✅ Vagas carregadas: {len(vdf):,} registros")

    except Exception as e:
        log_messages.append(f"❌ Erro ao carregar vagas: {str(e)}")
        vdf = pd.DataFrame()

    if cdf.empty or vdf.empty:
        log_messages.append("🚨 Crítico: Dados insuficientes para continuar.")

    return cdf, vdf, log_messages

@st.cache_data(
    show_spinner="Gerenciando cache de embeddings...",
    ttl=3600,
    hash_funcs={SentenceTransformer: lambda _: None}, 
)
def get_or_create_embeddings(
    df: pd.DataFrame,
    text_col: str,
    filename: str,
    encoder: SentenceTransformer,
    _use_cache: bool = True,
) -> np.ndarray:
    """Gerencia cache e cria embeddings (SBERT) para DataFrame."""
    if df.empty:
        return np.zeros((0, encoder.get_sentence_embedding_dimension()), dtype="float32")

    content_hash = _hash_df(df, [text_col], sample_rows=20000)
    cache_key = f"emb_{filename}_{content_hash}" 

    # 1. Tentar cache de sessão/memória do Streamlit
    if _use_cache and cache_key in st.session_state:
        logger.info(f"✅ Embeddings carregados do cache de sessão: {filename}")
        return st.session_state[cache_key]

    # 2. Tentar S3
    fs = get_s3_fs()
    s3_emb_path = f"{S3_BUCKET}/data/embeddings/{filename}"

    if _use_cache and fs.exists(s3_emb_path):
        with st.spinner(f"☁️ Carregando embeddings do S3: {filename}"):
            with fs.open(s3_emb_path, "rb") as f:
                embeddings = np.load(f)
        
        # Garante que o número de linhas é consistente
        if embeddings.shape[0] == len(df):
            embeddings = _l2_normalize(embeddings.astype("float32"))
            st.session_state[cache_key] = embeddings
            logger.info(f"✅ Embeddings carregados do S3: {filename}")
            return embeddings
        else:
            logger.warning(f"Embeddings do S3 ({filename}) com tamanho inconsistente. Recalculando.")

    # 3. Gerar novos embeddings (com barra de progresso)
    texts = df[text_col].astype(str).tolist()
    
    with st.spinner(f"🧠 Criando embeddings para {len(texts):,} registros. Pode demorar..."):
        batch_size = 64
        all_embeddings: List[np.ndarray] = []
        
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            
            batch_embeddings = encoder.encode(batch_texts, show_progress_bar=False, convert_to_numpy=True, batch_size=32).astype("float32")
            all_embeddings.append(batch_embeddings)

            if i % (batch_size * 10) == 0:  
                progress = min((i + batch_size) / len(texts), 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Processando: {min(i + batch_size, len(texts)):,} / {len(texts):,}")

    embeddings = _l2_normalize(np.vstack(all_embeddings).astype("float32"))
    progress_bar.empty()
    status_text.empty()
    logger.info(f"✅ Embeddings criados: {filename}")

    # 4. Salvar no S3 (Garante consistência para a próxima execução)
    try:
        with st.spinner("💾 Salvando embeddings no S3..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as tmp:
                np.save(tmp.name, embeddings)
                tmp_path = tmp.name
            fs.put(tmp_path, s3_emb_path)
            os.unlink(tmp_path)
        logger.info(f"✅ Embeddings salvos no S3: {filename}")
    except Exception as e:
        logger.warning(f"Falha ao salvar embeddings no S3 ({filename}): {e}")

    # 5. Salvar no cache de sessão
    st.session_state[cache_key] = embeddings
    return embeddings

# ==============================================================================
# 4. FUNÇÕES DE PREDIÇÃO E RANKING (POTENCIAL)
# ==============================================================================

def predict_match_and_rank(
    vaga_embedding: np.ndarray,
    all_candidate_embeddings: np.ndarray,
    cdf: pd.DataFrame,
    bst: Any, 
    top_k: int = 1000,
) -> pd.DataFrame:
    # ... (código omitido)

    # 2. Construção da Matriz de Predição
    X_left = all_candidate_embeddings[top_idx] # (K, 384)
    X_right = np.broadcast_to(vaga_embedding, X_left.shape) # (K, 384)
    
    # X_predict_base tem 768 features (384 + 384)
    X_predict_base = np.hstack([X_left, X_right]).astype(np.float32, copy=False)
    
    # 💥 CORREÇÃO: Duplica o vetor de entrada para atingir 1536 features
    # (768 + 768)
    X_predict = np.hstack([X_predict_base, X_predict_base]).astype(np.float32, copy=False)
    
    # 3. Predição
    try:
        if isinstance(bst, xgb.Booster):
            dtest = xgb.DMatrix(X_predict)
            predictions = bst.predict(dtest)
        else:
            if hasattr(bst, "predict_proba"):
                proba = bst.predict_proba(X_predict)
                predictions = proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba.ravel()
            else:
                predictions = bst.predict(X_predict).astype(np.float32, copy=False)

    except Exception as e:
        st.error(f"Erro ao gerar predições com o modelo XGBoost: {e}")
        return pd.DataFrame()

    # 4. Formatação e Ranking (Potencial = Probabilidade de Match)
    results_df = cdf_safe.iloc[top_idx].copy()
    results_df["probabilidade_match"] = predictions.astype(np.float32, copy=False)
    results_df = results_df.sort_values("probabilidade_match", ascending=False).reset_index(drop=True)
    results_df["rank"] = np.arange(1, len(results_df) + 1, dtype=int)
    
    return results_df

# ==============================================================================
# 5. FUNÇÕES DE PERSISTÊNCIA E INSERÇÃO DE DADOS
# ==============================================================================

def save_dataframe_to_s3(df: pd.DataFrame, filename: str, s3_dir: str = S3_DATA_PATH) -> bool:
    """Salva um DataFrame no S3, sobrescrevendo o arquivo existente."""
    try:
        fs = get_s3_fs()
        s3_path = f"{s3_dir.rstrip('/')}/{filename}"

        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False, encoding='utf-8') 
        
        with fs.open(s3_path, 'w', encoding='utf-8') as f:
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
    
    new_id = f"{id_prefix}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    new_row_series = pd.Series(new_data)
    new_row_series[id_col] = new_id
    
    df_temp = pd.DataFrame(columns=df.columns)
    new_row_final = new_row_series.reindex(df.columns).fillna(pd.NA)
    df_temp.loc[0] = new_row_final
    
    df_temp[text_col] = df_temp[text_col].apply(_decode_text)

    updated_df = pd.concat([df, df_temp], ignore_index=True)
    
    # Invalida o cache de DADOS e EMBEDDINGS para forçar a recarga
    st.cache_data.clear() 
    
    return updated_df

# ==============================================================================
# 6. FUNÇÕES AUXILIARES PARA UI
# ==============================================================================

def format_currency(value: Any) -> str:
    """Formata valor para o padrão monetário brasileiro."""
    try:
        val = float(value) if pd.notna(value) else 0.0
        return f"R$ {val:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "R$ 0,00"
    
def safe_display(value, default_str="N/A"):
    """Retorna o valor ou o default se for nulo/vazio."""
    if pd.isna(value) or value is None or str(value).lower() in ('nan', 'n/a', ''):
        return default_str
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(int(value)) if int(value) == value else f"{value:.2f}"
    return str(value)

def display_candidate_card(candidate_data: pd.Series, rank: int, vaga_row: pd.Series, vaga_embedding: np.ndarray, encoder: SentenceTransformer):
    with st.container(border=True):
        col1, col2, col3 = st.columns([3, 2, 1])

        with col1:
            nome_display = safe_display(candidate_data.get("nome", candidate_data.get(CANDIDATO_ID_COL, 'Candidato N/A')))
            st.subheader(f"#{rank} - {nome_display}")
            
            endereco_display = safe_display(candidate_data.get("endereco", "Localidade N/A"))
            st.write(f"**Endereço:** {endereco_display}")

            linha1 = []
            if pd.notna(candidate_data.get("local", None)):
                linha1.append(str(candidate_data.get("local")))
            elif pd.notna(candidate_data.get("cidade", None)):
                linha1.append(str(candidate_data.get("cidade")))
            st.caption(" | ".join(linha1) if linha1 else "Local/Cidade: N/A")

            st.write(f"**Escolaridade:** {safe_display(candidate_data.get('nivel_academico', 'Nível Acadêmico N/A'))}")
            if pd.notna(candidate_data.get("tempo_experiencia", None)):
                 st.write(f"**Experiência:** {candidate_data.get('tempo_experiencia')}")

        with col2:
            st.write(f"**Gênero:** {safe_display(candidate_data.get('genero', 'N/A'))}")
            st.write(f"**Área:** {safe_display(candidate_data.get('area_atuacao', 'N/A'))}")
            st.write(f"**Última experiência:** {safe_display(candidate_data.get('ultima_experiencia', 'N/A'))}")
            
            salary = candidate_data.get("remuneracao", 0) 
            st.write(f"**Salário/Remuneração:** {format_currency(salary)}")

            links = []
            if pd.notna(candidate_data.get("email", None)):
                links.append(f"📧 [Email Profissional]({'mailto:' + str(candidate_data.get('email'))})")
            if pd.notna(candidate_data.get("linkedin", None)):
                 links.append(f"🔗 [LinkedIn]({candidate_data.get('linkedin')})")
            if links:
                 st.markdown(" | ".join(links), unsafe_allow_html=True)

        with col3:
            prob = float(candidate_data.get("probabilidade_match", 0.0))
            st.metric(label="Match", value=f"{prob:.1%}", delta=f"Rank #{rank}" if rank <= 3 else None)

        # Skills
        vaga_text = str(vaga_row.get(VAGA_TEXT_COL, vaga_row.get("titulo_vaga", "")))
        target_skills = extract_skills(vaga_text)
        cv_text_norm = _norm(str(candidate_data.get(CV_TEXT_COL, "")))
        skills_hit = [s for s in target_skills if s in cv_text_norm]

        st.markdown("**Skills que batem com a vaga:**", help="Detectadas no texto da vaga e encontradas no CV")
        st.markdown(render_badges(skills_hit), unsafe_allow_html=True)

        # Explicabilidade
        with st.expander("🔎 Por que este candidato apareceu aqui? (trechos mais relevantes do CV)"):
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
    st.markdown("Use esta seção para **importar novos arquivos CSV** ou **adicionar registros manualmente** e garantir a consistência da base de dados no S3.")

    tab_upload, tab_manual = st.tabs(["📁 Importar Arquivo CSV (Substituir)", "✏️ Adicionar Manualmente"])

    # -----------------------------------------------------
    # TAB 1: UPLOAD DE ARQUIVO
    # -----------------------------------------------------
    with tab_upload:
        st.subheader("Substituir Bases de Dados no S3")
        st.warning("⚠️ **ATENÇÃO:** O upload **substituirá** a base inteira no S3 e forçará o recálculo dos embeddings. Mantenha as colunas originais!")

        data_type = st.radio("Tipo de Dados para Upload:", ["Candidatos", "Vagas"], key="upload_type_radio")

        uploaded_file = st.file_uploader(f"Selecione o arquivo CSV de {data_type.lower()}", type=["csv"], key="csv_uploader")

        if uploaded_file and st.button(f"📥 Substituir {data_type} no S3 e Recarregar", key="upload_s3"):
            try:
                try:
                    new_df = pd.read_csv(uploaded_file, encoding='latin-1', engine="python", on_bad_lines="skip")
                except:
                    uploaded_file.seek(0)
                    new_df = pd.read_csv(uploaded_file, encoding='utf-8', engine="python", on_bad_lines="skip")
                    
                filename = CANDIDATOS_FILE if data_type == "Candidatos" else VAGAS_FILE

                if save_dataframe_to_s3(new_df, filename):
                    st.success(f"✅ Arquivo de {data_type} substituído no S3 com sucesso! Recarregando...")
                    time.sleep(1)
                    st.rerun()
                
            except Exception as e:
                st.error(f"❌ Erro ao processar o arquivo: {e}")
                st.info("Verifique a codificação (latin-1 ou utf-8) e as colunas.")

    # -----------------------------------------------------
    # TAB 2: ADIÇÃO MANUAL
    # -----------------------------------------------------
    with tab_manual:
        st.subheader("Adicionar um único registro e Atualizar o S3")
        add_type = st.radio("Tipo de Registro para Adicionar:", ["Candidato", "Vaga"], key="add_type_radio")
        
        if add_type == "Candidato":
            st.markdown("Preencha os dados do novo **Candidato**:")
            with st.form("new_candidate_form"):
                col_name, col_email = st.columns(2)
                nome = col_name.text_input("Nome Completo", key="new_cand_nome")
                email = col_email.text_input("Email Principal", key="new_cand_email")
                cv_text = st.text_area("**Texto Completo do Currículo (CRÍTICO para o Match!)**", height=250, key="new_cand_cv")
                
                col_meta1, col_meta2, col_meta3 = st.columns(3)
                remuneracao = col_meta1.number_input("Remuneração Almejada (R$)", min_value=0.0, step=100.0, key="new_cand_rem")
                escolaridade_options = cdf["escolaridade"].unique() if not cdf.empty and "escolaridade" in cdf.columns else ["Superior Completo", "Mestrado"]
                escolaridade = col_meta2.selectbox("Escolaridade", list(escolaridade_options), key="new_cand_esc")
                area_atuacao = col_meta3.text_input("Área de Atuação", key="new_cand_area")
                
                submitted = st.form_submit_button("➕ Adicionar Novo Candidato e Recarregar")
                
                if submitted and cv_text and nome:
                    new_data = {
                        "nome": nome,
                        "email": email,
                        "remuneracao": remuneracao,
                        "escolaridade": escolaridade,
                        "area_atuacao": area_atuacao,
                        CV_TEXT_COL: cv_text,
                        CANDIDATO_ID_COL: None 
                    }
                    
                    updated_cdf = add_new_data_point(cdf.drop(columns=[CV_TEXT_COL], errors='ignore'), new_data, CANDIDATO_ID_COL, CV_TEXT_COL, id_prefix="custom_cand")
                    
                    if save_dataframe_to_s3(updated_cdf, CANDIDATOS_FILE):
                        st.success(f"✅ Candidato '{nome}' adicionado e base de candidatos salva no S3! Recarregando...")
                        time.sleep(1)
                        st.rerun()
                elif submitted:
                    st.error("❌ Por favor, preencha o Nome e o Currículo para adicionar o candidato.")
                    
        elif add_type == "Vaga":
            st.markdown("Preencha os dados da nova **Vaga**:")
            with st.form("new_vaga_form"):
                titulo = st.text_input("Título da Vaga", key="new_vaga_titulo")
                objetivo = st.text_area("Objetivo/Descrição Resumida", height=100, key="new_vaga_obj")
                atividades = st.text_area("Principais Atividades, Competências e Habilidades (CRÍTICO para o Match!)", height=250, key="new_vaga_ativ")
                
                submitted = st.form_submit_button("➕ Adicionar Nova Vaga e Recarregar")
                
                if submitted and titulo and atividades:
                    new_data = {
                        "titulo_vaga": titulo,
                        "objetivo_vaga": objetivo,
                        "principais_atividades": atividades,
                        VAGA_TEXT_COL: f"{titulo} {objetivo} {atividades}",
                        VAGA_ID_COL: None
                    }
                    
                    temp_vdf_cols = list(set(vdf.columns) | set(new_data.keys()))
                    temp_vdf = vdf.reindex(columns=temp_vdf_cols).copy()
                    
                    updated_vdf = add_new_data_point(temp_vdf, new_data, VAGA_ID_COL, VAGA_TEXT_COL, id_prefix="custom_vaga")
                    
                    if save_dataframe_to_s3(updated_vdf, VAGAS_FILE):
                        st.success(f"✅ Vaga '{titulo}' adicionada e base de vagas salva no S3! Recarregando...")
                        time.sleep(1)
                        st.rerun()
                elif submitted:
                    st.error("❌ Por favor, preencha o Título da Vaga e as Atividades/Habilidades para adicionar a vaga.")


# --- PÁGINA DE DASHBOARD (Visualização) ---
def page_dashboard(cdf: pd.DataFrame, vdf: pd.DataFrame):
    """Página de visualização geral dos dados."""
    st.header("📊 Dashboard de Dados")
    st.markdown("Visão geral sobre a base de **Candidatos** e **Vagas** carregadas.")

    if cdf.empty or vdf.empty:
        st.warning("⚠️ Dados de Candidatos ou Vagas não carregados. Verifique o Status de Carregamento na barra lateral.")
        return

    # -----------------------------------------------------
    # ESTATÍSTICAS GERAIS
    # -----------------------------------------------------
    st.subheader("Estatísticas Chave")
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total de Candidatos", f"{len(cdf):,}")
    col2.metric("Total de Vagas", f"{len(vdf):,}")
    
    if "area_atuacao" in cdf.columns:
        col3.metric("Áreas de Atuação (Candidatos)", f"{cdf['area_atuacao'].nunique():,} únicas")
    if "cidade" in vdf.columns:
        col4.metric("Cidades com Vagas", f"{vdf['cidade'].nunique():,} únicas")


    # -----------------------------------------------------
    # CANDIDATOS: Distribuição por Escolaridade
    # -----------------------------------------------------
    st.markdown("---")
    st.subheader("Distribuição dos Candidatos")
    
    col_c1, col_c2 = st.columns(2)

    if "escolaridade" in cdf.columns:
        escolaridade_counts = cdf["escolaridade"].value_counts().reset_index()
        escolaridade_counts.columns = ["Escolaridade", "Contagem"]
        
        fig_esc = px.bar(
            escolaridade_counts,
            x="Escolaridade",
            y="Contagem",
            title="Escolaridade dos Candidatos",
            color="Contagem",
            color_continuous_scale=px.colors.sequential.Plotly3,
        )
        fig_esc.update_layout(xaxis_title="", yaxis_title="")
        col_c1.plotly_chart(fig_esc, use_container_width=True)

    # -----------------------------------------------------
    # CANDIDATOS: Nuvem de Palavras (Áreas/Skills)
    # -----------------------------------------------------
    if CV_TEXT_COL in cdf.columns:
        text = " ".join(cdf[CV_TEXT_COL].dropna().astype(str).sample(min(1000, len(cdf))).tolist())
        
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color="white", 
            max_words=100,
            stopwords=set(["de", "e", "a", "o", "que", "do", "da", "em", "um", "uma", "para", "com", "os", "as", "você", "se", "no", "na"])
        ).generate(text)
        
        fig_wc = px.imshow(wordcloud.to_array(), title="Nuvem de Palavras nos Currículos (Amostra)")
        fig_wc.update_layout(xaxis_visible=False, yaxis_visible=False, coloraxis_showscale=False)
        col_c2.plotly_chart(fig_wc, use_container_width=True)

    # -----------------------------------------------------
    # VAGAS: Distribuição por Área (Título)
    # -----------------------------------------------------
    st.markdown("---")
    st.subheader("Distribuição das Vagas")

    col_v1, col_v2 = st.columns(2)
    
    if "titulo_vaga" in vdf.columns:
        vdf['titulo_limpo'] = vdf['titulo_vaga'].astype(str).str.split().str[:2].str.join(' ')
        area_counts = vdf['titulo_limpo'].value_counts().nlargest(15).reset_index()
        area_counts.columns = ["Título Simplificado", "Contagem"]
        
        fig_area = px.bar(
            area_counts,
            x="Contagem",
            y="Título Simplificado",
            orientation='h',
            title="Top 15 Títulos de Vaga Simplificados",
            color="Contagem",
            color_continuous_scale=px.colors.sequential.Viridis,
        )
        fig_area.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis_title="", yaxis_title="")
        col_v1.plotly_chart(fig_area, use_container_width=True)

    # -----------------------------------------------------
    # VAGAS: Distribuição por Local (Estado/Região)
    # -----------------------------------------------------
    if "estado" in vdf.columns:
        location_counts = vdf["estado"].value_counts().nlargest(10).reset_index()
        location_counts.columns = ["Estado", "Contagem"]

        fig_loc = px.pie(
            location_counts,
            names="Estado",
            values="Contagem",
            title="Distribuição Top 10 Estados (Vagas)",
        )
        col_v2.plotly_chart(fig_loc, use_container_width=True)

# --- PÁGINA DE MATCHING ---
def page_matching(cdf: pd.DataFrame, vdf: pd.DataFrame, encoder: SentenceTransformer, bst: Any):
    """Página principal de match crítico."""
    
    if cdf.empty or vdf.empty or encoder is None or bst is None:
        st.error("🚨 Dados ou Modelos não carregados. Verifique o 'Status de Carregamento' e recarregue a aplicação.")
        return

    st.header("🎯 Match Crítico: Vaga -> Candidatos")
    st.info(f"Modelo de Embeddings: **all-MiniLM-L6-v2**")
    
    # 1. Seleção da Vaga
    vagas_display = [f"{r['titulo_vaga']} (ID:{r[VAGA_ID_COL]})" for _, r in vdf.iterrows()]
    vagas_display.insert(0, "-- Selecione uma Vaga --")
    selected_vaga_display = st.selectbox(
        "Selecione uma Vaga da Base de Dados:",
        options=vagas_display,
        index=0,
        key="vaga_select"
    )

    if selected_vaga_display == "-- Selecione uma Vaga --":
        st.info("Aguardando seleção de vaga para iniciar o matching.")
        return

    # 2. Extração do ID da Vaga Selecionada (CORREÇÃO APLICADA AQUI)
    match = re.search(r'ID:([^)]+)\)', selected_vaga_display)
    
    if not match:
        st.error(f"❌ Erro de extração do ID da vaga. Formato esperado: Título (ID:XXX)")
        st.caption(f"Valor lido: {selected_vaga_display}")
        return
        
    vaga_id_match = match.group(1).strip()
    
    # 3. Busca da Vaga no DataFrame
    vdf_filtered = vdf[vdf[VAGA_ID_COL].astype(str).str.strip() == vaga_id_match]

    if vdf_filtered.empty:
        st.error(f"❌ A vaga com ID '{vaga_id_match}' não foi encontrada na base de dados de vagas carregada. Recarregue os dados ou selecione outra vaga.")
        return

    vaga_row = vdf_filtered.iloc[0]
    vaga_index = vaga_row.name # Indice original no vdf

    st.subheader(f"Vaga Selecionada: {vaga_row.get('titulo_vaga', 'N/A')}")
    st.markdown(f"**ID:** {vaga_id_match} | **Objetivo:** {safe_display(vaga_row.get('objetivo_vaga', 'N/A'))}")
    st.caption(f"Texto para Matching: {vaga_row[VAGA_TEXT_COL][:200]}...")


    # 4. Gerenciamento e Carregamento dos Embeddings
    with st.spinner("Preparando embeddings (Candidatos e Vaga)..."):
        c_emb = get_or_create_embeddings(cdf, CV_TEXT_COL, EMBEDDINGS_FILE, encoder)
        v_emb_all = get_or_create_embeddings(vdf, VAGA_TEXT_COL, VAGAS_EMBEDDINGS_FILE, encoder)
        
        if vaga_index >= len(v_emb_all):
            st.error("Erro: O índice da vaga selecionada não corresponde ao array de embeddings de vagas. Tente recarregar os dados.")
            return

        vaga_embedding = v_emb_all[vaga_index]
        
        if c_emb.size == 0 or vaga_embedding.size == 0:
            st.error("Embeddings não disponíveis. Não é possível calcular o match.")
            return

    # 5. Cálculo do Match e Ranking
    top_k_candidates = st.sidebar.slider("Quantos candidatos ranquear?", 10, 500, 100)
    
    with st.spinner(f"Calculando Match (XGBoost) para Top {top_k_candidates} candidatos..."):
        results_df = predict_match_and_rank(
            vaga_embedding,
            c_emb,
            cdf,
            bst,
            top_k=top_k_candidates
        )
    
    st.markdown("---")
    
    if results_df.empty:
        st.warning("Nenhum candidato encontrado no Top K para esta vaga.")
        return

    # 6. Exibição do Ranking
    st.subheader(f"🏆 Top {len(results_df)} Candidatos para a Vaga")
    
    n_display = st.slider("Mostrar Top N Candidatos:", 5, min(100, len(results_df)), 10)
    
    for rank, (idx, row) in enumerate(results_df.head(n_display).iterrows(), 1):
        display_candidate_card(row, rank, vaga_row, vaga_embedding, encoder)
        

# ==============================================================================
# 8. FUNÇÃO PRINCIPAL
# ==============================================================================

def main():
    # Cria uma imagem placeholder para Streamlit Cloud se não existir
    if not os.path.exists("logo.png"):
        try:
            from PIL import Image, ImageDraw
            img = Image.new('RGB', (100, 100), color = (7, 54, 66))
            d = ImageDraw.Draw(img)
            d.text((10,10), "LOGO", fill=(255, 255, 255))
            img.save("logo.png")
        except:
             pass 

    st.sidebar.image("logo.png", use_column_width=True)
    st.sidebar.title("Navegação")
    
    # 1. Carregamento de Dados
    cdf, vdf, log_messages = load_data(MAX_ROWS_INITIAL_LOAD)
    
    # 2. Carregamento de Modelos
    bst = None
    encoder = None
    
    data_loaded_ok = display_load_logs(log_messages)

    if data_loaded_ok and not cdf.empty and not vdf.empty:
        try:
            bst = load_models()
            encoder = load_encoder()
            st.sidebar.success("✅ Modelos carregados com sucesso.")
        except Exception as e:
            st.sidebar.error(f"❌ Erro ao carregar modelos: {e}")
    else:
        st.sidebar.warning("⚠️ Carregamento de dados falhou ou resultou em bases vazias.")
        
    # 3. Definição da Página
    
    if cdf.empty or vdf.empty or bst is None or encoder is None:
        page = st.sidebar.radio(
            "Selecione a Página (Acesso restrito)", 
            ["🛠️ Administração", "📊 Dashboard"], 
            index=0,
        )
    else:
        page = st.sidebar.radio(
            "Selecione a Página", 
            ["🎯 Match Crítico", "📊 Dashboard", "🛠️ Administração"], 
            index=0,
        )

    # 4. Renderização da Página
    if page == "🎯 Match Crítico":
        page_matching(cdf, vdf, encoder, bst)
    elif page == "📊 Dashboard":
        page_dashboard(cdf, vdf)
    elif page == "🛠️ Administração":
        page_admin(cdf, vdf)
    
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Última atualização: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
