# ======================== IMPORTS ========================
import os, re, json, hashlib, io
from pathlib import Path
from typing import List, Tuple, Optional
import csv
from typing import Optional 
import numpy as np
import pandas as pd
import streamlit as st
import requests
import joblib
from xgboost import XGBClassifier

# Imports tolerantes (para não derrubar o app se a lib não existir)
try:
    import psutil
except Exception:
    psutil = None

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

SAFE_BOOT = os.getenv("SAFE_BOOT", "true").lower() == "true"

# ======================== CONFIG BÁSICA / CONSTANTES ========================
APP_NAME = "RECRUT.AI 🚀"
APP_VERSION = "1.3.4 (Base Completa)"

# 1) A PRIMEIRA CHAMADA DE UI DO STREAMLIT DEVE SER set_page_config
st.set_page_config(
    page_title=APP_NAME,
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================== DETECÇÃO DE AMBIENTE ========================
IS_DEPLOY = os.getenv("IS_DEPLOY", "false").lower() == "true"

# ======================== CONFIGURAÇÕES INICIAIS (Lógica Pura) ========================

# Limiar padrão para "Aprovação" no ranking
DEFAULT_LIMIAR = float(os.getenv("SCORE_LIMIAR", "0.75"))

# Modelo: Pasta local do seu projeto (O seu sbert_encoder)
MODEL_DIR = os.getenv("MODEL_DIR", "./models/sbert_encoder")
MODEL_NAME = os.getenv("MODEL_NAME", f"Local_Custom:{MODEL_DIR}")
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Modelo XGBoost
XGB_MODEL_NAME = os.getenv("XGB_MODEL_NAME", "modelo_match_xgboost.pkl")
XGB_MODEL_PATH = Path(os.getenv("XGB_MODEL_PATH", "models")) / XGB_MODEL_NAME

# Dados CSV (Local e URLs)
BASE_CANDIDATOS_PATH = os.getenv("BASE_CANDIDATOS_PATH", "data/applicants_clean.csv")
BASE_VAGAS_PATH = os.getenv("BASE_VAGAS_PATH", "data/vagas_clean.csv")
CANDIDATOS_CSV_URL = os.getenv("CANDIDATOS_CSV_URL", "https://raw.githubusercontent.com/laribar/Datathon/main/data/applicants_clean.csv")
VAGAS_CSV_URL = os.getenv("VAGAS_CSV_URL", "https://raw.githubusercontent.com/laribar/Datathon/main/data/vagas_clean.csv")

# Cache de embeddings: Remoto (URLs RAW do GitHub - PRIORIDADE MÁXIMA)
CAND_EMB_URL = os.getenv("CAND_EMB_URL", "https://raw.githubusercontent.com/laribar/Datathon/main/data/embeddings/candidatos.npy")
CAND_META_URL = os.getenv("CAND_META_URL", "https://raw.githubusercontent.com/laribar/Datathon/main/data/embeddings/candidatos.meta.json")
VAGA_EMB_URL = os.getenv("VAGA_EMB_URL", "https://raw.githubusercontent.com/laribar/Datathon/main/data/embeddings/vagas.npy")
VAGA_META_URL = os.getenv("VAGA_META_URL", "https://raw.githubusercontent.com/laribar/Datathon/main/data/embeddings/vagas.meta.json")

# Cache de embeddings: Local (Fallback)
EMB_DIR = Path(os.getenv("EMB_DIR", "data/embeddings"))
CAND_EMB_PATH = EMB_DIR / "candidatos.npy"
CAND_META_PATH = EMB_DIR / "candidatos.meta.json"
VAGA_EMB_PATH = EMB_DIR / "vagas.npy"
VAGA_META_PATH = EMB_DIR / "vagas.meta.json"

# Colunas de metadados dos candidatos a exibir no ranking (AJUSTE CONFORME SEU CSV)
CAND_META_COLS = ["nome", "cidade", "experiencia", "skills", "id"]


# --- Funções de Inicialização de Sistema (Apenas Python, SEM Streamlit) ---
def setup_paths() -> str:
    """Configura caminhos para compatibilidade com deploy e cria diretórios."""
    base_path = (
        os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    )
    os.makedirs(os.path.join(base_path, "models"), exist_ok=True)
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    return base_path


def get_available_memory() -> float:
    """Retorna a memória disponível em GB (Apenas lógica Python)."""
    if psutil is None:
        return 0.0
    memory_info = psutil.virtual_memory()
    return memory_info.available / (1024**3)


# Chamadas de setup
BASE_PATH = setup_paths()
AVAILABLE_GB = get_available_memory()  # Checagem de memória feita antes da UI
CAN_PROCEED = AVAILABLE_GB >= 2

# Mensagem de modo deploy (depois do set_page_config)
if IS_DEPLOY:
    st.sidebar.info("🚀 Modo Deploy Ativo")


# ======================== TEXT UTILS ========================
_whitespace_re = re.compile(r"\s+")


def clean_text(t: str) -> str:
    """Limpa e normaliza texto (minúsculas e remoção de espaços extras)."""
    if t is None:
        return ""
    t = t.strip().lower()
    return _whitespace_re.sub(" ", t)


def proportional_score(sim: float, limiar: float) -> float:
    """Calcula o score de 0 a 100 baseado no limiar de aprovação."""
    if limiar <= 0:
        return 0.0
    if sim >= limiar:
        return 100.0
    return max(0.0, (sim / limiar) * 100.0)


def _concat_all_columns(df: pd.DataFrame, new_col_name: str) -> pd.DataFrame:
    """Combina colunas de texto em uma única coluna 'new_col_name', tratando erros de colunas vazias."""
    df = df.copy()

    # 1. Identifica colunas a serem excluídas (exceto o próprio ID)
    cols_to_exclude = {new_col_name, "indice_origem", "versao"}

    # 2. Seleciona TODAS as outras colunas para concatenação
    text_cols = [col for col in df.columns if col not in cols_to_exclude]

    if not text_cols:
        # ❗ TRATAMENTO DE ERRO
        df[new_col_name] = ""
        st.warning(
            f"⚠️ A função de concatenação ({new_col_name}) não encontrou colunas válidas no DataFrame. Verifique a estrutura do CSV."
        )
        return df

    # 3. Combina colunas: Converte para string e concatena
    df[new_col_name] = df[text_cols].astype(str).agg(" ".join, axis=1)

    # 4. Limpeza
    df[new_col_name] = df[new_col_name].str.strip()
    df[new_col_name] = df[new_col_name].apply(clean_text)

    return df


# ======================== DATA LOADERS ========================
@st.cache_data(show_spinner=False)
def _read_csv_local_or_url(local_path: str, url_env: Optional[str]) -> Optional[pd.DataFrame]:
    """Carrega CSV da URL (Prioridade) ou do disco local (Fallback)."""
    READ_CSV_PARAMS = {
        "sep": ",",
        "encoding": "utf-8",
        "on_bad_lines": "skip",
        "engine": "python",
        "quoting": csv.QUOTE_MINIMAL,
        "skipinitialspace": True,
    }

    if url_env:
        try:
            df = pd.read_csv(url_env, **READ_CSV_PARAMS)
            st.info(f"🔗 Dados lidos da URL: {url_env} ({len(df)} registros)", icon="✅")
            return df
        except Exception:
            pass

    try:
        if os.path.exists(local_path):
            df = pd.read_csv(local_path, **READ_CSV_PARAMS)
            st.info(f"📂 Dados lidos do disco local: {local_path} ({len(df)} registros)", icon="✅")
            return df
    except Exception:
        pass

    return None


@st.cache_data(show_spinner="Carregando bases de dados...", ttl=None)
def load_fixed_bases() -> Tuple[pd.DataFrame, pd.DataFrame, list]:
    """Carrega as bases de candidatos e vagas e concatena colunas de texto."""
    logs = []

    cand = _read_csv_local_or_url(BASE_CANDIDATOS_PATH, CANDIDATOS_CSV_URL)
    vaga = _read_csv_local_or_url(BASE_VAGAS_PATH, VAGAS_CSV_URL)

    # Lógica de fallback para DataFrame de amostra
    if cand is None or cand.empty:
        logs.append("⚠️ Não encontrei candidatos via URL ou local. Usando amostra.")
        cand = pd.DataFrame(
            {
                "nome": ["Ana Silva", "Carlos Souza"],
                "skills": ["Python Airflow Spark", "Java Spring SQL"],
                "experiencia": ["3 anos em dados", "5 anos em backend"],
                "cidade": ["São Paulo", "Rio de Janeiro"],
                "id": [1, 2],
            }
        )
    else:
        logs.append(f"✅ Candidatos carregados: {len(cand)} registros.")

    if vaga is None or vaga.empty:
        logs.append("⚠️ Não encontrei vagas via URL ou local. Usando amostra.")
        vaga = pd.DataFrame(
            {
                "titulo_vaga": ["Engenheira de Dados Senior", "Desenvolvedor Backend Java"],
                "requisitos": ["Python, Spark, Airflow, AWS", "Java, Spring Boot, SQL, REST APIs"],
                "descricao": [
                    "Projetos de dados em ambiente cloud. Criação de pipelines ETL.",
                    "Desenvolvimento de microserviços de alta performance.",
                ],
            }
        )
    else:
        logs.append(f"✅ Vagas carregadas: {len(vaga)} registros.")

    # Concatena colunas de texto para o SBERT
    cand = _concat_all_columns(cand, "cv_text")
    vaga = _concat_all_columns(vaga, "vaga_text")

    logs.append(f"✅ Bases processadas: {len(cand)} candidatos e {len(vaga)} vagas.")

    return cand, vaga, logs


# ======================== EMBEDDING CACHE UTILS ========================
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
        st.info(f"💾 Embeddings salvos no disco: {npy_path.name}", icon="💾")
    except Exception:
        # Silencia a falha em ambientes efêmeros (e.g., Streamlit Cloud)
        st.warning(
            f"⚠️ Falha ao salvar cache no disco ({npy_path.name}). Isso é normal em alguns deploys.",
            icon="⚠️",
        )


def _load_embeddings_url(
    npy_url: Optional[str], meta_url: Optional[str]
) -> Tuple[Optional[np.ndarray], dict]:
    """Tenta carregar embeddings via URL RAW do GitHub."""
    if not npy_url or not meta_url:
        return None, {}
    try:
        # Carrega o arquivo .meta.json
        meta_response = requests.get(meta_url, timeout=5)
        if meta_response.status_code != 200:
            return None, {}
        meta = json.loads(meta_response.text)

        # Carrega o arquivo .npy (binário)
        npy_response = requests.get(npy_url, timeout=20)
        if npy_response.status_code != 200:
            return None, {}

        embs = np.load(io.BytesIO(npy_response.content))
        st.info(f"⚡ Cache de embeddings carregado via URL: {npy_url}", icon="⚡")
        return embs, meta
    except Exception:
        return None, {}


def _load_embeddings_local(npy_path: Path, meta_path: Path) -> Tuple[Optional[np.ndarray], dict]:
    """Tenta carregar embeddings do disco local (Fallback)."""
    if not npy_path.exists() or not meta_path.exists():
        return None, {}
    try:
        arr = np.load(npy_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        st.info(f"💾 Cache de embeddings carregado do disco: {npy_path.name}", icon="💾")
        return arr, meta
    except Exception:
        return None, {}


# ❗ CORREÇÃO CRÍTICA AQUI: O model_source no metadado sempre usa MODEL_DIR,
# e a flag "FORCE_REBUILD" é usada apenas para controle de fluxo interno.
def get_or_build_embeddings(df: pd.DataFrame, text_col: str, model_dir_or_flag: str) -> np.ndarray:
    """Carrega do cache (URL > Local) ou constrói (lento) os embeddings."""
    if text_col == "cv_text":
        npy_path, meta_path, npy_url, meta_url = (
            CAND_EMB_PATH,
            CAND_META_PATH,
            CAND_EMB_URL,
            CAND_META_URL,
        )
    else:
        npy_path, meta_path, npy_url, meta_url = (
            VAGA_EMB_PATH,
            VAGA_META_PATH,
            VAGA_EMB_URL,
            VAGA_META_URL,
        )

    # Metadados esperados (usa o MODEL_DIR real, não a flag de rebuild)
    sig = _hash_dataframe(df[[text_col]])
    meta_expected = {
        "model_source": MODEL_DIR,
        "text_col": text_col,
        "signature": sig,
        "version": APP_VERSION,
    }

    def _is_valid(embs: Optional[np.ndarray], meta: dict, df_: pd.DataFrame) -> bool:
        """Verifica se o cache é válido."""
        # Se a flag de força-rebuild for passada, o cache é inválido.
        if model_dir_or_flag == "FORCE_REBUILD":
            return False

        # Checa se o conteúdo do meta.json e o tamanho do NPY batem
        return embs is not None and meta == meta_expected and embs.ndim == 2 and embs.shape[0] == len(df_)

    # 1. Tenta carregar do cache remoto (URL) - PRIORIDADE MÁXIMA
    embs, meta = _load_embeddings_url(npy_url, meta_url)
    if _is_valid(embs, meta, df):
        return embs  # type: ignore

    # 2. Tenta carregar do cache local
    embs, meta = _load_embeddings_local(npy_path, meta_path)
    if _is_valid(embs, meta, df):
        return embs  # type: ignore

    # 3. Se falhar ou não existir, constrói (Lento)
    st.warning(
        f"⏳ Reconstruindo embeddings para '{text_col}' (Não há cache válido). Isso pode levar alguns minutos...",
        icon="⏳",
    )
    texts = df[text_col].astype(str).tolist()

    embs = embed_texts(texts, MODEL_DIR)

    _save_embeddings(npy_path, meta_path, embs, meta_expected)  # Tenta salvar localmente para próxima vez
    st.success(f"✅ Embeddings gerados para '{text_col}'.", icon="✅")
    return embs


# ======================== MODEL / ENCODER ========================
@st.cache_resource(show_spinner="Carregando Modelo de Ranqueamento (XGBoost)...", ttl=None)
def load_xgb_model(model_path: Path) -> Optional["XGBClassifier"]: # type: ignore
    """Carrega o modelo XGBoost treinado a partir de um arquivo .pkl."""
    # evita erro de import no type checker
    global XGBClassifier
    try:
        from xgboost import XGBClassifier
    except ImportError:
        XGBClassifier = None

    if XGBClassifier is None:
        st.warning("⚠️ XGBoost indisponível. Ranqueando por Similaridade de Cosseno.")
        return None
    if not model_path.exists():
        st.error(f"❌ Falha crítica: Modelo XGBoost não encontrado em: {model_path}")
        return None

    try:
        model = joblib.load(model_path)
        st.success(f"✅ Modelo XGBoost carregado de: **{model_path}**", icon="🧠")
        return model
    except Exception as e:
        st.error(f"❌ Falha ao carregar modelo XGBoost: {e}")
        return None

@st.cache_resource(show_spinner="Carregando Encoder (Priorizando Modelo Local)...", ttl=None)
def load_model(model_path: str):
    """Carrega o SentenceTransformer, priorizando o modelo local."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        st.error("❌ Falha crítica: A biblioteca 'sentence-transformers' não está instalada.")
        st.stop()

    try:
        model = SentenceTransformer(model_path)
        return model
    except Exception:
        # Tenta fallback para o modelo Hugging Face
        try:
            model = SentenceTransformer(HF_MODEL_NAME)
            st.warning("⚠️ Modelo padrão do Hugging Face carregado como fallback.", icon="⚠️")
            return model
        except Exception:
            st.error("❌ Falha crítica: Não foi possível carregar o modelo de nenhuma fonte.")
            st.stop()


@st.cache_data(show_spinner="Gerando embeddings em lote...", ttl=3600, max_entries=5)
def embed_texts(texts: List[str], model_path: str) -> np.ndarray:
    """Gera embeddings para uma lista de textos."""
    model = load_model(model_path)
    texts = [clean_text(t) for t in texts]
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)


@st.cache_data(show_spinner="Gerando embedding...", ttl=3600, max_entries=20)
def embed_text(text: str, model_path: str) -> np.ndarray:
    """Gera o embedding para um único texto, garantindo que o resultado seja 1D."""
    model = load_model(model_path)
    text_clean = clean_text(text)
    emb = model.encode(text_clean, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
    return emb.flatten()


def generate_xgb_features(vaga_emb: np.ndarray, cv_embs: np.ndarray) -> np.ndarray:
    """
    Gera o array de features NxF para o modelo XGBoost.
    """
    if vaga_emb.ndim == 1:
        N = cv_embs.shape[0]
        vaga_embs_repeated = np.repeat(vaga_emb[np.newaxis, :], N, axis=0)
    else:
        vaga_embs_repeated = vaga_emb

    features = np.hstack(
        [vaga_embs_repeated, cv_embs, np.abs(vaga_embs_repeated - cv_embs), vaga_embs_repeated * cv_embs]
    )
    return features


# ======================== LÓGICA DE INICIALIZAÇÃO DO APP ========================
with st.spinner("Carregando modelos (SBERT e XGBoost)..."):
    load_model(MODEL_DIR)  # Carrega e cacheia o SBERT
    xgb_ranking_model = load_xgb_model(XGB_MODEL_PATH)
    st.session_state["xgb_ranking_model"] = xgb_ranking_model

# 3. Carrega Bases Fixas (e concatena o texto)
candidatos_df, vagas_df, _logs = load_fixed_bases()

# Guarda bases na session_state
st.session_state["candidatos_df"] = candidatos_df.copy()
st.session_state["vagas_df"] = vagas_df.copy()

# 4. Tenta carregar embeddings de base OU recalcula
if "cache_loaded" not in st.session_state or not st.session_state["cache_loaded"]:
    st.session_state["cache_loaded"] = False

    if _logs:
        with st.expander("Logs de Carregamento de Bases", expanded=False):
            for m in _logs:
                st.caption(m)

    with st.spinner("⚡ Preparando e carregando embeddings iniciais (Cache ou Reconstrução)..."):
        try:
            # Tenta carregar ou construir os embeddings dos candidatos
            cand_embs_cache = get_or_build_embeddings(st.session_state["candidatos_df"], "cv_text", MODEL_DIR)
            vaga_embs_cache = get_or_build_embeddings(st.session_state["vagas_df"], "vaga_text", MODEL_DIR)

            if cand_embs_cache is not None and vaga_embs_cache is not None:
                st.session_state["cache_loaded"] = True
                st.session_state["cand_embs_cache"] = cand_embs_cache
                st.session_state["vaga_embs_cache"] = vaga_embs_cache
            else:
                st.error("Não foi possível obter os embeddings para as bases fixas.")
                st.session_state["cache_loaded"] = False
                st.session_state["cand_embs_cache"] = None
                st.session_state["vaga_embs_cache"] = None

        except Exception as e:
            st.error(f"❌ Falha crítica ao carregar/gerar embeddings. Aplicativo interrompido. Erro: {e}")
            st.session_state["cache_loaded"] = False
            st.session_state["cand_embs_cache"] = None
            st.session_state["vaga_embs_cache"] = None
            st.stop()  # PARADA CRÍTICA: Impede que o Streamlit continue sem os embeddings


# ======================== SIDEBAR E UI PRINCIPAL ========================
with st.sidebar:
    st.markdown(f"## {APP_NAME} 🤖")
    st.caption(f"Versão {APP_VERSION}")
    st.write("---")

    # Informações de Memória
    st.subheader("🛠️ Status do Sistema")
    st.write(f"💾 Memória disponível: {AVAILABLE_GB:.1f} GB")
    if not CAN_PROCEED:
        st.warning("⚠️ Memória limitada detectada. Algumas funcionalidades podem ser desativadas.")

    st.divider()
    st.markdown("#### Configurações Globais")
    st.write("**Encoder:**", MODEL_NAME)
    limiar = st.slider("Limiar de Aprovação (Cosine)", 0.50, 0.95, DEFAULT_LIMIAR, 0.01)

    st.divider()
    st.markdown("#### 🔄 Cache de Embeddings")
    cache_loaded = st.session_state["cache_loaded"]
    st.info(f"Status do Cache: {'✅ Disponível' if cache_loaded else '❌ Indisponível'}")

    if st.button("⚡ Gerar/Atualizar Cache de Base", help="Força a reconstrução e tentativa de salvamento do cache local."):
        with st.spinner("Gerando e salvando cache (candidatos e vagas)…"):
            # Usa a flag "FORCE_REBUILD" para forçar o recálculo
            st.session_state["cand_embs_cache"] = get_or_build_embeddings(
                st.session_state["candidatos_df"], "cv_text", "FORCE_REBUILD"
            )
            st.session_state["vaga_embs_cache"] = get_or_build_embeddings(
                st.session_state["vagas_df"], "vaga_text", "FORCE_REBUILD"
            )

            st.session_state["cache_loaded"] = True

            st.success(f"Cache gerado. **Recarregue a página (F5)** para usar o novo cache na inicialização.")

# DEBUG: Informações de debug na sidebar
st.sidebar.write("---")
st.sidebar.subheader("📊 Debug Info")
st.sidebar.write(f"Candidatos: {len(st.session_state['candidatos_df'])} registros")
st.sidebar.write(f"Vagas: {len(st.session_state['vagas_df'])} registros")
st.sidebar.write(f"XGBoost Status: {'✅ Pronto' if xgb_ranking_model else '❌ Ausente'}")

# Exibir logs detalhados
with st.sidebar.expander("Logs de Carregamento"):
    for log in _logs:
        st.sidebar.text(log)


# -------------------- UI Principal --------------------
st.title("🔎 RECRUT.AI - Match Semântico (Especialização)")
st.markdown(f"Análise de similaridade entre Curricula e Vagas usando **Sentence-BERT** (Modelo: **{MODEL_DIR}**)")

tab_ranking, tab_bases = st.tabs(["📊 Ranking por Vaga", "📋 Bases de Dados"])


# ############## TAB 1: Ranking por Vaga (Base Fixa) ##############
with tab_ranking:
    st.header("Ranking de Candidatos por Vaga (N×1)")
    st.caption("Compara todos os candidatos da base fixa contra a vaga selecionada, utilizando embeddings.")

    vdf = st.session_state["vagas_df"]
    cdf = st.session_state["candidatos_df"]

    def _vaga_label(row: pd.Series) -> str:
        title = row.get("titulo_vaga") or ""
        vt = str(row.get("vaga_text", ""))
        base_txt = title.strip() or (vt[:80] + ("…" if len(vt) > 80 else ""))
        return f"({row.name}) {base_txt}"

    if len(vdf) > 0 and len(cdf) > 0:
        col_sel, col_limpeza = st.columns([3, 1])
        with col_sel:
            options = vdf.apply(_vaga_label, axis=1).tolist()
            idx = st.selectbox(
                "Selecione a vaga para ranquear",
                options=range(len(options)),
                format_func=lambda i: options[i],
                key="sel_vaga_ranking",
            )

        vaga_text_sel = str(vdf.iloc[idx]["vaga_text"])

        with col_limpeza:
            clean_rank = st.checkbox("Aplicar limpeza nos textos (Recalcula embeddings)", not cache_loaded, key="clean_ranking")
            is_using_cache = cache_loaded and not clean_rank
            st.caption(f"Cache de Embeddings: {'✅ Ativo (Base)' if is_using_cache else '❌ Inativo (Recálculo)'}")

        with st.expander("Ver descrição completa da vaga"):
            st.write(vaga_text_sel)

        # --- Controles de Ranking ---
        col_controles = st.columns([1, 4])
        with col_controles[0]:
            max_topn = len(cdf)
            default_topn = min(50, max_topn)
            top_n = st.number_input("Top N Candidatos", 1, max_topn, default_topn, key="topn_ranking")

        with col_controles[1]:
            if st.button("🔍 GERAR RANKING", key="btn_ranking", use_container_width=True):
                with st.spinner("Calculando ranking..."):

                    # 1. Embeddings: Prioriza cache ou recalcula
                    if is_using_cache:
                        emb_cvs = st.session_state["cand_embs_cache"]
                        emb_vaga = st.session_state["vaga_embs_cache"][idx]
                    else:
                        # Recálculo forçado
                        cvs = cdf["cv_text"].astype(str).tolist()
                        emb_cvs = embed_texts(cvs, MODEL_DIR)
                        emb_vaga = embed_text(vaga_text_sel, MODEL_DIR)

                    # 2. CÁLCULO DO SCORE
                    xgb_model = st.session_state.get("xgb_ranking_model")

                    if xgb_model is not None:
                        st.info("🧠 Usando modelo **XGBoost** para Ranqueamento por **Probabilidade de Match**.")

                        features = generate_xgb_features(emb_vaga, emb_cvs)
                        probs = xgb_model.predict_proba(features)[:, 1]

                        scores_array = probs
                        limiar_aprovacao = 0.50

                        # Similaridade de Cosseno (via produto interno pois vetores estão normalizados)
                        sims = emb_cvs @ emb_vaga.T

                    else:
                        st.warning("⚠️ Modelo XGBoost indisponível. Usando **Similaridade de Cosseno** para ranqueamento.")

                        sims = emb_cvs @ emb_vaga.T
                        scores_array = sims
                        limiar_aprovacao = limiar  # Usa o limiar do slider

                    # 3. Ordenação e Top N
                    order = np.argsort(-scores_array)[:int(top_n)]

                    # 4. Geração do DataFrame de Resultados
                    rows = []
                    for rank, i in enumerate(order, start=1):
                        main_score = float(scores_array[i])
                        cossine_sim = float(sims[i])

                        if xgb_model is not None:
                            score_porcentagem = round(main_score * 100, 2)
                            aprovado = bool(main_score >= limiar_aprovacao)
                        else:
                            score_porcentagem = proportional_score(main_score, limiar_aprovacao)
                            aprovado = bool(main_score >= limiar_aprovacao)

                        row = {
                            "Rank": rank,
                            "Similaridade": round(cossine_sim, 6),
                            "Score (%)": round(score_porcentagem, 2),
                            "Aprovado": aprovado,
                        }

                        for col in CAND_META_COLS:
                            if col in cdf.columns:
                                val = cdf.iloc[i][col]
                                if col in ["experiencia", "skills"] and isinstance(val, str):
                                    row[col] = val[:100] + "..." if len(val) > 100 else val
                                else:
                                    row[col] = val

                        rows.append(row)

                    res = pd.DataFrame(rows)
                    st.success(f"🎉 Ranking gerado: Top {len(res)} candidatos.")

                    # --- Visualização de Resultados ---
                    col_config = {
                        "Similaridade": st.column_config.ProgressColumn(
                            "Similaridade (Cosseno)", format="%.4f", min_value=0.0, max_value=1.0
                        ),
                        "Score (%)": st.column_config.ProgressColumn("Score Principal (%)", format="%f", min_value=0, max_value=100),
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
                    with col1:
                        st.metric("Candidatos Aprovados", aprovados)
                    with col2:
                        st.metric("Taxa de Aprovação", f"{aprovados/len(res)*100:.1f}%")
                    with col3:
                        st.metric("Melhor Score", f"{res.iloc[0]['Score (%)']:.2f}%")

                    st.download_button(
                        "💾 Baixar Ranking (CSV)",
                        res.to_csv(index=False).encode("utf-8"),
                        file_name="ranking_por_vaga.csv",
                        mime="text/csv",
                        key="dl_ranking",
                    )
    else:
        st.warning("Nenhuma vaga ou candidato disponível na base de dados carregada para ranking.")


# ############## TAB 2: Bases de Dados ##############
with tab_bases:
    st.header("Visualização das Bases de Dados Carregadas")

    if not cdf.empty:
        with st.expander(f"Candidatos ({len(cdf)} registros)", expanded=True):
            st.dataframe(cdf, use_container_width=True)
            st.caption("Colunas combinadas para embedding: **cv_text**")
            st.download_button(
                "Baixar CSV Candidatos",
                cdf.to_csv(index=False).encode("utf-8"),
                file_name="candidatos_processado.csv",
                mime="text/csv",
                key="dl_cand",
            )

    if not vdf.empty:
        with st.expander(f"Vagas ({len(vdf)} registros)", expanded=True):
            st.dataframe(vdf, use_container_width=True)
            st.caption("Colunas combinadas para embedding: **vaga_text**")
            st.download_button(
                "Baixar CSV Vagas",
                vdf.to_csv(index=False).encode("utf-8"),
                file_name="vagas_processado.csv",
                mime="text/csv",
                key="dl_vaga",
            )
