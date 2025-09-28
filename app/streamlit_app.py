import os, re, json, hashlib, io 
from pathlib import Path
from typing import List, Tuple
# Importações essenciais para o modelo de ranqueamento
import joblib 
import io 
from xgboost import XGBClassifier 
import numpy as np
import pandas as pd
import streamlit as st
import requests 
from scipy.spatial.distance import cosine # Mantida por consistência

# ======================== CONFIG ========================
APP_NAME = "RECRUT.AI 🚀"
APP_VERSION = "1.3.4 (Base Completa)" # Versão atualizada

# Limiar padrão para "Aprovação" no ranking
DEFAULT_LIMIAR = float(os.getenv("SCORE_LIMIAR", "0.75"))

# Modelo: Pasta local do seu projeto (O seu sbert_encoder)
MODEL_DIR = os.getenv("MODEL_DIR", "models/sbert_encoder")
MODEL_NAME = os.getenv("MODEL_NAME", f"Local_Custom:{MODEL_DIR}") 
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") 

# Dados CSV (Local e URLs)
BASE_CANDIDATOS_PATH = os.getenv("BASE_CANDIDATOS_PATH", "data/applicants_clean.csv")
BASE_VAGAS_PATH = os.getenv("BASE_VAGAS_PATH", "data/vagas_clean.csv")
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

def _concat_all_columns(df: pd.DataFrame, new_col_name: str) -> pd.DataFrame:
    """
    CORREÇÃO: Esta função estava definida, mas era reportada como 'not defined'
    no escopo global. A ordem de definição do Python é estrita.
    A definição AQUI, antes da `load_fixed_bases`, corrige o erro de escopo.
    """
    df = df.copy()
    
    # Exclui a nova coluna e colunas não textuais/não essenciais se necessário
    cols_to_exclude = {new_col_name, 'id', 'indice_origem', 'versao'} # Adicione mais colunas não textuais aqui se tiver
    text_cols = [col for col in df.columns if col not in cols_to_exclude]
    
    # Combina todas as colunas de texto em uma única string, separando por espaço
    df[new_col_name] = df[text_cols].astype(str).agg(' '.join, axis=1)
    
    # Remove espaços extras e aplica clean_text
    df[new_col_name] = df[new_col_name].str.strip()
    df[new_col_name] = df[new_col_name].apply(clean_text)
    
    return df

# ======================== DATA LOADERS ========================

@st.cache_data(show_spinner=False)
def _read_csv_local_or_url(local_path: str, url_env: str | None) -> pd.DataFrame | None:
    """Carrega CSV da URL (Prioridade) ou do disco local (Fallback)."""
    
    # ❗ CORREÇÃO 1: Parâmetros robustos para o seu CSV completo. 
    # Mudar 'sep: None' para ',' e 'quoting: 3' (QUOTE_NONE) para 0 (QUOTE_MINIMAL)
    READ_CSV_PARAMS = {
        'sep': ',',             
        'encoding': 'utf-8', 
        'on_bad_lines': 'skip',
        'engine': 'python',
        'quoting': 0, 
        'skipinitialspace': True
    }

    # 1. Tenta URL (Prioridade - Para deploy Cloud)
    if url_env:
        try:
            r = requests.head(url_env, timeout=10)
            if r.status_code == 200:
                df = pd.read_csv(url_env, **READ_CSV_PARAMS)
                st.info(f"🔗 Dados lidos da URL: {url_env} ({len(df)} registros)")
                return df
        except Exception as e:
            st.warning(f"⚠️ Falha ao carregar URL {url_env}: {e}")
            pass
            
    # 2. Tenta caminho local (Fallback - Para teste local)
    try:
        if os.path.exists(local_path): 
            df = pd.read_csv(local_path, **READ_CSV_PARAMS)
            st.info(f"📂 Dados lidos do disco local: {local_path} ({len(df)} registros)")
            return df
    except Exception as e:
        st.warning(f"⚠️ Falha ao carregar arquivo local {local_path}: {e}")
        pass
        
    return None
    
@st.cache_data(show_spinner="Carregando bases de dados...", ttl=None)
def load_fixed_bases() -> Tuple[pd.DataFrame, pd.DataFrame, list]:
    """Carrega as bases de candidatos e vagas e concatena colunas de texto."""
    logs = []
    
    # As funções auxiliares _read_csv_local_or_url e _concat_all_columns agora estão definidas
    cand = _read_csv_local_or_url(BASE_CANDIDATOS_PATH, CANDIDATOS_CSV_URL) 
    vaga = _read_csv_local_or_url(BASE_VAGAS_PATH, VAGAS_CSV_URL)

    # Verificação mais detalhada dos dados carregados
    if cand is None or cand.empty:
        logs.append("⚠️ Não encontrei candidatos via URL ou local. Usando amostra.")
        # Se for a amostra, certifique-se de que a coluna 'id' exista
        cand = pd.DataFrame({
            "nome": ["Ana Silva", "Carlos Souza"],
            "skills": ["Python Airflow Spark", "Java Spring SQL"],
            "experiencia": ["3 anos em dados", "5 anos em backend"],
            "cidade": ["São Paulo", "Rio de Janeiro"],
            "id": [1, 2]
        })
    else:
        logs.append(f"✅ Candidatos carregados: {len(cand)} registros, {len(cand.columns)} colunas")
        logs.append(f"   Colunas: {list(cand.columns)}")
        
    if vaga is None or vaga.empty:
        logs.append("⚠️ Não encontrei vagas via URL ou local. Usando amostra.")
        # Se for a amostra, certifique-se de que 'titulo_vaga' exista (ou mapeie)
        vaga = pd.DataFrame({
            "titulo_vaga": ["Engenheira de Dados Senior", "Desenvolvedor Backend Java"], # Nome corrigido
            "requisitos": ["Python, Spark, Airflow, AWS", "Java, Spring Boot, SQL, REST APIs"],
            "descricao": ["Projetos de dados em ambiente cloud. Criação de pipelines ETL.", "Desenvolvimento de microserviços de alta performance."]
        })
    else:
        logs.append(f"✅ Vagas carregadas: {len(vaga)} registros, {len(vaga.columns)} colunas")
        logs.append(f"   Colunas: {list(vaga.columns)}")
        
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

XGB_MODEL_NAME = os.getenv("XGB_MODEL_NAME", "modelo_match_xgboost.pkl")
XGB_MODEL_PATH = Path(os.getenv("XGB_MODEL_PATH", "models")) / XGB_MODEL_NAME

@st.cache_resource(show_spinner="Carregando Modelo de Ranqueamento (XGBoost)...", ttl=None)
def load_xgb_model(model_path: Path) -> XGBClassifier | None:
    """Carrega o modelo XGBoost treinado a partir de um arquivo .pkl."""
    # ❗ CORREÇÃO 2: Adiciona a verificação de importação do joblib para evitar erro 'no module named joblib'
    try:
        import joblib
    except ImportError:
        st.error("❌ Falha crítica: A biblioteca 'joblib' não está instalada.")
        return None
        
    if not model_path.exists():
        st.error(f"❌ Falha crítica: Modelo XGBoost não encontrado em: {model_path}")
        return None
    try:
        model = joblib.load(model_path)
        st.success(f"✅ Modelo XGBoost carregado de: **{model_path}**")
        return model
    except Exception as e:
        st.error(f"❌ Falha ao carregar modelo XGBoost: {e}")
        return None


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
    # **IMPORTANTE**: Normaliza os embeddings
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True)

@st.cache_data(show_spinner="Gerando embedding...", ttl=3600, max_entries=20)
def embed_text(text: str, model_path: str) -> np.ndarray:
    """Gera o embedding para um único texto, garantindo que o resultado seja 1D."""
    model = load_model(model_path)
    text_clean = clean_text(text)
    # **IMPORTANTE**: Normaliza os embeddings
    emb = model.encode(text_clean, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True)
    return emb.flatten()

def generate_xgb_features(vaga_emb: np.ndarray, cv_embs: np.ndarray) -> np.ndarray:
    """
    Gera o array de features NxF para o modelo XGBoost, replicando a lógica de treino:
    [vaga_emb, cv_embs, |vaga_emb - cv_embs|, vaga_emb * cv_embs]
    
    vaga_emb: (D,) - Embedding da vaga (1D)
    cv_embs: (N, D) - Embeddings dos candidatos (2D)
    Retorna: (N, 4*D) - Features prontos para o predict do XGBoost
    """
    if vaga_emb.ndim == 1:
        # Repete o embedding da vaga N vezes para o empilhamento
        N = cv_embs.shape[0]
        vaga_embs_repeated = np.repeat(vaga_emb[np.newaxis, :], N, axis=0)
    else:
        # Se for um batch de vagas (N, D), usa diretamente
        vaga_embs_repeated = vaga_emb

    # A função np.hstack faz o empilhamento horizontal das quatro matrizes
    features = np.hstack([
        vaga_embs_repeated, # 1. Vaga (v)
        cv_embs,            # 2. CV (c)
        np.abs(vaga_embs_repeated - cv_embs), # 3. Diferença Absoluta (|v - c|)
        vaga_embs_repeated * cv_embs          # 4. Produto Hadamard (v * c)
    ])
    
    return features

# ======================== LÓGICA DE INICIALIZAÇÃO ========================

# 1. Configurações da Página 
st.set_page_config(
    page_title=APP_NAME, 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# 2. Carrega os Modelos (SBERT e XGBoost)
# O SBERT é carregado primeiro, pois é usado para gerar embeddings para o XGBoost.
with st.spinner("Carregando modelos (SBERT e XGBoost)..."):
    # Carrega o Encoder SBERT
    load_model(MODEL_DIR)
    st.success(f"✅ Encoder SBERT carregado com sucesso de: **{MODEL_DIR}**")

    # Carrega o Modelo de Ranqueamento XGBoost
    xgb_ranking_model = load_xgb_model(XGB_MODEL_PATH)
    # Armazena o modelo na session state para uso posterior
    st.session_state["xgb_ranking_model"] = xgb_ranking_model 

# 3. Carrega Bases Fixas (e concatena o texto)
candidatos_df, vagas_df, _logs = load_fixed_bases()

# DEBUG: Informações de debug na sidebar
st.sidebar.write("---")
st.sidebar.subheader("📊 Debug Info")
st.sidebar.write(f"Candidatos: {len(candidatos_df)} registros")
st.sidebar.write(f"Vagas: {len(vagas_df)} registros")
st.sidebar.write(f"XGBoost Status: {'✅ Pronto' if xgb_ranking_model else '❌ Ausente'}") # Novo item

# Exibir logs detalhados
with st.sidebar.expander("Logs de Carregamento"):
    for log in _logs:
        st.sidebar.text(log)

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
                st.session_state["cache_loaded"] = False 
                st.session_state["cand_embs_cache"] = None
                st.session_state["vaga_embs_cache"] = None

        except Exception as e:
            # Captura a falha de memória ou qualquer outra exceção grave
            st.error(f"❌ Falha crítica ao carregar/gerar embeddings. Aplicativo interrompido. Erro: {e}")
            
            st.session_state["cache_loaded"] = False
            st.session_state["cand_embs_cache"] = None
            st.session_state["vaga_embs_cache"] = None
            
            # 🛑 PARADA CRÍTICA: Impede que o Streamlit continue a execução sem os dados essenciais (Embeddings)
            st.stop()

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
        """
        CORREÇÃO 3: Formata o rótulo da vaga para o selectbox, usando a coluna 
        'titulo_vaga' que está presente na sua base de dados.
        """
        title = row.get("titulo_vaga") or "" 
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
            # ❗ CORREÇÃO 4: O valor máximo (max_value) do Top N precisa ser o tamanho do cdf.
            # O valor padrão é min(10, len(cdf)), mas o max_value deve ser len(cdf) 
            # para que ele carregue o valor correto (40492)
            # O bug da imagem (max 2) estava relacionado à falha de carregamento do CSV, corrigida acima.
            max_topn = len(cdf)
            default_topn = min(50, max_topn) # Sugestão: Top 50 por padrão
            top_n = st.number_input("Top N Candidatos", 1, max_topn, default_topn, key="topn_ranking")
        
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

                    # =======================================================
                    # 2. CÁLCULO DO SCORE: XGBoost vs. Similaridade de Cosseno
                    # =======================================================
                    
                    xgb_model = st.session_state.get("xgb_ranking_model")
                    
                    if xgb_model:
                        st.info("🧠 Usando modelo **XGBoost** para Ranqueamento por **Probabilidade de Match**.")
                        
                        # Geração dos features (combinação de embeddings)
                        features = generate_xgb_features(emb_vaga, emb_cvs)

                        # Previsão da probabilidade de ser "Match" (label 1)
                        probs = xgb_model.predict_proba(features)[:, 1] 
                        
                        # O score principal para ranqueamento é a probabilidade
                        scores_array = probs 
                        
                        # O limiar de aprovação é 0.50 (padrão de classificação)
                        limiar_aprovacao = 0.50 
                        
                        # Calcula a Similaridade de Cosseno para referência
                        sims = emb_cvs @ emb_vaga.T
                        
                    else:
                        # Fallback para o Match Semântico (SBERT Simples)
                        st.warning("⚠️ Modelo XGBoost indisponível. Usando **Similaridade de Cosseno** para ranqueamento.")
                        
                        # 2. Cálculo de similaridade (Produto escalar, pois ambos estão normalizados)
                        sims = emb_cvs @ emb_vaga.T
                        
                        # A similaridade é o score a ser ranqueado
                        scores_array = sims
                        
                        # O limiar é o do slider
                        limiar_aprovacao = limiar 
                        
                    # 3. Ordenação e Top N (sempre ordena pelo scores_array)
                    order = np.argsort(-scores_array)[:int(top_n)]
                    
                    # =======================================================
                    # 4. Geração do DataFrame de Resultados
                    # =======================================================
                    rows = []
                    for rank, i in enumerate(order, start=1):
                        main_score = float(scores_array[i])
                        cossine_sim = float(sims[i]) # Similaridade de Cosseno real (sempre calculada)

                        if xgb_model:
                            # Se for XGBoost, o Score é a Probabilidade * 100
                            score_porcentagem = round(main_score * 100, 2)
                            aprovado = bool(main_score >= limiar_aprovacao)
                        else:
                            # Se for SBERT, usa a lógica proporcional
                            score_porcentagem = proportional_score(main_score, limiar_aprovacao)
                            aprovado = bool(main_score >= limiar_aprovacao)
                            
                        
                        row = {
                            "Rank": rank, 
                            "Similaridade": round(cossine_sim, 6),
                            # Renomeado para Score (%) para refletir a probabilidade (XGBoost) ou proporcional (SBERT)
                            "Score (%)": round(score_porcentagem, 2), 
                            "Aprovado": aprovado,
                        }
                        
                        # Adiciona metadados do candidato para exibição (sem alteração)
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
                        "Similaridade": st.column_config.ProgressColumn("Similaridade (Cosseno)", format="%.4f", min_value=0.0, max_value=1.0),
                        # O Score (%) agora é o principal ranqueador
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
                    with col1: st.metric("Candidatos Aprovados", aprovados)
                    with col2: st.metric("Taxa de Aprovação", f"{aprovados/len(res)*100:.1f}%")
                    # O melhor score agora é o Score Principal (%)
                    with col3: st.metric("Melhor Score", f"{res.iloc[0]['Score (%)']:.2f}%")
                    
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

    # Garante que, mesmo que a chave falhe, teremos um DataFrame vazio para usar
    cand_full = st.session_state.get("candidatos_df", pd.DataFrame()) 
    vaga_full = st.session_state.get("vagas_df", pd.DataFrame())

    def _preview_df(df: pd.DataFrame, text_cols: list[str], max_chars: int = 300) -> pd.DataFrame:
        """Prepara um dataframe para visualização, truncando colunas de texto longas."""
        df = df.copy()
        view_cols = [c for c in df.columns if c not in text_cols] + text_cols
        dfv = df[view_cols].copy()
        for c in dfv.columns:
            if c in text_cols and dfv[c].dtype == object:
                # Esta é a lógica final e robusta para truncar strings:
                dfv[c] = dfv[c].str.slice(0, max_chars) + (dfv[c].apply(lambda x: '...' if isinstance(x, str) and len(x) > max_chars else ''))
        return dfv

    # ---- Candidatos ----
    st.subheader("Candidatos")
    st.metric("Total de Candidatos", len(cand_full)) # OK porque len(DataFrame vazio) = 0
    st.caption(f"Colunas concatenadas para match: **cv_text**")
    
    col_dl_c, col_prev_c = st.columns([1, 4])
    with col_dl_c:
        # Adiciona verificação .empty para desativar o download se o DataFrame estiver vazio
        if not cand_full.empty:
            st.download_button(
                "💾 Baixar Candidatos (CSV)", 
                # Esta linha 681 agora é segura
                cand_full.to_csv(index=False).encode("utf-8"),
                file_name="candidatos_full.csv", 
                mime="text/csv"
            )
        else:
            st.caption("Base de candidatos vazia ou não carregada.")

    st.divider()

    # ---- Vagas ----
    st.subheader("Vagas")
    vaga_full = st.session_state["vagas_df"]
    st.metric("Total de Vagas", len(vaga_full))
    st.caption(f"Colunas concatenadas para match: **vaga_text**")

    col_dl_v, col_prev_v = st.columns([1, 4])
    with col_dl_v:
        st.download_button(
            "💾 Baixar Vagas (CSV)", 
            vaga_full.to_csv(index=False).encode("utf-8"),
            file_name="vagas_full.csv", 
            mime="text/csv"
        )
    with col_prev_v:
        if st.checkbox("Mostrar Preview de Vagas"):
            st.dataframe(_preview_df(vaga_full, ["vaga_text", "requisitos", "descricao"]), use_container_width=True, height=300, hide_index=True)