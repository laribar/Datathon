# ==============================================================================
# 1. IMPORTS E CONFIGURAÇÕES INICIAIS
# ==============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import s3fs
import boto3
import os
import joblib
import time
import tempfile 
import shutil 
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import logging

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

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

# Caminho do SBERT no S3
SBERT_MODEL_DIR = "sbert_encoder" 

# Nomes dos arquivos
CANDIDATOS_FILE = "applicants_clean.csv"
VAGAS_FILE = "vagas_clean.csv"
EMBEDDINGS_FILE = "candidatos.npy"
VAGAS_EMBEDDINGS_FILE = "vagas.npy"
MODEL_FILE = "modelo_match_xgboost.pkl"
ENCODER_FILE = "encoder_le.pkl" 

# Colunas para o embedding e IDs
CV_TEXT_COL = 'curriculo_pt' 
VAGA_ID_COL = 'id' 
CANDIDATO_ID_COL = 'id' 

# Coluna para o texto combinado da vaga
VAGA_TEXT_COL = 'vaga_text' 

# --- 🎯 INJEÇÃO DE SECRETS DO STREAMLIT CLOUD ---
if "aws" in st.secrets:
    try:
        # Tenta pegar as chaves em MAIÚSCULAS (formato ideal)
        access_key = st.secrets["aws"].get("AWS_ACCESS_KEY_ID")
        secret_key = st.secrets["aws"].get("AWS_SECRET_ACCESS_KEY")
        aws_region = st.secrets["aws"].get("AWS_REGION") or st.secrets["aws"].get("AWS_DEFAULT_REGION")
        
        # Se as chaves em maiúsculas falharem, tenta pegar em minúsculas (seu formato original)
        if not access_key:
             access_key = st.secrets["aws"]["aws_access_key_id"]
             secret_key = st.secrets["aws"]["aws_secret_access_key"]
             aws_region = st.secrets["aws"].get("region_name")
        
        # Exporta as chaves para as variáveis de ambiente
        os.environ["AWS_ACCESS_KEY_ID"] = access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = secret_key
        os.environ["AWS_DEFAULT_REGION"] = aws_region
        
        logger.info(f"Credenciais AWS carregadas via st.secrets. Região: {aws_region}")
    except KeyError as e:
        logger.error(f"Erro: Segredo AWS faltando. Chave não encontrada: {e}. Verifique o formato TOML.")
        # Mantemos o st.error e st.stop() aqui porque é uma falha de configuração inicial
        st.error(f"❌ Segredo AWS faltando. Verifique se as chaves (ID, SECRET, REGION) estão no formato [aws] correto.")
        st.stop()
# -----------------------------------------------------------------------------


# ==============================================================================
# 3. FUNÇÕES DE CACHE E CARREGAMENTO (PURAS: SEM ST.INFO/TOAST/ERROR)
# ==============================================================================

@st.cache_resource(show_spinner=False)
def get_s3_fs():
    """Retorna o filesystem do S3 com configuração correta. Não usa comandos de UI."""
    try:
        fs = s3fs.S3FileSystem(anon=False) 
        fs.ls(S3_BUCKET)
        return fs
    except Exception as e:
        # 🚨 CORREÇÃO CACHE: Levanta exceção em vez de st.error/st.stop()
        raise RuntimeError(f"Erro de conexão com S3: {e}. Verifique as credenciais AWS.")

@st.cache_resource(show_spinner="Carregando modelo XGBoost do S3...")
def load_models() -> Any: 
    """Carrega o modelo XGBoost do S3 (Puro, sem comandos Streamlit de UI)."""
    
    fs = get_s3_fs()
    model_s3_path = f"{S3_BUCKET}/data/models/{MODEL_FILE}"
    
    # 🚨 CORREÇÃO CACHE: Remove comandos st.info, st.error, st.toast e o with st.spinner interno.
    
    if not fs.exists(model_s3_path):
        raise FileNotFoundError(f"Arquivo do modelo XGBoost ESSENCIAL não encontrado: s3://{model_s3_path}")

    with fs.open(model_s3_path, 'rb') as f:
        bst = joblib.load(f)
            
    if bst is None:
        raise ValueError("Modelo XGBoost está vazio")
            
    return bst

@st.cache_resource(show_spinner="Carregando Sentence Transformer...")
def load_encoder(model_name: str = 'all-MiniLM-L6-v2') -> SentenceTransformer:
    """Carrega o modelo SBERT (Puro, sem comandos Streamlit de UI)."""
    
    temp_dir = None
    try:
        fs = get_s3_fs()
        sbert_s3_path = f"{S3_BUCKET}/{SBERT_MODEL_DIR}"
        test_file_path = f"{sbert_s3_path}/config.json"
        
        if not fs.exists(test_file_path):
             logger.warning(f"Modelo SBERT não encontrado em S3 ({sbert_s3_path}). Tentando baixar da internet...")
             encoder = SentenceTransformer(model_name)
             return encoder

        # O with st.spinner do decorator é usado. Este é um processo de longa duração.
        temp_dir = tempfile.mkdtemp()
        local_model_path = os.path.join(temp_dir, SBERT_MODEL_DIR)
        fs.get(sbert_s3_path, local_model_path, recursive=True)
        
        encoder = SentenceTransformer(local_model_path)
            
        test_embedding = encoder.encode(["teste"], convert_to_numpy=True)
        if test_embedding.shape[1] == 0:
            raise ValueError("Embedding de teste vazio")
            
        return encoder
        
    except Exception as e:
        # 🚨 CORREÇÃO CACHE: Levanta exceção em vez de st.error/st.stop()
        raise RuntimeError(f"Falha crítica ao carregar SBERT: {e}")
    finally:
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"Diretório temporário limpo: {temp_dir}")
            except Exception as e:
                logger.error(f"Erro ao limpar diretório temporário: {e}")

@st.cache_data(show_spinner="Carregando dados dos candidatos e vagas do S3...")
def load_data(_max_rows: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Carrega os DataFrames de candidatos e vagas do S3. 
    Retorna os DataFrames e uma lista de logs/mensagens para exibição na UI.
    """
    
    log_messages = []
    cdf = pd.DataFrame()
    vdf = pd.DataFrame()
    fs = None

    # --- Carregar Candidatos ---
    try:
        fs = get_s3_fs()
        candidatos_s3_path = f"{S3_BUCKET}/data/{CANDIDATOS_FILE}"
        
        # 🚨 CORREÇÃO CACHE: Remove st.info
        log_messages.append(f"📁 Buscando candidatos em: s3://{candidatos_s3_path}")
        
        if not fs.exists(candidatos_s3_path):
            raise FileNotFoundError(f"Arquivo de candidatos não encontrado: s3://{candidatos_s3_path}")
        
        with fs.open(candidatos_s3_path, 'rb') as f:
            cdf = pd.read_csv(f, nrows=_max_rows, encoding='latin-1', engine='python', on_bad_lines='skip') 
        
        required_candidato_cols = [CANDIDATO_ID_COL, CV_TEXT_COL]
        if not all(col in cdf.columns for col in required_candidato_cols):
             missing_cols = [col for col in required_candidato_cols if col not in cdf.columns]
             raise KeyError(f"Candidatos sem colunas: {missing_cols}")
             
        cdf = cdf.dropna(subset=required_candidato_cols)
        cdf[CV_TEXT_COL] = cdf[CV_TEXT_COL].astype(str)
        
        log_messages.append(f"✅ Candidatos carregados: {len(cdf):,} registros")

    except Exception as e:
        log_messages.append(f"❌ Erro ao carregar candidatos: {str(e)}")
        cdf = pd.DataFrame() 

    # --- Carregar Vagas ---
    try:
        vagas_s3_path = f"{S3_BUCKET}/data/{VAGAS_FILE}"
        
        # 🚨 CORREÇÃO CACHE: Remove st.info
        log_messages.append(f"📁 Buscando vagas em: s3://{vagas_s3_path}")
        
        if not fs.exists(vagas_s3_path):
            raise FileNotFoundError(f"Arquivo de vagas não encontrado: s3://{vagas_s3_path}")
        
        with fs.open(vagas_s3_path, 'rb') as f:
            vdf = pd.read_csv(f, encoding='latin-1')
        
        required_vaga_cols = [VAGA_ID_COL, 'titulo_vaga']
        if not all(col in vdf.columns for col in required_vaga_cols):
             missing_cols = [col for col in required_vaga_cols if col not in vdf.columns]
             raise KeyError(f"Vagas sem colunas: {missing_cols}")
             
        text_cols_to_combine = ['titulo_vaga', 'objetivo_vaga', 'nivel_profissional', 'principais_atividades', 'competencias', 'habilidades_comportamentais']
        existing_text_cols = [col for col in text_cols_to_combine if col in vdf.columns]
        
        if len(existing_text_cols) > 0:
            vdf[VAGA_TEXT_COL] = vdf[existing_text_cols].fillna('').astype(str).agg(' '.join, axis=1)
        else:
             raise ValueError(f"Nenhuma coluna base encontrada para criar a coluna '{VAGA_TEXT_COL}'.")

        vdf = vdf.dropna(subset=[VAGA_TEXT_COL, VAGA_ID_COL])
        
        log_messages.append(f"✅ Vagas carregadas: {len(vdf):,} registros")

    except Exception as e:
        log_messages.append(f"❌ Erro ao carregar vagas: {str(e)}")
        vdf = pd.DataFrame() 

    if cdf.empty or vdf.empty:
        log_messages.append("🚨 Crítico: Dados insuficientes para continuar.")
    
    # 🚨 CORREÇÃO CACHE: Remove o bloco de st.subheader/st.success/st.error/st.info/st.stop()
    return cdf, vdf, log_messages

@st.cache_data(
    show_spinner="Gerenciando cache de embeddings...",
    # 🚨 CORREÇÃO UNHASHABLEPARAMERROR: Ignora o encoder no cálculo do hash
    hash_funcs={SentenceTransformer: lambda _: None} 
)
def get_or_create_embeddings(
    df: pd.DataFrame, 
    text_col: str, 
    filename: str, 
    encoder: SentenceTransformer,
    _use_cache: bool = True
) -> np.ndarray:
    """Gerencia cache de embeddings (Puro, sem comandos Streamlit de UI, exceto progress)."""
    
    start_time = time.time()
    
    cache_key = f"embeddings_{filename}_{len(df)}"
    # 🚨 CORREÇÃO CACHE: Remove st.info
    if _use_cache and cache_key in st.session_state:
        return st.session_state[cache_key]

    try:
        fs = get_s3_fs()
        s3_emb_path = f"{S3_BUCKET}/data/embeddings/{filename}"
        
        # 1. Tentar carregar do S3
        if _use_cache and fs.exists(s3_emb_path):
             with st.spinner(f"☁️ Carregando embeddings do S3: {filename}"):
                 with fs.open(s3_emb_path, 'rb') as f:
                     embeddings = np.load(f)
                 
                 if embeddings.shape[0] == len(df):
                     st.session_state[cache_key] = embeddings
                     # 🚨 CORREÇÃO CACHE: Remove st.success
                     return embeddings
                 # 🚨 CORREÇÃO CACHE: Remove st.warning
                    
    except Exception:
        # 🚨 CORREÇÃO CACHE: Remove st.info
        pass 

    # 2. Gerar novos embeddings
    # 🚨 CORREÇÃO CACHE: Remove st.warning
    
    texts = df[text_col].tolist()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    batch_size = 64
    all_embeddings = []
    
    # ... (lógica de encode com st.progress) ...
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = encoder.encode(batch_texts, show_progress_bar=False, convert_to_numpy=True, batch_size=min(batch_size, 32))
        all_embeddings.append(batch_embeddings)
        
        progress = min((i + batch_size) / len(texts), 1.0)
        progress_bar.progress(progress)
        status_text.text(f"Processando: {min(i + batch_size, len(texts)):,} / {len(texts):,}")
    
    embeddings = np.vstack(all_embeddings)
    progress_bar.empty()
    status_text.empty()
    
    # 3. Tentar salvar no S3 para uso futuro
    try:
        fs = get_s3_fs()
        s3_emb_path = f"{S3_BUCKET}/data/embeddings/{filename}"
        
        with st.spinner("💾 Salvando embeddings no S3..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.npy') as tmp_file:
                tmp_path = tmp_file.name
            
            np.save(tmp_path, embeddings)
            
            with open(tmp_path, 'rb') as f:
                fs.put(f, s3_emb_path)
            
            os.unlink(tmp_path)
            
        # 🚨 CORREÇÃO CACHE: Remove st.success
        
    except Exception:
        # 🚨 CORREÇÃO CACHE: Remove st.warning
        pass

    st.session_state[cache_key] = embeddings
    elapsed_time = time.time() - start_time
    # 🚨 CORREÇÃO CACHE: Remove st.info
    
    return embeddings

# ==============================================================================
# 4. FUNÇÕES DE PREDIÇÃO
# ==============================================================================

def predict_match_and_rank(
    vaga_embedding: np.ndarray, 
    all_candidate_embeddings: np.ndarray, 
    cdf: pd.DataFrame, 
    bst: xgb.Booster, 
    le: LabelEncoder, 
    top_k: int = 1000
) -> pd.DataFrame:
    """Calcula matching e ranking de forma otimizada."""
    
    if len(cdf) > top_k:
        similarities = cosine_similarity(vaga_embedding.reshape(1, -1), all_candidate_embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:]
        candidate_embeddings_subset = all_candidate_embeddings[top_indices]
        cdf_subset = cdf.iloc[top_indices].copy()
    else:
        candidate_embeddings_subset = all_candidate_embeddings
        cdf_subset = cdf.copy()

    vaga_emb_tiled = np.tile(vaga_embedding, (candidate_embeddings_subset.shape[0], 1))
    X_predict = np.hstack([candidate_embeddings_subset, vaga_emb_tiled])

    dtest = xgb.DMatrix(X_predict)
    predictions = bst.predict(dtest)

    results_df = cdf_subset.copy()
    results_df['probabilidade_match'] = predictions
    
    results_df = results_df.sort_values('probabilidade_match', ascending=False).reset_index(drop=True)
    results_df['rank'] = results_df.index + 1
    
    return results_df

# ==============================================================================
# 5. FUNÇÕES AUXILIARES PARA UI
# ==============================================================================

def format_currency(value: float) -> str:
    """Formata valor monetário."""
    try:
        return f"R$ {value:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
    except:
        return "R$ 0,00"

def display_candidate_card(candidate_data: pd.Series, rank: int):
    """Exibe um card formatado para cada candidato."""
    with st.container(border=True): 
        col1, col2, col3 = st.columns([3, 2, 1])
        
        with col1:
            st.subheader(f"#{rank} - {candidate_data[CANDIDATO_ID_COL]}")
            st.write(f"**Status:** {candidate_data.get('status', 'N/A')}")
            st.write(f"**Nível:** {candidate_data.get('nivel_hierarquico', 'N/A')}")
        
        with col2:
            st.write(f"**Gênero:** {candidate_data.get('genero', 'N/A')}")
            salary = candidate_data.get('salario_atual', 0)
            st.write(f"**Salário:** {format_currency(salary)}")
        
        with col3:
            prob = candidate_data['probabilidade_match']
            st.metric(
                label="Match", 
                value=f"{prob:.1%}",
                delta=f"Rank #{rank}" if rank <= 3 else None
            )
        
        with st.expander("📄 Ver CV Resumido"):
            cv_text = candidate_data.get(CV_TEXT_COL, '')
            preview = cv_text[:300] + "..." if len(cv_text) > 300 else cv_text
            st.text(preview)

def display_load_logs(log_messages: List[str]):
    """Exibe os logs de carregamento de forma organizada (Chamado no main)."""
    with st.container():
        st.subheader("📊 Status do Carregamento de Dados")
        data_loaded_ok = True
        for msg in log_messages:
            if "✅" in msg:
                st.success(msg)
            elif "❌" in msg:
                st.error(msg)
                data_loaded_ok = False
            elif "🚨" in msg:
                 st.error(msg)
                 data_loaded_ok = False
            else:
                st.info(msg)
        return data_loaded_ok

# ==============================================================================
# 6. EXECUÇÃO PRINCIPAL DO APLICATIVO
# ==============================================================================

def main():
    """Função principal do aplicativo Streamlit."""
    
    st.title("🎯 RECRUT.AI - Sistema de Match de Talentos")
    st.markdown("""
    **Tecnologias:** - 🤖 **Sentence Transformers (SBERT)** para embeddings de texto
    - 🌳 **XGBoost** para classificação de matching
    - ☁️ **AWS S3** para armazenamento e cache
    - ⚡ Otimizações de cache e performance
    """)
    
    # --- Sidebar: Configurações Iniciais ---
    with st.sidebar:
        st.header("🔐 Configurações AWS")
        
        current_region = os.environ.get("AWS_DEFAULT_REGION", "N/A")
        
        st.info(f"**Bucket:** {S3_BUCKET}")
        st.info(f"**Região:** {current_region}")
        
        if st.button("🧪 Testar Conexão S3"):
            try:
                fs = get_s3_fs()
                files = fs.ls(S3_BUCKET)
                st.success(f"✅ Conexão OK! {len(files)} itens no bucket")
                for file in files[:5]:
                    st.write(f"📁 {file}")
            except Exception as e:
                st.error(f"❌ Falha na conexão: {e}")
        
        st.markdown("---")
        st.header("⚙️ Configurações")
        
        # Controles de performance
        st.subheader("Performance")
        max_candidates = st.slider("Nº Máximo de Candidatos a Carregar", 100, 10000, 5000, 100)
        top_k_for_xgboost = st.slider("Top K Candidatos para Predição XGBoost", 100, 5000, 1000, 100)
        use_cache = st.checkbox("Usar Cache de Embeddings", value=True)
        
        st.markdown("---")
        st.subheader("Seleção de Vaga")
        
        # Carregar dados e exibir logs (Tratamento de exceções aqui)
        try:
            cdf, vdf, log_messages = load_data(max_candidates)
            data_ok = display_load_logs(log_messages)
            
            if not data_ok or cdf.empty or vdf.empty:
                 st.error("🚨 Crítico: Falha no carregamento dos dados. Verifique os logs acima.")
                 st.stop()
                 
        except RuntimeError as e: # Erros de S3 vêm do get_s3_fs
            st.error(f"❌ Erro Crítico de Conexão: {e}")
            st.stop()
        except Exception as e:
            st.error(f"❌ Erro Crítico no load_data: {e}")
            st.stop()
        
        # Seleção de vaga
        vaga_options = vdf.set_index(VAGA_ID_COL)['titulo_vaga'].to_dict()
        selected_vaga_id = st.selectbox(
            "Selecione a Vaga:",
            options=list(vaga_options.keys()),
            format_func=lambda x: f"{x} - {vaga_options[x]}"
        )
        
        # Número de resultados
        top_n = st.slider("Top Candidatos para Exibir", 5, 50, 15)
        
        st.markdown("---")
        st.info(f"""
        **Estatísticas:**
        - 📊 {len(cdf):,} candidatos carregados
        - 💼 {len(vdf):,} vagas disponíveis
        - 🎯 {top_n} resultados exibidos
        """)
    
    # --- Carregar Modelos ---
    with st.spinner("🚀 Inicializando modelos de IA..."):
        try:
            # 1. Carregar XGBoost
            bst = load_models() 
            st.toast("✅ Modelo XGBoost carregado com sucesso!", icon="✅")

            # 2. Carregar SBERT
            encoder = load_encoder()
            st.toast("✅ Encoder SBERT carregado com sucesso!", icon="✅")
            
        except RuntimeError as e: # Erros de S3 ou SBERT
            st.error(f"❌ Erro Crítico no Carregamento de Modelos: {e}")
            st.stop()
        except FileNotFoundError as e:
            st.error(f"❌ Erro Crítico: {e}")
            st.stop()
        except Exception as e:
            st.error(f"❌ Erro ao inicializar modelos: {e}")
            st.stop()
    
    # --- Carregar/Gerar Embeddings ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Embeddings de Candidatos")
        candidate_embeddings = get_or_create_embeddings(
            cdf, CV_TEXT_COL, EMBEDDINGS_FILE, encoder, use_cache
        )
        if use_cache and f"embeddings_{EMBEDDINGS_FILE}_{len(cdf)}" in st.session_state:
             st.success(f"✅ Embeddings Candidatos: {candidate_embeddings.shape} (Cache)")
        else:
             st.warning(f"🔄 Embeddings Candidatos gerados: {candidate_embeddings.shape}")
    
    with col2:
        st.subheader("Embeddings de Vagas")
        vaga_embeddings = get_or_create_embeddings(
            vdf, VAGA_TEXT_COL, VAGAS_EMBEDDINGS_FILE, encoder, use_cache
        )
        if use_cache and f"embeddings_{VAGAS_EMBEDDINGS_FILE}_{len(vdf)}" in st.session_state:
             st.success(f"✅ Embeddings Vagas: {vaga_embeddings.shape} (Cache)")
        else:
             st.warning(f"🔄 Embeddings Vagas gerados: {vaga_embeddings.shape}")
    
    st.markdown("---")
    
    # --- Processar Matching ---
    if selected_vaga_id:
        vaga_row = vdf[vdf[VAGA_ID_COL] == selected_vaga_id].iloc[0]
        vaga_index = vdf.index.get_loc(vaga_row.name)
        selected_vaga_emb = vaga_embeddings[vaga_index]
        
        st.header(f"🎯 Vaga Selecionada: {vaga_row['titulo_vaga']}")
        
        # ... (Exibição de detalhes da vaga) ...
        
        with st.spinner(f"🔍 Analisando {len(cdf):,} candidatos (Top {top_k_for_xgboost} para XGBoost)..."):
            start_time = time.time()
            le_mock = LabelEncoder() # LabelEncoder mock (vazio)
            
            results_df = predict_match_and_rank(
                selected_vaga_emb, candidate_embeddings, cdf, bst, le_mock, top_k=top_k_for_xgboost
            )
            processing_time = time.time() - start_time
        
        st.header(f"🏆 Top {top_n} Candidatos Recomendados")
        st.caption(f"⏱️ Tempo de Processamento do Ranking: {processing_time:.2f} segundos")
        
        # ... (Exibição de métricas e cards de candidatos) ...
        top_candidates = results_df.head(top_n)
        avg_prob = top_candidates['probabilidade_match'].mean()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Melhor Match", f"{results_df.iloc[0]['probabilidade_match']:.1%}")
        col2.metric("Match Médio (Top)", f"{avg_prob:.1%}")
        col3.metric("Candidatos Analisados", f"{len(results_df):,}")
        
        st.markdown("---")
        
        for idx, candidate in top_candidates.iterrows():
            display_candidate_card(candidate, candidate['rank'])
        
        # ... (Exportar Resultados) ...
        st.markdown("---")
        st.subheader("📊 Exportar Resultados")
        
        col1, col2 = st.columns(2)
        
        download_data_top_n = top_candidates.to_csv(index=False, encoding='utf-8').encode('utf-8')
        download_data_full = results_df.to_csv(index=False, encoding='utf-8').encode('utf-8')

        with col1:
            st.download_button(
                label=f"📥 Baixar Top {top_n} (CSV)",
                data=download_data_top_n,
                file_name=f"top_{top_n}_vaga_{selected_vaga_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
        
        with col2:
            st.download_button(
                label="📥 Baixar Ranking Completo (CSV)",
                data=download_data_full,
                file_name=f"ranking_completo_vaga_{selected_vaga_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

# ==============================================================================
# 7. EXECUÇÃO DO APLICATIVO
# ==============================================================================

if __name__ == "__main__":
    if 'embeddings_cache' not in st.session_state:
        st.session_state.embeddings_cache = {}
    
    main()