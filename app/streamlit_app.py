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
from datetime import datetime
import logging

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Any, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configuração da página Streamlit
st.set_page_config(
    page_title="RECRUT.AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# 2. VARIÁVEIS DE CONFIGURAÇÃO (S3, PATHS, ETC.)
# ==============================================================================
# Configurações do S3
S3_BUCKET = "datathon-recrutai"
S3_DATA_PATH = f"s3://{S3_BUCKET}/data"
S3_MODEL_PATH = f"s3://{S3_BUCKET}/data/models" 

# Nomes dos arquivos
CANDIDATOS_FILE = "aplicante_clean.csv"
VAGAS_FILE = "vagas_clean.csv"
EMBEDDINGS_FILE = "candidatos.npy"
VAGAS_EMBEDDINGS_FILE = "vagas.npy"
MODEL_FILE = "modelo_match_xgboost.pkl"
ENCODER_FILE = "encoder_le.pkl"

# Colunas para o embedding
CV_TEXT_COL = 'cv_text'
VAGA_TEXT_COL = 'vaga_text'

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================================================
# 3. FUNÇÕES DE CACHE E CARREGAMENTO OTIMIZADAS
# ==============================================================================

@st.cache_resource(show_spinner="Carregando modelos (SBERT e XGBoost)...")
def load_models() -> Tuple[Any, LabelEncoder]:
    """Carrega o modelo XGBoost e o LabelEncoder do S3 com tratamento robusto de erros."""
    try:
        # Verificar se estamos em ambiente local primeiro
        local_model_path = f"./modelo/{MODEL_FILE}"
        local_encoder_path = f"./modelo/{ENCODER_FILE}"
        
        if os.path.exists(local_model_path) and os.path.exists(local_encoder_path):
            # Carregar do local
            with st.spinner("Carregando modelos do disco local..."):
                bst = joblib.load(local_model_path)
                le = joblib.load(local_encoder_path)
                st.toast("✅ Modelos carregados do disco local.", icon="✅")
        else:
            # Carregar do S3 explicitamente
            with st.spinner("Carregando modelos do S3..."):
                fs = s3fs.S3FileSystem() # Instancia o FileSystem

                # NOTE: Removendo 's3://' pois s3fs.S3FileSystem() é usado explicitamente
                model_s3_path = os.path.join(S3_MODEL_PATH, MODEL_FILE).replace('s3://', '')
                encoder_s3_path = os.path.join(S3_MODEL_PATH, ENCODER_FILE).replace('s3://', '')

                # Carregamento binário do S3
                with fs.open(model_s3_path, 'rb') as f:
                    bst = joblib.load(f)

                with fs.open(encoder_s3_path, 'rb') as f:
                    le = joblib.load(f)

                st.toast("✅ Modelos carregados do S3.", icon="✅")
        
        # Validação dos modelos
        if bst is None:
            raise ValueError("Modelo XGBoost está vazio")
        if le is None:
            raise ValueError("LabelEncoder está vazio")
            
        return bst, le
        
    except Exception as e:
        st.error(f"❌ Erro crítico ao carregar modelos: {str(e)}")
        logger.error(f"Erro ao carregar modelos: {e}", exc_info=True)
        st.stop()

@st.cache_resource(show_spinner="Carregando Sentence Transformer...")
def load_encoder(model_name: str = 'all-MiniLM-L6-v2') -> SentenceTransformer:
    """Carrega o modelo SBERT com fallback para modelos menores."""
    try:
        # Tenta carregar o modelo especificado
        encoder = SentenceTransformer(model_name)
        
        # Testa o encoder com um texto pequeno
        test_embedding = encoder.encode(["teste"], convert_to_numpy=True)
        if test_embedding.shape[1] == 0:
            raise ValueError("Embedding de teste vazio")
            
        st.toast(f"✅ Encoder '{model_name}' carregado com sucesso.", icon="✅")
        return encoder
        
    except Exception as e:
        st.warning(f"⚠️ Não foi possível carregar {model_name}. Tentando modelo alternativo...")
        try:
            # Fallback para modelo menor
            encoder = SentenceTransformer('paraphrase-albert-small-v2')
            st.toast("✅ Encoder alternativo carregado.", icon="✅")
            return encoder
        except Exception as fallback_error:
            st.error(f"❌ Falha ao carregar qualquer encoder: {fallback_error}")
            st.stop()

@st.cache_data(show_spinner="Carregando dados dos candidatos e vagas...")
def load_data(_max_rows: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega os DataFrames de candidatos e vagas com opção de limite de linhas."""
    log_messages = []
    cdf = pd.DataFrame()
    vdf = pd.DataFrame()

    try:
        # Carregar Candidatos
        c_path = os.path.join(S3_DATA_PATH, CANDIDATOS_FILE)
        cdf = pd.read_csv(c_path, nrows=_max_rows)
        
        # Limpeza básica dos dados
        cdf = cdf.dropna(subset=[CV_TEXT_COL])
        cdf[CV_TEXT_COL] = cdf[CV_TEXT_COL].astype(str)
        
        log_messages.append(f"✅ Candidatos: {len(cdf):,} registros")

    except Exception as e:
        log_messages.append(f"❌ Erro ao carregar candidatos: {str(e)}")

    try:
        # Carregar Vagas
        v_path = os.path.join(S3_DATA_PATH, VAGAS_FILE)
        vdf = pd.read_csv(v_path)
        
        # Limpeza básica das vagas
        vdf = vdf.dropna(subset=[VAGA_TEXT_COL])
        vdf[VAGA_TEXT_COL] = vdf[VAGA_TEXT_COL].astype(str)
        
        log_messages.append(f"✅ Vagas: {len(vdf):,} registros")

    except Exception as e:
        log_messages.append(f"❌ Erro ao carregar vagas: {str(e)}")

    # Exibir logs de forma organizada
    with st.container():
        st.subheader("📊 Status do Carregamento de Dados")
        for msg in log_messages:
            if "✅" in msg:
                st.success(msg)
            elif "❌" in msg:
                st.error(msg)
            else:
                st.info(msg)

    # Validação final
    if cdf.empty or vdf.empty:
        st.error("🚨 Crítico: Dados insuficientes para continuar.")
        st.stop()
    
    return cdf, vdf

@st.cache_data(show_spinner="Gerenciando cache de embeddings...")
def get_or_create_embeddings(
    df: pd.DataFrame, 
    text_col: str, 
    filename: str, 
    encoder: SentenceTransformer,
    _use_cache: bool = True
) -> np.ndarray:
    """Gerencia cache de embeddings de forma eficiente com múltiplas estratégias."""
    
    start_time = time.time()
    
    # Verificar se já temos embeddings em cache
    cache_key = f"embeddings_{filename}_{len(df)}"
    if _use_cache and cache_key in st.session_state:
        st.info(f"♻️ Usando embeddings em cache: {filename}")
        return st.session_state[cache_key]

    local_path = f"./cache/{filename}"
    s3_emb_path = os.path.join(S3_DATA_PATH, "embeddings", filename)

    # 1. Tentar carregar do cache local
    if _use_cache and os.path.exists(local_path):
        try:
            with st.spinner(f"📥 Carregando embeddings locais: {filename}"):
                embeddings = np.load(local_path)
            if embeddings.shape[0] == len(df):
                st.session_state[cache_key] = embeddings
                st.success(f"✅ Embeddings locais carregados: {embeddings.shape}")
                return embeddings
        except Exception as e:
            st.warning(f"⚠️ Cache local corrompido: {e}")

    # 2. Tentar carregar do S3
    if _use_cache:
        try:
            with st.spinner(f"☁️ Buscando embeddings no S3: {filename}"):
                fs = s3fs.S3FileSystem()
                with fs.open(s3_emb_path, 'rb') as f:
                    embeddings = np.load(f)
                
                if embeddings.shape[0] == len(df):
                    # Salvar localmente para cache futuro
                    os.makedirs('./cache', exist_ok=True)
                    np.save(local_path, embeddings)
                    st.session_state[cache_key] = embeddings
                    st.success(f"✅ Embeddings S3 carregados: {embeddings.shape}")
                    return embeddings
                else:
                    st.warning("⚠️ Cache S3 incompatível. Gerando novos embeddings.")
        except Exception as e:
            st.info(f"ℹ️ Cache S3 não disponível: {e}")

    # 3. Gerar novos embeddings
    st.warning(f"🔄 Gerando novos embeddings para {len(df):,} registros...")
    
    # Preparar textos
    texts = df[text_col].tolist()
    
    # Otimização: processar em lotes com progresso
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    batch_size = 64
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = encoder.encode(
            batch_texts, 
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=min(batch_size, 32)
        )
        all_embeddings.append(batch_embeddings)
        
        # Atualizar progresso
        progress = min((i + batch_size) / len(texts), 1.0)
        progress_bar.progress(progress)
        status_text.text(f"Processando: {min(i + batch_size, len(texts)):,} / {len(texts):,}")
    
    embeddings = np.vstack(all_embeddings)
    progress_bar.empty()
    status_text.empty()
    
    # Salvar em cache
    try:
        os.makedirs('./cache', exist_ok=True)
        np.save(local_path, embeddings)
        st.session_state[cache_key] = embeddings
        st.success(f"✅ Novos embeddings gerados e salvos: {embeddings.shape}")
    except Exception as e:
        st.warning(f"⚠️ Não foi possível salvar cache local: {e}")

    elapsed_time = time.time() - start_time
    st.info(f"⏱️ Tempo total de processamento: {elapsed_time:.2f}s")
    
    return embeddings

# ==============================================================================
# 4. FUNÇÕES DE PREDIÇÃO OTIMIZADAS
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
    
    # Limitar número de candidatos para predição se necessário
    if len(cdf) > top_k:
        # Selecionar candidatos mais promissores por similaridade inicial
        similarities = cosine_similarity(vaga_embedding.reshape(1, -1), all_candidate_embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:]
        candidate_embeddings_subset = all_candidate_embeddings[top_indices]
        cdf_subset = cdf.iloc[top_indices].copy()
    else:
        candidate_embeddings_subset = all_candidate_embeddings
        cdf_subset = cdf.copy()

    # Preparar features para XGBoost
    vaga_emb_tiled = np.tile(vaga_embedding, (candidate_embeddings_subset.shape[0], 1))
    X_predict = np.hstack([candidate_embeddings_subset, vaga_emb_tiled])

    # Fazer predição
    dtest = xgb.DMatrix(X_predict)
    predictions = bst.predict(dtest)

    # Criar resultados
    results_df = cdf_subset.copy()
    results_df['probabilidade_match'] = predictions
    results_df['rank'] = range(1, len(results_df) + 1)
    
    # Ordenar por probabilidade
    results_df = results_df.sort_values('probabilidade_match', ascending=False)
    
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
    with st.container():
        col1, col2, col3 = st.columns([3, 2, 1])
        
        with col1:
            st.subheader(f"#{rank} - {candidate_data['id_candidato']}")
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
            cv_text = candidate_data.get('cv_text', '')
            preview = cv_text[:300] + "..." if len(cv_text) > 300 else cv_text
            st.text(preview)
        
        st.markdown("---")

# ==============================================================================
# 6. EXECUÇÃO PRINCIPAL DO APLICATIVO
# ==============================================================================

def main():
    """Função principal do aplicativo Streamlit."""
    
    # Header principal
    st.title("🎯 Sistema Avançado de Match de Talentos")
    st.markdown("""
    **Tecnologias:** 
    - 🤖 Sentence Transformers (SBERT) para embeddings de texto
    - 🌳 XGBoost para classificação de matching
    - ☁️ AWS S3 para armazenamento
    - ⚡ Otimizações de cache e performance
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        # Controles de performance
        st.subheader("Performance")
        max_candidates = st.slider(
            "Nº Máximo de Candidatos", 
            100, 10000, 5000, 100,
            help="Limite para processamento (afeta performance)"
        )
        
        use_cache = st.checkbox("Usar Cache", value=True, help="Usar embeddings em cache")
        
        st.markdown("---")
        st.subheader("Seleção de Vaga")
        
        # Carregar dados
        cdf, vdf = load_data(max_candidates)
        
        # Seleção de vaga
        vaga_options = vdf.set_index('id_vaga')['titulo_vaga'].to_dict()
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
        - 📊 {len(cdf):,} candidatos
        - 💼 {len(vdf):,} vagas
        - 🎯 {top_n} resultados
        """)
    
    # Carregar modelos
    with st.spinner("🚀 Inicializando modelos de IA..."):
        bst, le = load_models()
        encoder = load_encoder()
    
    # Carregar embeddings
    col1, col2 = st.columns(2)
    
    with col1:
        with st.spinner("📥 Carregando embeddings dos candidatos..."):
            candidate_embeddings = get_or_create_embeddings(
                cdf, CV_TEXT_COL, EMBEDDINGS_FILE, encoder, use_cache
            )
    
    with col2:
        with st.spinner("📥 Carregando embeddings das vagas..."):
            vaga_embeddings = get_or_create_embeddings(
                vdf, VAGA_TEXT_COL, VAGAS_EMBEDDINGS_FILE, encoder, use_cache
            )
    
    # Processar matching
    if selected_vaga_id:
        # Obter dados da vaga selecionada
        vaga_row = vdf[vdf['id_vaga'] == selected_vaga_id].iloc[0]
        vaga_index = vdf.index.get_loc(vaga_row.name)
        selected_vaga_emb = vaga_embeddings[vaga_index]
        
        # Exibir detalhes da vaga
        st.header(f"🎯 Vaga Selecionada: {vaga_row['titulo_vaga']}")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            with st.expander("📋 Ver Descrição Completa da Vaga", expanded=False):
                st.write(vaga_row[VAGA_TEXT_COL])
        
        with col2:
            st.metric("ID da Vaga", selected_vaga_id)
            st.metric("Candidatos para Análise", f"{len(cdf):,}")
        
        st.markdown("---")
        
        # Executar matching
        with st.spinner(f"🔍 Analisando {len(cdf):,} candidatos..."):
            start_time = time.time()
            results_df = predict_match_and_rank(
                selected_vaga_emb, candidate_embeddings, cdf, bst, le
            )
            processing_time = time.time() - start_time
        
        # Exibir resultados
        st.header(f"🏆 Top {top_n} Candidatos Recomendados")
        st.caption(f"⏱️ Processamento: {processing_time:.2f} segundos")
        
        # Métricas gerais
        top_candidates = results_df.head(top_n)
        avg_prob = top_candidates['probabilidade_match'].mean()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Melhor Match", f"{results_df.iloc[0]['probabilidade_match']:.1%}")
        col2.metric("Match Médio (Top)", f"{avg_prob:.1%}")
        col3.metric("Candidatos Analisados", f"{len(results_df):,}")
        
        # Lista de candidatos
        st.markdown("---")
        
        for idx, (_, candidate) in enumerate(top_candidates.iterrows(), 1):
            display_candidate_card(candidate, idx)
        
        # Download dos resultados
        st.markdown("---")
        st.subheader("📊 Exportar Resultados")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label=f"📥 Baixar Top {top_n} (CSV)",
                data=top_candidates.to_csv(index=False, encoding='utf-8'),
                file_name=f"top_{top_n}_vaga_{selected_vaga_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
        
        with col2:
            st.download_button(
                label="📥 Baixar Ranking Completo (CSV)",
                data=results_df.to_csv(index=False, encoding='utf-8'),
                file_name=f"ranking_completo_vaga_{selected_vaga_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

# ==============================================================================
# 7. EXECUÇÃO DO APLICATIVO
# ==============================================================================

if __name__ == "__main__":
    # Inicializar session state para cache
    if 'embeddings_cache' not in st.session_state:
        st.session_state.embeddings_cache = {}
    
    main()