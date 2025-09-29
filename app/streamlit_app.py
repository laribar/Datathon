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

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Any, List

# Configuração da página Streamlit
st.set_page_config(
    page_title="Seletor de Talentos com IA",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# 2. VARIÁVEIS DE CONFIGURAÇÃO (S3, PATHS, ETC.)
# ==============================================================================
# Configurações do S3
S3_BUCKET = "datathon-recrutai"
S3_DATA_PATH = f"s3://{S3_BUCKET}/data"
S3_MODEL_PATH = f"s3://{S3_BUCKET}/data/model"

# Nomes dos arquivos
CANDIDATOS_FILE = "aplicante_clean.csv"
VAGAS_FILE = "vagas_clean.csv"
EMBEDDINGS_FILE = "candidatos.npy"
MODEL_FILE = "modelo_match_xgboost.pkl"
ENCODER_FILE = "encoder_le.pkl"

# Colunas para o embedding
CV_TEXT_COL = 'cv_text'
VAGA_TEXT_COL = 'vaga_text'

# ==============================================================================
# 3. FUNÇÕES DE CACHE E CARREGAMENTO
# ==============================================================================

@st.cache_resource(show_spinner="Carregando modelos (SBERT e XGBoost)...")
def load_models():
    """Carrega o modelo XGBoost e o LabelEncoder do S3."""
    try:
        # Carregar XGBoost
        model_path = os.path.join(S3_MODEL_PATH, MODEL_FILE)
        bst = joblib.load(model_path)
        st.toast("Modelo XGBoost carregado com sucesso.", icon="✅")
        st.markdown(f'<div style="background-color:#003300; padding:10px; border-radius:5px; color:white;">Modelo XGBoost carregado do disco/modelo/{MODEL_FILE}.</div>', unsafe_allow_html=True)

        # Carregar LabelEncoder
        encoder_path = os.path.join(S3_MODEL_PATH, ENCODER_FILE)
        le = joblib.load(encoder_path)
        
        return bst, le
    except Exception as e:
        st.error(f"Erro ao carregar modelos do S3. Verifique as permissões/caminho: {e}")
        st.stop()

@st.cache_resource(show_spinner="Carregando Encoder (Priorizando Modelo Local)...")
def load_encoder(local_model_name='all-MiniLM-L6-v2'):
    """Carrega o modelo SBERT para geração de embeddings."""
    try:
        # Tenta carregar o modelo SBERT localmente
        encoder = SentenceTransformer(local_model_name)
        st.toast("Encoder (SBERT) carregado com sucesso.", icon="✅")
        return encoder
    except Exception as e:
        st.error(f"Erro ao carregar o modelo SBERT. Certifique-se de que a biblioteca 'sentence-transformers' está instalada e o modelo '{local_model_name}' existe ou está acessível: {e}")
        st.stop()

@st.cache_data(show_spinner="Carregando dados bases do disco...")
def load_data():
    """Carrega os DataFrames de candidatos e vagas do S3."""
    cdf = pd.DataFrame()
    vdf = pd.DataFrame()
    log_messages = []

    # Carregar Candidatos
    try:
        c_path = os.path.join(S3_DATA_PATH, CANDIDATOS_FILE)
        cdf = pd.read_csv(c_path)
        log_messages.append(f"✅ Dados de candidatos carregados: {len(cdf)} registros")
    except Exception as e:
        log_messages.append(f"⚠️ Não foi possível carregar o arquivo {CANDIDATOS_FILE}. Erro: {e}")

    # Carregar Vagas
    try:
        v_path = os.path.join(S3_DATA_PATH, VAGAS_FILE)
        vdf = pd.read_csv(v_path)
        log_messages.append(f"✅ Dados de vagas carregados: {len(vdf)} registros")
    except Exception as e:
        log_messages.append(f"⚠️ Não foi possível carregar o arquivo {VAGAS_FILE}. Erro: {e}")

    # Log de carregamento (exibido na UI)
    st.subheader("Logs do Carregamento de Bases")
    for msg in log_messages:
        if "✅" in msg:
            st.success(msg.replace("✅ ", "").strip())
        elif "⚠️" in msg:
            st.warning(msg.replace("⚠️ ", "").strip())
        else:
            st.info(msg)

    # Verifica se os DFs estão vazios
    if cdf.empty or vdf.empty:
        st.error("Falha Crítica: Um ou ambos os DataFrames não puderam ser carregados. Verifique as configurações do S3.")
        st.stop()
    
    return cdf, vdf

@st.cache_data(show_spinner="Preparando e carregando embeddings iniciais (Cache S3 ou Reconstrução)...")
def get_or_create_embeddings(df: pd.DataFrame, text_col: str, filename: str, encoder: SentenceTransformer) -> np.ndarray:
    """Tenta carregar embeddings do S3/disco ou as gera e salva."""
    
    local_path = filename
    s3_emb_path = os.path.join(S3_DATA_PATH, "embeddings", filename)

    # 1. Tentar carregar do S3
    try:
        with st.spinner(f"Cache de embeddings carregado do S3: {filename}"):
            # Usando fsspec (s3fs) para carregar o arquivo do S3
            fs = s3fs.S3FileSystem()
            with fs.open(s3_emb_path, 'rb') as f:
                embeddings = np.load(f)
            
            if embeddings.shape[0] == len(df):
                st.info(f"💾 Cache de embeddings carregado do disco/candidatos.npy")
                return embeddings
            else:
                st.warning("Cache S3 inválido (tamanho não bate). Reconstruindo embeddings.")
    except Exception as e:
        st.info(f"Não foi possível carregar o cache S3 ou disco ({e}). Reconstruindo embeddings...")

    # 2. Gerar embeddings (se o cache falhou)
    st.warning("Reconstruindo embeddings para 'cv_text' (Não há cache válido). Isso pode levar alguns minutos...")
    
    # Prepara os textos (garante que não há NaN)
    texts = df[text_col].astype(str).tolist()
    
    # Geração dos embeddings em lotes (para eficiência)
    with st.spinner("Gerando embeddings em lote..."):
        # Ajuste o tamanho do lote conforme a memória disponível
        batch_size = 32
        embeddings = encoder.encode(
            texts, 
            show_progress_bar=True, 
            convert_to_numpy=True, 
            batch_size=batch_size
        )
    
    # 3. Salvar no S3 para futuro cache
    try:
        with st.spinner(f"Salvando novos embeddings no S3 em {s3_emb_path}"):
            # Salvar no disco local temporariamente (boa prática antes de subir)
            np.save(local_path, embeddings)

            # Usando boto3 para upload
            s3 = boto3.client('s3')
            s3.upload_file(local_path, S3_BUCKET, f"data/embeddings/{filename}")
            
            # Limpar arquivo local
            os.remove(local_path)
            
            st.success("✅ Novos embeddings salvos no S3 com sucesso.")
    except Exception as e:
        st.error(f"Erro ao salvar novos embeddings no S3: {e}")

    return embeddings

# ==============================================================================
# 4. FUNÇÃO PRINCIPAL DE PREDIÇÃO E MATCHING
# ==============================================================================

def predict_match_and_rank(vaga_embedding: np.ndarray, all_candidate_embeddings: np.ndarray, cdf: pd.DataFrame, bst: xgb.Booster, le: LabelEncoder) -> pd.DataFrame:
    """Calcula similaridade, alimenta o XGBoost e classifica os candidatos."""
    
    # 1. Calcular Similaridade Cosseno (Medida de distância inicial)
    # Expande o embedding da vaga para ter o mesmo número de linhas dos candidatos
    vaga_emb_tiled = np.tile(vaga_embedding, (all_candidate_embeddings.shape[0], 1))
    
    # Concatena os embeddings para formar o X do modelo
    X_predict = np.hstack([all_candidate_embeddings, vaga_emb_tiled])

    # 2. Criar a Matriz DMatrix do XGBoost
    dtest = xgb.DMatrix(X_predict)
    
    # 3. Fazer a Predição (Probabilidade de Match)
    preds = bst.predict(dtest)
    
    # 4. Preparar o DataFrame de Resultados
    # Adiciona a probabilidade de match ao DF de candidatos
    results_df = cdf.copy()
    results_df['probabilidade_match'] = preds
    
    # 5. Aplicar o LabelEncoder inverso na coluna 'status_le' (se houver)
    # O modelo pode ter sido treinado com uma coluna target codificada
    if 'status_le' in results_df.columns:
        try:
            results_df['status_original'] = le.inverse_transform(results_df['status_le'])
        except Exception:
            # Ignora se a coluna não está codificada ou o encoder falhar
            pass
    
    # 6. Rankear
    results_df = results_df.sort_values(by='probabilidade_match', ascending=False)
    
    return results_df

# ==============================================================================
# 5. EXECUÇÃO DO APLICATIVO STREAMLIT
# ==============================================================================

# Carregar Modelos e Dados
bst, le = load_models()
encoder = load_encoder()
cdf, vdf = load_data()

# ==============================================================================
# TÍTULO E SIDEBAR
# ==============================================================================

st.title("Sistema de Match de Talentos (Candidato ↔ Vaga)")
st.caption("Baseado em SBERT para Embeddings de CV/Vagas e XGBoost para Classificação de Match.")

# ==============================================================================
# SIDEBAR
# ==============================================================================

with st.sidebar:
    st.header("Configurações")
    
    # Seleção de Vaga
    vaga_ids = vdf['id_vaga'].unique().tolist()
    selected_vaga_id = st.selectbox(
        "Selecione o ID da Vaga:",
        vaga_ids,
        index=0 # Seleciona a primeira vaga por padrão
    )
    
    # Filtro de Candidatos (Exemplo: apenas para fins de demonstração, pode ser removido)
    # status_list = ['todos'] + cdf['status'].unique().tolist()
    # selected_status = st.selectbox("Filtrar Candidatos por Status:", status_list)

    # Número de Top Candidatos a exibir
    top_n = st.slider("Número de Candidatos para Exibir:", 5, 100, 20)
    
    st.markdown("---")
    st.write(f"Dados carregados: **{len(cdf)} Candidatos** e **{len(vdf)} Vagas**.")
    st.markdown("---")
    st.info("Para este app, é necessário que as credenciais do AWS (s3fs, boto3) estejam configuradas para acesso ao bucket.")

# ==============================================================================
# CARREGAMENTO DE EMBEDDINGS (CACHE E GERAÇÃO)
# ==============================================================================

# Carrega/Gera embeddings dos candidatos (grande)
candidate_embeddings = get_or_create_embeddings(
    cdf, 
    CV_TEXT_COL, 
    EMBEDDINGS_FILE, 
    encoder
)

# Carrega/Gera embeddings das vagas (pequeno, pode ser feito em tempo real, mas usamos cache aqui)
vaga_embeddings = get_or_create_embeddings(
    vdf, 
    VAGA_TEXT_COL, 
    'vagas.npy', 
    encoder
)

# ==============================================================================
# LÓGICA DE MATCHING
# ==============================================================================

# 1. Filtrar a vaga selecionada
vaga_row = vdf[vdf['id_vaga'] == selected_vaga_id].iloc[0]
vaga_index = vdf.index.get_loc(vaga_row.name) # Posição da vaga no array de embeddings
selected_vaga_emb = vaga_embeddings[vaga_index]

# 2. Exibir a vaga selecionada
st.subheader(f"Vaga Selecionada: {selected_vaga_id}")
st.write(f"**Título:** {vaga_row['titulo_vaga']}")
with st.expander("Ver Descrição Completa (vaga_text)"):
    st.text(vaga_row[VAGA_TEXT_COL])

st.markdown("---")
st.subheader(f"Resultados do Match (Top {top_n} Candidatos)")

# 3. Executar o Matching e Ranking
ranked_results_df = predict_match_and_rank(
    selected_vaga_emb, 
    candidate_embeddings, 
    cdf, 
    bst, 
    le
)

# 4. Exibir o resultado
# Seleciona as colunas mais relevantes para exibição
cols_to_show = [
    'id_candidato', 
    'probabilidade_match', 
    'status', 
    'genero', 
    'nivel_hierarquico', 
    'salario_atual', 
    'cv_text'
]
display_df = ranked_results_df.head(top_n)[cols_to_show].copy()

# Formatação
display_df['probabilidade_match'] = (display_df['probabilidade_match'] * 100).round(2).astype(str) + '%'
display_df['salario_atual'] = display_df['salario_atual'].apply(lambda x: f'R$ {x:,.2f}'.replace(',', 'X').replace('.', ',').replace('X', '.'))


# Exibição
st.dataframe(display_df, use_container_width=True, hide_index=True)

# 5. Botão de Download (para o resultado completo)
st.download_button(
    f"Baixar CSV do Ranking Completo ({len(ranked_results_df)} registros)",
    ranked_results_df.to_csv(index=False).encode("utf-8"),
    file_name=f"ranking_vaga_{selected_vaga_id}.csv",
    mime="text/csv",
    key="dl_ranking",
)

# FIM DO CÓDIGO