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
import shutil # Importação adicional para limpeza
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

# 🚨 AJUSTE 1: Caminho do SBERT no S3
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

# Coluna para o texto combinado da vaga (Criada em load_data)
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
        st.error(f"❌ Segredo AWS faltando. Verifique se as chaves (ID, SECRET, REGION) estão no formato [aws] correto.")
        st.stop()
# -----------------------------------------------------------------------------


# ==============================================================================
# 3. FUNÇÕES DE CACHE E CARREGAMENTO
# ==============================================================================

@st.cache_resource(show_spinner=False)
def get_s3_fs():
    """Retorna o filesystem do S3 com configuração correta"""
    try:
        # Tenta pegar as variáveis de ambiente, que foram setadas no bloco acima
        fs = s3fs.S3FileSystem(anon=False) 
        # Testa o acesso listando o bucket
        fs.ls(S3_BUCKET)
        return fs
    except Exception as e:
        # Se falhar, é erro de permissão (Forbidden) ou credenciais ausentes.
        st.error(f"❌ Erro ao conectar com S3: {e}")
        st.info("ℹ️ Verifique se as credenciais AWS estão configuradas corretamente no Streamlit Secrets.")
        st.stop()

@st.cache_resource(show_spinner="Carregando modelos (SBERT e XGBoost)...")
def load_models() -> Any: 
    """Carrega o modelo XGBoost do S3 com tratamento robusto de erros."""
    try:
        fs = get_s3_fs()
        
        with st.spinner("Carregando modelos do S3..."):
            # Caminhos completos no S3
            model_s3_path = f"{S3_BUCKET}/data/models/{MODEL_FILE}"
            
            st.info(f"📁 Buscando modelo em: s3://{model_s3_path}")
            
            # Verifica se o arquivo do modelo (essencial) existe
            if not fs.exists(model_s3_path):
                st.error(f"❌ Arquivo do modelo XGBoost não encontrado: s3://{model_s3_path}. O modelo é ESSENCIAL para o match.")
                st.stop()

            # Carregamento do arquivo do modelo
            with fs.open(model_s3_path, 'rb') as f:
                bst = joblib.load(f)

            # O LabelEncoder (le) foi ignorado para contornar o erro de arquivo ausente.
            st.toast("✅ Modelo XGBoost carregado do S3 com sucesso! (LabelEncoder opcional foi ignorado)", icon="✅")
        
        # Validação do modelo
        if bst is None:
            raise ValueError("Modelo XGBoost está vazio")
            
        return bst
        
    except Exception as e:
        st.error(f"❌ Erro crítico ao carregar modelos: {str(e)}")
        logger.error(f"Erro ao carregar modelos: {e}", exc_info=True)
        st.stop()

@st.cache_resource(show_spinner="Carregando Sentence Transformer...")
def load_encoder(model_name: str = 'all-MiniLM-L6-v2') -> SentenceTransformer:
    """
    Carrega o modelo SBERT, priorizando o S3 e baixando-o localmente.
    A pasta do SBERT (sbert_encoder/) deve estar na raiz do bucket.
    """
    temp_dir = None
    try:
        fs = get_s3_fs()
        sbert_s3_path = f"{S3_BUCKET}/{SBERT_MODEL_DIR}"
        
        # 1. Criar um diretório temporário para armazenar o modelo
        temp_dir = tempfile.mkdtemp()
        local_model_path = os.path.join(temp_dir, SBERT_MODEL_DIR)
        
        # 2. Verificar se o modelo SBERT existe no S3
        # Procuramos por um arquivo chave dentro da pasta SBERT_MODEL_DIR
        test_file_path = f"{sbert_s3_path}/config.json"
        
        if not fs.exists(test_file_path):
             st.warning(f"⚠️ Modelo SBERT não encontrado em S3 ({sbert_s3_path}). Tentando baixar da internet...")
             # Fallback para download da internet (o comportamento original, mas agora é fallback)
             encoder = SentenceTransformer(model_name)
             st.toast("✅ Encoder carregado da internet.", icon="✅")
             return encoder

        # 3. Baixar toda a pasta do S3 para o disco local
        with st.spinner(f"☁️ Baixando modelo SBERT do S3 para cache local..."):
            # O comando get precisa do caminho completo do S3 para o caminho completo local
            # O s3fs.get é usado para copiar pastas inteiras recursivamente (rpath, lpath)
            fs.get(sbert_s3_path, local_model_path, recursive=True)
        
        # 4. Carregar o modelo SBERT a partir da pasta local
        encoder = SentenceTransformer(local_model_path)
            
        # 5. Testar o encoder
        test_embedding = encoder.encode(["teste"], convert_to_numpy=True)
        if test_embedding.shape[1] == 0:
            raise ValueError("Embedding de teste vazio")
            
        st.toast(f"✅ Encoder SBERT carregado de: {sbert_s3_path}", icon="✅")
        return encoder
        
    except Exception as e:
        st.error(f"❌ Falha crítica ao carregar SBERT: {e}")
        st.stop()
    finally:
        # 6. Limpar o diretório temporário após o carregamento
        if temp_dir and os.path.exists(temp_dir):
            try:
                # O shutils.rmtree é usado para remover pastas recursivamente
                shutil.rmtree(temp_dir)
                logger.info(f"Diretório temporário limpo: {temp_dir}")
            except Exception as e:
                logger.error(f"Erro ao limpar diretório temporário: {e}")


@st.cache_data(show_spinner="Carregando dados dos candidatos e vagas do S3...")
def load_data(_max_rows: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega os DataFrames de candidatos e vagas do S3."""
    
    log_messages = []
    cdf = pd.DataFrame()
    vdf = pd.DataFrame()

    # --- Carregar Candidatos ---
    try:
        fs = get_s3_fs()
        
        candidatos_s3_path = f"{S3_BUCKET}/data/{CANDIDATOS_FILE}"
        st.info(f"📁 Carregando candidatos de: s3://{candidatos_s3_path}")
        
        if not fs.exists(candidatos_s3_path):
            raise FileNotFoundError(f"Arquivo não encontrado: s3://{candidatos_s3_path}")
        
        with fs.open(candidatos_s3_path, 'rb') as f:
            # Usando o engine Python, mais robusto para CSVs mal-formatados ou com aspas complexas.
            cdf = pd.read_csv(
                f, 
                nrows=_max_rows, 
                encoding='latin-1', 
                engine='python',  # Engine Python
                on_bad_lines='skip' # Tenta ignorar linhas que causam o erro "tokenizing data"
            ) 
        
        # 🎯 VALIDAÇÃO DAS COLUNAS ESSENCIAIS DOS CANDIDATOS
        required_candidato_cols = [CANDIDATO_ID_COL, CV_TEXT_COL]
        if not all(col in cdf.columns for col in required_candidato_cols):
             missing_cols = [col for col in required_candidato_cols if col not in cdf.columns]
             # Levanta um erro específico que reflete o problema
             raise KeyError(f"O arquivo de candidatos não contém as colunas necessárias: {missing_cols}. Revise a variável CANDIDATO_ID_COL ou CV_TEXT_COL.")
             
        # Limpeza básica dos dados
        cdf = cdf.dropna(subset=required_candidato_cols)
        cdf[CV_TEXT_COL] = cdf[CV_TEXT_COL].astype(str)
        
        log_messages.append(f"✅ Candidatos: {len(cdf):,} registros")

    except Exception as e:
        log_messages.append(f"❌ Erro ao carregar candidatos: {str(e)}")
        # Garante que cdf seja vazio em caso de erro.
        cdf = pd.DataFrame() 

    # --- Carregar Vagas ---
    try:
        # Carregar Vagas
        vagas_s3_path = f"{S3_BUCKET}/data/{VAGAS_FILE}"
        st.info(f"📁 Carregando vagas de: s3://{vagas_s3_path}")
        
        if not fs.exists(vagas_s3_path):
            raise FileNotFoundError(f"Arquivo não encontrado: s3://{vagas_s3_path}")
        
        with fs.open(vagas_s3_path, 'rb') as f:
            # Mantendo o padrão para vagas
            vdf = pd.read_csv(f, encoding='latin-1')
        
        # VALIDAÇÃO DAS COLUNAS ESSENCIAIS
        required_vaga_cols = [VAGA_ID_COL, 'titulo_vaga']
        if not all(col in vdf.columns for col in required_vaga_cols):
             missing_cols = [col for col in required_vaga_cols if col not in vdf.columns]
             raise KeyError(f"O arquivo de vagas não contém as colunas necessárias: {missing_cols}")
             
        # CRIAÇÃO DA COLUNA VAGA_TEXT (texto combinado)
        text_cols_to_combine = [
            'titulo_vaga', 'objetivo_vaga', 'nivel_profissional',
            'principais_atividades', 'competencias', 'habilidades_comportamentais'
        ]
        
        existing_text_cols = [col for col in text_cols_to_combine if col in vdf.columns]
        
        if len(existing_text_cols) > 0:
            vdf[VAGA_TEXT_COL] = vdf[existing_text_cols].fillna('').astype(str).agg(' '.join, axis=1)
        else:
             raise ValueError(f"Nenhuma coluna base encontrada para criar a coluna '{VAGA_TEXT_COL}'. Verifique seu arquivo CSV.")

        # Limpeza básica das vagas 
        vdf = vdf.dropna(subset=[VAGA_TEXT_COL, VAGA_ID_COL])
        
        log_messages.append(f"✅ Vagas: {len(vdf):,} registros")

    except Exception as e:
        log_messages.append(f"❌ Erro ao carregar vagas: {str(e)}")
        vdf = pd.DataFrame() 

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
        st.info(f"""
        **Atenção aos Erros de Dados:**
        1. **Engine Python Ativado:** O motor de leitura Python foi ativado, junto com a opção de ignorar linhas mal-formatadas (`on_bad_lines='skip'`).
        2. **Delimitador:** O delimitador está configurado corretamente como vírgula (`,`).
        3. **Colunas ID:** O ID do candidato é esperado como **'{CANDIDATO_ID_COL}'** (apenas 'id') e o CV como **'{CV_TEXT_COL}'**.
        """)
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
    """Gerencia cache de embeddings de forma eficiente, priorizando session_state, depois S3."""
    
    start_time = time.time()
    
    # Verificar se já temos embeddings em cache de sessão
    cache_key = f"embeddings_{filename}_{len(df)}"
    if _use_cache and cache_key in st.session_state:
        st.info(f"♻️ Usando embeddings em cache (Session State): {filename}")
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
                    st.success(f"✅ Embeddings S3 carregados: {embeddings.shape}")
                    return embeddings
                else:
                    st.warning("⚠️ Cache S3 incompatível (tamanho diferente). Gerando novos embeddings.")
                    
    except Exception as e:
        st.info(f"ℹ️ Cache S3 não disponível ou erro de leitura: {e}")

    # 2. Gerar novos embeddings
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
        # Otimizado: batch_size no encoder.encode
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
    
    # 3. Tentar salvar no S3 para uso futuro
    try:
        fs = get_s3_fs()
        s3_emb_path = f"{S3_BUCKET}/data/embeddings/{filename}"
        
        with st.spinner("💾 Salvando embeddings no S3..."):
            # Criar arquivo temporário para garantir que o upload seja seguro
            with tempfile.NamedTemporaryFile(delete=False, suffix='.npy') as tmp_file:
                tmp_path = tmp_file.name
            
            # Salvar numpy array no arquivo temporário
            np.save(tmp_path, embeddings)
            
            # Fazer upload para S3
            with open(tmp_path, 'rb') as f:
                fs.put(f, s3_emb_path)
            
            # Limpar arquivo temporário
            os.unlink(tmp_path)
            
        st.success(f"✅ Novos embeddings salvos no S3: {embeddings.shape}")
        
    except Exception as e:
        st.warning(f"⚠️ Não foi possível salvar no S3: {e}")

    st.session_state[cache_key] = embeddings
    elapsed_time = time.time() - start_time
    st.info(f"⏱️ Tempo total de processamento: {elapsed_time:.2f}s")
    
    return embeddings

# ==============================================================================
# 4. FUNÇÕES DE PREDIÇÃO
# ==============================================================================

def predict_match_and_rank(
    vaga_embedding: np.ndarray, 
    all_candidate_embeddings: np.ndarray, 
    cdf: pd.DataFrame, 
    bst: xgb.Booster, 
    le: LabelEncoder, # Mantida para compatibilidade, mas não é usada no corpo da função
    top_k: int = 1000
) -> pd.DataFrame:
    """
    Calcula matching e ranking de forma otimizada.
    Usa similaridade cosseno para pré-filtrar o Top K para o XGBoost, 
    se o número total de candidatos for muito grande.
    """
    
    # Limitar número de candidatos para predição se necessário
    if len(cdf) > top_k:
        # Calcular Similaridade Cosseno para pré-seleção
        similarities = cosine_similarity(vaga_embedding.reshape(1, -1), all_candidate_embeddings)[0]
        # Pega os índices dos top_k candidatos mais similares
        top_indices = np.argsort(similarities)[-top_k:]
        # Filtra embeddings e DataFrame
        candidate_embeddings_subset = all_candidate_embeddings[top_indices]
        cdf_subset = cdf.iloc[top_indices].copy()
    else:
        candidate_embeddings_subset = all_candidate_embeddings
        cdf_subset = cdf.copy()

    # Preparar features para XGBoost: concatenação dos embeddings
    vaga_emb_tiled = np.tile(vaga_embedding, (candidate_embeddings_subset.shape[0], 1))
    X_predict = np.hstack([candidate_embeddings_subset, vaga_emb_tiled])

    # Fazer predição
    dtest = xgb.DMatrix(X_predict)
    predictions = bst.predict(dtest)

    # Criar resultados
    results_df = cdf_subset.copy()
    results_df['probabilidade_match'] = predictions
    
    # Ordenar por probabilidade e definir o rank
    results_df = results_df.sort_values('probabilidade_match', ascending=False).reset_index(drop=True)
    results_df['rank'] = results_df.index + 1
    
    return results_df

# ==============================================================================
# 5. FUNÇÕES AUXILIARES PARA UI
# ==============================================================================

def format_currency(value: float) -> str:
    """Formata valor monetário."""
    try:
        # Formato brasileiro: R$ 1.234,56
        return f"R$ {value:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
    except:
        return "R$ 0,00"

def display_candidate_card(candidate_data: pd.Series, rank: int):
    """Exibe um card formatado para cada candidato."""
    # Garante que o container use o fundo do tema Streamlit
    with st.container(border=True): 
        col1, col2, col3 = st.columns([3, 2, 1])
        
        with col1:
            # Usando a coluna ID de candidato definida
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
            # Usa a variável global CV_TEXT_COL
            cv_text = candidate_data.get(CV_TEXT_COL, '')
            preview = cv_text[:300] + "..." if len(cv_text) > 300 else cv_text
            st.text(preview)

# ==============================================================================
# 6. EXECUÇÃO PRINCIPAL DO APLICATIVO
# ==============================================================================

def main():
    """Função principal do aplicativo Streamlit."""
    
    # Header principal
    st.title("🎯 RECRUT.AI - Sistema de Match de Talentos")
    st.markdown("""
    **Tecnologias:** - 🤖 **Sentence Transformers (SBERT)** para embeddings de texto
    - 🌳 **XGBoost** para classificação de matching
    - ☁️ **AWS S3** para armazenamento e cache
    - ⚡ Otimizações de cache e performance
    """)
    
    # Sidebar: Configurações AWS e Dados
    with st.sidebar:
        st.header("🔐 Configurações AWS")
        
        current_region = os.environ.get("AWS_DEFAULT_REGION", "N/A")
        
        aws_region = st.selectbox(
            "Região AWS (Definida nos Secrets):",
            [current_region] if current_region != "N/A" else ["us-east-1", "us-east-2", "sa-east-1"],
            index=0
        )
        
        st.info(f"**Bucket:** {S3_BUCKET}")
        st.info(f"**Região:** {aws_region}")
        
        if st.button("🧪 Testar Conexão S3"):
            try:
                fs = get_s3_fs()
                files = fs.ls(S3_BUCKET)
                st.success(f"✅ Conexão OK! {len(files)} itens no bucket")
                for file in files[:5]:  # Mostra apenas os primeiros 5
                    st.write(f"📁 {file}")
            except Exception as e:
                st.error(f"❌ Falha na conexão: {e}")
        
        st.markdown("---")
        st.header("⚙️ Configurações")
        
        # Controles de performance
        st.subheader("Performance")
        max_candidates = st.slider(
            "Nº Máximo de Candidatos a Carregar", 
            100, 10000, 5000, 100,
            help="Limite para carregar o DataFrame de candidatos (afeta o tempo de carregamento inicial)"
        )
        top_k_for_xgboost = st.slider(
            "Top K Candidatos para Predição XGBoost", 
            100, 5000, 1000, 100,
            help="Número de candidatos pré-selecionados por similaridade cosseno para a predição final do XGBoost. Reduz para performance."
        )
        
        use_cache = st.checkbox("Usar Cache de Embeddings", value=True, help="Usar embeddings em cache (Session State e S3)")
        
        st.markdown("---")
        st.subheader("Seleção de Vaga")
        
        # Carregar dados
        # O max_candidates agora é passado para o load_data como o limite de linhas a carregar
        cdf, vdf = load_data(max_candidates)
        
        # Seleção de vaga
        # Usa VAGA_ID_COL para indexação e identificação
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
    
    # Carregar modelos (SBERT e XGBoost)
    with st.spinner("🚀 Inicializando modelos de IA..."):
        # Apenas carregamos o bst (Booster). O le (LabelEncoder) foi removido.
        bst = load_models() 
        # O encoder agora prioriza o S3
        encoder = load_encoder()
    
    # Carregar/Gerar embeddings
    col1, col2 = st.columns(2)
    
    with col1:
        candidate_embeddings = get_or_create_embeddings(
            cdf, CV_TEXT_COL, EMBEDDINGS_FILE, encoder, use_cache
        )
    
    with col2:
        vaga_embeddings = get_or_create_embeddings(
            vdf, VAGA_TEXT_COL, VAGAS_EMBEDDINGS_FILE, encoder, use_cache
        )
    
    # Processar matching
    if selected_vaga_id:
        # Obter dados da vaga selecionada
        # Usa VAGA_ID_COL para filtrar
        vaga_row = vdf[vdf[VAGA_ID_COL] == selected_vaga_id].iloc[0]
        # Obter o índice do embedding da vaga
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
        with st.spinner(f"🔍 Analisando {len(cdf):,} candidatos (Top {top_k_for_xgboost} para XGBoost)..."):
            start_time = time.time()
            
            # Criamos um LabelEncoder mock (vazio) apenas para satisfazer a assinatura 
            # da função predict_match_and_rank, mas ele não será usado.
            le_mock = LabelEncoder() 
            
            results_df = predict_match_and_rank(
                selected_vaga_emb, candidate_embeddings, cdf, bst, le_mock, top_k=top_k_for_xgboost
            )
            processing_time = time.time() - start_time
        
        # Exibir resultados
        st.header(f"🏆 Top {top_n} Candidatos Recomendados")
        st.caption(f"⏱️ Tempo de Processamento do Ranking: {processing_time:.2f} segundos")
        
        # Métricas gerais
        top_candidates = results_df.head(top_n)
        avg_prob = top_candidates['probabilidade_match'].mean()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Melhor Match", f"{results_df.iloc[0]['probabilidade_match']:.1%}")
        col2.metric("Match Médio (Top)", f"{avg_prob:.1%}")
        col3.metric("Candidatos Analisados", f"{len(results_df):,}")
        
        # Lista de candidatos
        st.markdown("---")
        
        for idx, candidate in top_candidates.iterrows():
            display_candidate_card(candidate, candidate['rank'])
        
        # Download dos resultados
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
    # Inicializar session state para cache de embeddings (se ainda não existir)
    if 'embeddings_cache' not in st.session_state:
        st.session_state.embeddings_cache = {}
    
    main()