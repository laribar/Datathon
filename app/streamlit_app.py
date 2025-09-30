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
from sklearn.metrics.pairwise import cosine_similarity  # (mantido p/ compat.)
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
ENCODER_FILE = "encoder_le.pkl"  # (não utilizado diretamente aqui)

# Colunas para o embedding e IDs
CV_TEXT_COL = "curriculo_pt"
VAGA_ID_COL = "id"
CANDIDATO_ID_COL = "id"

# Coluna para o texto combinado da vaga
VAGA_TEXT_COL = "vaga_text"

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
        st.stop()

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
        fs = s3fs.S3FileSystem(anon=False)
        # valida acesso ao bucket
        fs.ls(S3_BUCKET)
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
            # teste rápido
            _ = encoder.encode(["probe"], convert_to_numpy=True)
            return encoder

        temp_dir = tempfile.mkdtemp()
        local_model_path = os.path.join(temp_dir, SBERT_MODEL_DIR)
        fs.get(sbert_s3_path, local_model_path, recursive=True)

        encoder = SentenceTransformer(local_model_path)
        _ = encoder.encode(["probe"], convert_to_numpy=True)
        return encoder

    except Exception as e:
        raise RuntimeError(f"Falha crítica ao carregar SBERT: {e}")
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
    Retorna os DataFrames e uma lista de logs/mensagens para exibição na UI.
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
                encoding="latin-1",
                engine="python",
                on_bad_lines="skip",
            )

        required_candidato_cols = [CANDIDATO_ID_COL, CV_TEXT_COL]
        missing_cols = [col for col in required_candidato_cols if col not in cdf.columns]
        if missing_cols:
            raise KeyError(f"Candidatos sem colunas: {missing_cols}")

        cdf = cdf.dropna(subset=required_candidato_cols)
        cdf[CV_TEXT_COL] = cdf[CV_TEXT_COL].astype(str)

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
        if existing_text_cols:
            vdf[VAGA_TEXT_COL] = vdf[existing_text_cols].fillna("").astype(str).agg(" ".join, axis=1)
        else:
            raise ValueError(f"Nenhuma coluna base encontrada para criar '{VAGA_TEXT_COL}'.")

        vdf = vdf.dropna(subset=[VAGA_TEXT_COL, VAGA_ID_COL])

        log_messages.append(f"✅ Vagas carregadas: {len(vdf):,} registros")

    except Exception as e:
        log_messages.append(f"❌ Erro ao carregar vagas: {str(e)}")
        vdf = pd.DataFrame()

    if cdf.empty or vdf.empty:
        log_messages.append("🚨 Crítico: Dados insuficientes para continuar.")

    return cdf, vdf, log_messages

@st.cache_data(
    show_spinner="Gerenciando cache de embeddings...",
    hash_funcs={SentenceTransformer: lambda _: None},  # evita UnhashableParam
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
        return np.zeros((0, 384), dtype="float32")  # dimensão padrão MiniLM

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
        # normaliza (idempotente) e garante dtype
        embeddings = _l2_normalize(embeddings.astype("float32"))
        st.session_state[cache_key] = embeddings
        return embeddings

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

    # 3) Salvar no S3 (corrigido: fs.put espera PATH local, não file-like)
    try:
        with st.spinner("💾 Salvando embeddings no S3..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as tmp:
                np.save(tmp.name, embeddings)
                tmp_path = tmp.name
            fs.put(tmp_path, s3_emb_path)  # ✅ correto
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
    bst: xgb.Booster,
    le: LabelEncoder,  # mantido p/ compatibilidade, não usado
    top_k: int = 1000,
) -> pd.DataFrame:
    """
    Calcula matching e ranking de forma otimizada.
    Pressupõe embeddings L2-normalizados (cosine -> produto interno).
    """
    if cdf.empty or all_candidate_embeddings.size == 0:
        return pd.DataFrame()

    # produto interno = similaridade cosseno (com normalização L2 prévia)
    sims = all_candidate_embeddings @ vaga_embedding.astype("float32")
    # pega top_k sem ordenar tudo
    k = min(top_k, len(sims))
    top_idx = np.argpartition(sims, -k)[-k:]
    # ordena desc nos top_k
    top_idx = top_idx[np.argsort(-sims[top_idx])]

    # evita np.tile: usa broadcast para concatenar (cand | vaga)
    X_left = all_candidate_embeddings[top_idx]
    X_right = np.broadcast_to(vaga_embedding, X_left.shape)
    X_predict = np.hstack([X_left, X_right]).astype("float32")

    dtest = xgb.DMatrix(X_predict)
    predictions = bst.predict(dtest)

    results_df = cdf.iloc[top_idx].copy()
    results_df["probabilidade_match"] = predictions
    results_df = results_df.sort_values("probabilidade_match", ascending=False).reset_index(drop=True)
    results_df["rank"] = results_df.index + 1
    return results_df

# ==============================================================================
# 5. FUNÇÕES AUXILIARES PARA UI
# ==============================================================================

def format_currency(value: float) -> str:
    try:
        return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "R$ 0,00"

def display_candidate_card(candidate_data: pd.Series, rank: int):
    with st.container(border=True):
        col1, col2, col3 = st.columns([3, 2, 1])

        with col1:
            st.subheader(f"#{rank} - {candidate_data.get(CANDIDATO_ID_COL, 'N/A')}")
            st.write(f"**Status:** {candidate_data.get('status', 'N/A')}")
            st.write(f"**Nível:** {candidate_data.get('nivel_hierarquico', 'N/A')}")

        with col2:
            st.write(f"**Gênero:** {candidate_data.get('genero', 'N/A')}")
            salary = candidate_data.get("salario_atual", 0)
            st.write(f"**Salário:** {format_currency(salary)}")

        with col3:
            prob = float(candidate_data.get("probabilidade_match", 0.0))
            st.metric(label="Match", value=f"{prob:.1%}", delta=f"Rank #{rank}" if rank <= 3 else None)

        with st.expander("📄 Ver CV Resumido"):
            cv_text = str(candidate_data.get(CV_TEXT_COL, ""))
            preview = cv_text[:300] + "..." if len(cv_text) > 300 else cv_text
            st.text(preview)

def display_load_logs(log_messages: List[str]) -> bool:
    with st.container():
        st.subheader("📊 Status do Carregamento de Dados")
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
# 6. EXECUÇÃO PRINCIPAL DO APLICATIVO
# ==============================================================================

def main():
    st.title("🎯 RECRUT.AI - Sistema de Match de Talentos")
    st.markdown(
        """
    **Tecnologias:**  
    - 🤖 **Sentence Transformers (SBERT)** para embeddings de texto  
    - 🌳 **XGBoost** para classificação de matching  
    - ☁️ **AWS S3** para armazenamento e cache  
    - ⚡ Otimizações de cache e performance  
    """
    )

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

        # Carregar dados e exibir logs
        try:
            cdf, vdf, log_messages = load_data(max_candidates)
            data_ok = display_load_logs(log_messages)

            if not data_ok or cdf.empty or vdf.empty:
                st.error("🚨 Crítico: Falha no carregamento dos dados. Verifique os logs acima.")
                st.stop()

        except RuntimeError as e:
            st.error(f"❌ Erro Crítico de Conexão: {e}")
            st.stop()
        except Exception as e:
            st.error(f"❌ Erro Crítico no load_data: {e}")
            st.stop()

        vaga_options = vdf.set_index(VAGA_ID_COL)["titulo_vaga"].to_dict()
        selected_vaga_id = st.selectbox(
            "Selecione a Vaga:",
            options=list(vaga_options.keys()),
            format_func=lambda x: f"{x} - {vaga_options[x]}",
        )

        # Número de resultados por página
        top_n = st.slider("Resultados por página", 5, 50, 15)

        st.markdown("---")
        st.info(
            f"""
        **Estatísticas:**
        - 📊 {len(cdf):,} candidatos carregados
        - 💼 {len(vdf):,} vagas disponíveis
        - 🎯 exibindo {top_n} por página
        """
        )

    # --- Carregar Modelos ---
    with st.spinner("🚀 Inicializando modelos de IA..."):
        try:
            bst = load_models()
            st.toast("✅ Modelo XGBoost carregado com sucesso!", icon="✅")

            encoder = load_encoder()
            st.toast("✅ Encoder SBERT carregado com sucesso!", icon="✅")

        except RuntimeError as e:
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
        candidate_embeddings = get_or_create_embeddings(cdf, CV_TEXT_COL, EMBEDDINGS_FILE, encoder, use_cache)
        if use_cache:
            st.success(f"✅ Embeddings Candidatos: {candidate_embeddings.shape} (cache/gerado)")

    with col2:
        st.subheader("Embeddings de Vagas")
        vaga_embeddings = get_or_create_embeddings(vdf, VAGA_TEXT_COL, VAGAS_EMBEDDINGS_FILE, encoder, use_cache)
        if use_cache:
            st.success(f"✅ Embeddings Vagas: {vaga_embeddings.shape} (cache/gerado)")

    st.markdown("---")

    # --- Processar Matching ---
    if selected_vaga_id:
        vaga_row = vdf[vdf[VAGA_ID_COL] == selected_vaga_id].iloc[0]
        vaga_index = vdf.index.get_loc(vaga_row.name)
        selected_vaga_emb = vaga_embeddings[vaga_index]

        st.header(f"🎯 Vaga Selecionada: {vaga_row['titulo_vaga']}")
        with st.expander("📌 Detalhes da Vaga"):
            st.write(vaga_row.dropna().to_dict())

        with st.spinner(f"🔍 Analisando {len(cdf):,} candidatos (Top {top_k_for_xgboost} para XGBoost)..."):
            start_time = time.time()
            le_mock = LabelEncoder()  # mantido p/ compat.
            results_df = predict_match_and_rank(
                selected_vaga_emb, candidate_embeddings, cdf, bst, le_mock, top_k=top_k_for_xgboost
            )
            processing_time = time.time() - start_time

        if results_df.empty:
            st.warning("Nenhum resultado para exibir.")
            st.stop()

        st.header("🏆 Candidatos Recomendados")
        st.caption(f"⏱️ Tempo de Processamento do Ranking: {processing_time:.2f} s")

        # Métricas rápidas
        top1 = float(results_df.iloc[0]["probabilidade_match"])
        avg_prob_15 = float(results_df.head(min(15, len(results_df)))["probabilidade_match"].mean())
        colm1, colm2, colm3 = st.columns(3)
        colm1.metric("Melhor Match", f"{top1:.1%}")
        colm2.metric("Match Médio (Top 15)", f"{avg_prob_15:.1%}")
        colm3.metric("Candidatos Analisados", f"{len(results_df):,}")

        # Paginação
        total_pages = (len(results_df) + top_n - 1) // top_n
        page = st.number_input("Página", min_value=1, max_value=max(1, total_pages), value=1, step=1)
        start = (page - 1) * top_n
        end = min(start + top_n, len(results_df))

        st.write(f"Mostrando {start + 1}–{end} de {len(results_df)}")

        for _, candidate in results_df.iloc[start:end].iterrows():
            display_candidate_card(candidate, int(candidate["rank"]))

        # Exportações
        st.markdown("---")
        st.subheader("📊 Exportar Resultados")

        colx1, colx2, colx3 = st.columns(3)

        # Top página atual (CSV)
        download_page_csv = results_df.iloc[start:end].to_csv(index=False, encoding="utf-8").encode("utf-8")
        with colx1:
            st.download_button(
                label=f"📥 Baixar Página {page} (CSV)",
                data=download_page_csv,
                file_name=f"resultados_pagina_{page}_vaga_{selected_vaga_id}_{datetime.now():%Y%m%d_%H%M}.csv",
                mime="text/csv",
            )

        # Ranking completo (CSV)
        download_full_csv = results_df.to_csv(index=False, encoding="utf-8").encode("utf-8")
        with colx2:
            st.download_button(
                label="📥 Baixar Ranking Completo (CSV)",
                data=download_full_csv,
                file_name=f"ranking_completo_vaga_{selected_vaga_id}_{datetime.now():%Y%m%d_%H%M}.csv",
                mime="text/csv",
            )

        # Ranking completo (Parquet)
        try:
            buf = io.BytesIO()
            results_df.to_parquet(buf, index=False)
            with colx3:
                st.download_button(
                    label="📥 Baixar Ranking Completo (Parquet)",
                    data=buf.getvalue(),
                    file_name=f"ranking_completo_vaga_{selected_vaga_id}_{datetime.now():%Y%m%d_%H%M}.parquet",
                    mime="application/octet-stream",
                )
        except Exception as e:
            # Parquet é opcional; se faltar pyarrow/fastparquet, apenas oculta
            logger.warning(f"Parquet indisponível: {e}")

# ==============================================================================
# 7. EXECUÇÃO DO APLICATIVO
# ==============================================================================

if __name__ == "__main__":
    if "embeddings_cache" not in st.session_state:
        st.session_state.embeddings_cache = {}
    main()
