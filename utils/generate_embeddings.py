import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import s3fs
import warnings

# === Variáveis de Configuração do Streamlit ===
# Apenas as variáveis essenciais para este script
S3_BUCKET = "datathon-recrutai"
CANDIDATOS_FILE = "applicants_clean.csv"
CV_TEXT_COL = "curriculo_pt"
EMBEDDINGS_FILE = "candidatos.npy"
# ---

warnings.filterwarnings("ignore")

def _l2_normalize(M: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    return M / n

def _decode_text(text: str) -> str:
    # Use a mesma função de correção de texto que você já tem
    if not isinstance(text, str): return ""
    try:
        return text.encode('latin-1').decode('utf-8', 'ignore')
    except:
        return text

def load_data_offline(file_name):
    # Simula a leitura que o Streamlit faz (ajuste as credenciais se necessário)
    fs = s3fs.S3FileSystem(anon=False)
    s3_path = f"{S3_BUCKET}/data/{file_name}"
    print(f"Lendo dados de s3://{s3_path}")
    
    # Use as mesmas configurações de leitura do Streamlit
    with fs.open(s3_path, "rb") as f:
        df = pd.read_csv(
            f,
            encoding="latin-1",
            engine="python",
            on_bad_lines="skip",
        )
    df[CV_TEXT_COL] = df[CV_TEXT_COL].astype(str).apply(_decode_text)
    return df.dropna(subset=[CV_TEXT_COL]).reset_index(drop=True)

def generate_and_upload_embeddings():
    print("Iniciando geração de embeddings...")
    
    # 1. Carregar Dados
    cdf = load_data_offline(CANDIDATOS_FILE)
    texts = cdf[CV_TEXT_COL].tolist()
    print(f"Total de candidatos a processar: {len(cdf):,}")
    
    # 2. Carregar Modelo
    # Baixa o modelo SBERT para a máquina local (ou usa cache se já tiver)
    print("Carregando Sentence Transformer (all-MiniLM-L6-v2)...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    
    # 3. Gerar Embeddings (este é o passo lento)
    print("Iniciando cálculo de embeddings. Isso pode levar vários minutos...")
    embeddings = encoder.encode(
        texts, 
        show_progress_bar=True, 
        convert_to_numpy=True, 
        batch_size=64
    ).astype("float32")
    
    embeddings = _l2_normalize(embeddings)
    print(f"Embeddings gerados com sucesso. Shape: {embeddings.shape}")

    # 4. Salvar Localmente
    local_file_path = f"local_{EMBEDDINGS_FILE}"
    np.save(local_file_path, embeddings)
    print(f"Arquivo temporário salvo em: {local_file_path}")
    
    # 5. Fazer Upload para o S3
    fs = s3fs.S3FileSystem(anon=False)
    s3_emb_path = f"{S3_BUCKET}/data/embeddings/{EMBEDDINGS_FILE}"
    print(f"Fazendo upload para s3://{S3_BUCKET}/data/embeddings/...")
    
    fs.put(local_file_path, s3_emb_path)
    print("✅ Upload concluído com sucesso!")
    
    # 6. Limpar
    os.remove(local_file_path)

if __name__ == "__main__":
    generate_and_upload_embeddings()