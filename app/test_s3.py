# test_s3.py
import pandas as pd
import s3fs

def check_s3_files():
    print("🔍 Iniciando verificação dos arquivos no S3...")
    
    try:
        fs = s3fs.S3FileSystem(anon=False)
        print("✅ Conectado ao S3 com sucesso!")
        
        # Verificar arquivo de candidatos
        candidatos_path = "datathon-recrutai/data/applicants_clean.csv"
        print(f"\n📁 Verificando: {candidatos_path}")
        
        if fs.exists(candidatos_path):
            with fs.open(candidatos_path, 'r') as f:
                df_candidatos = pd.read_csv(f, nrows=5)
                print("✅ Arquivo de CANDIDATOS encontrado!")
                print(f"📋 Colunas: {df_candidatos.columns.tolist()}")
                print(f"📊 Shape: {df_candidatos.shape}")
                print(f"🔍 Primeiras 3 linhas:")
                print(df_candidatos.head(3))
                print(f"💡 Tipos de dados:")
                print(df_candidatos.dtypes)
        else:
            print("❌ Arquivo de CANDIDATOS não encontrado!")
        
        # Verificar arquivo de vagas
        vagas_path = "datathon-recrutai/data/vagas_clean.csv"
        print(f"\n📁 Verificando: {vagas_path}")
        
        if fs.exists(vagas_path):
            with fs.open(vagas_path, 'r') as f:
                df_vagas = pd.read_csv(f, nrows=5)
                print("✅ Arquivo de VAGAS encontrado!")
                print(f"📋 Colunas: {df_vagas.columns.tolist()}")
                print(f"📊 Shape: {df_vagas.shape}")
                print(f"🔍 Primeiras 3 linhas:")
                print(df_vagas.head(3))
                print(f"💡 Tipos de dados:")
                print(df_vagas.dtypes)
        else:
            print("❌ Arquivo de VAGAS não encontrado!")
            
        # Listar outros arquivos no diretório
        print(f"\n📂 Listando todos os arquivos em datathon-recrutai/data/")
        try:
            files = fs.ls("datathon-recrutai/data/")
            for file in files:
                print(f"   📄 {file}")
        except Exception as e:
            print(f"⚠️ Erro ao listar arquivos: {e}")
            
    except Exception as e:
        print(f"❌ Erro geral: {e}")

if __name__ == "__main__":
    check_s3_files()