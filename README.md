Com certeza! Para garantir que a documentação do seu projeto fique profissional e com a formatação correta no GitHub, usei uma estrutura de README.md que enfatiza a clareza, os blocos de código (para evitar problemas de alinhamento) e o fluxo do projeto.

🚀 Datathon — AI Recruitment: Plataforma de Match Inteligente
📌 Visão Geral
Este projeto desenvolve uma Plataforma de Otimização de Recrutamento e Seleção que utiliza Deep Learning e Processamento de Linguagem Natural (NLP) semântico para realizar o match inteligente entre currículos (CVs) e vagas.

O objetivo principal é reduzir o tempo gasto pelo RH em triagens manuais, aplicando modelos avançados de similaridade semântica para ranqueamento e um classificador robusto para predição de compatibilidade.

✨ Pilares da Solução
Embeddings Semânticos: Uso de Sentence Transformers (SBERT) para codificar o significado de CVs e vagas.

Classificação Avançada: Utilização do XGBoost para prever a probabilidade de match (compatibilidade).

Interface de Usuário: Dashboard interativo construído com Streamlit.

API Escalável: Rotas de predição implementadas com FastAPI.

Infraestrutura Cloud: Deploy via Render e armazenamento de modelos e dados grandes na AWS S3.

🔗 Acesso ao Aplicativo
O dashboard está publicado e pode ser acessado em:

👉 https://recrutai.streamlit.app/

Os arquivos mais pesados (modelos e dados limpos) foram colocados na AWS S3 para facilitar o deploy e garantir a performance, com o Streamlit consumindo diretamente da nuvem.

📂 Estrutura do Repositório
Aqui está a estrutura de pastas do projeto, garantindo o alinhamento correto com o bloco de código:
'''
DATATHON/
│
├── app/                        # Backend da Aplicação
│   ├── main.py                  # API REST (FastAPI)
│   ├── requirements.txt        # Dependências
│   └── streamlit_app.py        # Dashboard Interativo (Streamlit)
│
├── data/                       # Bases de dados
│   └── processed/              # Dados tratados
│       ├── applicants_clean.csv
│       └── vagas_clean.csv
│
├── models/                     # Modelos Treinados
│   ├── sbert_encoder/          # Sentence Transformer (SBERT)
│   └── modelo_match_xgboost.pkl # Modelo XGBoost final
│
├── notebooks/                  # Análise e Treinamento
│   └── Datathon.ipynb          # Notebook que faz pré-processamento, FE e treinamento.
│
├── utils/                      # Scripts Auxiliares (Limpeza e Conversão)
│
├── .env                        # Variáveis de ambiente
├── render.yaml                  # Configuração de Deploy (Render)
└── runtime.txt                  # Versão do Python utilizada'''



⚙️ Fluxo de Trabalho (Pipeline ML)
Etapa	Ferramentas	Descrição
1. Coleta/Processamento	Pandas, utils/	Conversão de JSON para CSVs limpos (applicants_clean.csv, vagas_clean.csv) e padronização.
2. Modelagem Semântica	SBERT, notebooks/	Geração de embeddings (vetores numéricos) dos CVs e Vagas.
3. Treinamento	XGBoost, Scikit-learn	Treinamento do classificador de Match usando os embeddings como features.
4. Deploy e Serviço	Streamlit, FastAPI, AWS S3	O Streamlit serve o dashboard, o FastAPI provê a API, e ambos carregam os modelos do S3.

Exportar para Sheets
💻 Como Executar o Projeto Localmente
1. Pré-requisitos e Credenciais AWS
Para carregar os dados e modelos, você precisará configurar suas credenciais AWS, pois o aplicativo busca artefatos do bucket S3.

Crie um arquivo .env (ou configure suas variáveis de ambiente) na raiz do projeto:

AWS_ACCESS_KEY_ID=sua_chave_id
AWS_SECRET_ACCESS_KEY=sua_chave_secreta
AWS_REGION=us-east-1 # (ou a região do seu bucket)
2. Setup do Ambiente
Clone o repositório e crie um ambiente virtual:

Bash

# 1. Clonar o repositório
git clone https://github.com/laribar/datathon.git
cd datathon

# 2. Criar ambiente virtual
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
.venv\Scripts\activate       # Windows

# 3. Instalar dependências
pip install -r app/requirements.txt
3. Executando a Aplicação
A) Iniciar a API com FastAPI
Bash

uvicorn app.main:app --reload --port 8000
Acesse a documentação interativa (Swagger UI) em: http://localhost:8000/docs

B) Iniciar o Dashboard Streamlit
Bash

streamlit run app/streamlit_app.py
O dashboard será aberto automaticamente no seu navegador, geralmente em: http://localhost:8501
