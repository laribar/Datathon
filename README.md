🚀 Datathon — AI Recruitment: Plataforma de Match Inteligente
📌 Visão Geral
Este projeto desenvolve uma Plataforma de Otimização de Recrutamento e Seleção utilizando Deep Learning e Processamento de Linguagem Natural (NLP) semântico para realizar o match inteligente entre currículos (CVs) e vagas.

O objetivo principal é reduzir o tempo gasto pelo RH em triagens manuais, aplicando modelos de similaridade semântica para ranqueamento e um classificador robusto para predição de compatibilidade.

✨ Pilares da Solução
Embeddings Semânticos: Uso de Sentence Transformers (SBERT) para codificar o significado de CVs e vagas.

Classificação Avançada: Utilização do XGBoost para prever a probabilidade de match (compatibilidade) entre candidato e vaga.

Interface de Usuário: Dashboard interativo construído com Streamlit.

API Escalável: Rotas de predição e match em batch implementadas com FastAPI.

Infraestrutura: Deploy facilitado pelo Render e armazenamento de modelos e dados grandes na AWS S3.

🔗 Acesso ao Aplicativo
O app está publicado e pode ser acessado em:
https://recrutai.streamlit.app/

As bases pesadas foram enviadas para AWS S3 e o streamlit consome diretamente de lá.
<img width="1595" height="540" alt="image" src="https://github.com/user-attachments/assets/780bc7dd-e34b-420d-85f0-3e3a2f004f79" />



⚙️ Fluxo de Trabalho (Pipeline ML)
O projeto segue um pipeline robusto, desde a ingestão de dados até o deploy da aplicação:

📥 Coleta de Dados: Ingestão das bases em formato JSON (applicants.json, prospects.json, vagas.json).

🧹 Processamento e Feature Engineering: Limpeza de texto, padronização e conversão de JSON para CSV (scripts em utils/).

🧠 Treinamento de Modelos:

Criação de Embeddings para todos os CVs e vagas usando o SBERT Encoder.

Treinamento do XGBoost (notebook Datathon.ipynb) para classificação de match binário.

💾 Armazenamento: Modelos pesados (SBERT e XGBoost) e dados limpos são versionados e armazenados na AWS S3 para facilitar o deploy (o Streamlit App carrega diretamente do S3).

🖥️ Aplicação: A camada de front-end (Streamlit) e a API de back-end (FastAPI) consomem os modelos prontos para servir as predições.

📂 Estrutura do Repositório
DATATHON/
│
├── app/                        # Backend da Aplicação
│   ├── main.py                  # API REST (FastAPI)
│   ├── requirements.txt        # Dependências da aplicação
│   └── streamlit_app.py        # Dashboard Interativo (Streamlit)
│
├── data/                       # Bases de dados
│   └── processed/              # Dados tratados (CSV e JSON originais)
│
├── models/                     # Modelos Treinados
│   ├── sbert_encoder/          # Sentence Transformer (SBERT)
│   └── modelo_match_xgboost.pkl # Modelo XGBoost final
│
├── notebooks/                  # Análise, Processamento e Treinamento
│   └── Datathon.ipynb          # Notebook de pré-processamento, FE, treinamento e validação.
│
├── utils/                      # Scripts Auxiliares
│   ├── json_to_csv_*.py        # Scripts de conversão de dados
│   └── transformar.py          # Script de limpeza e padronização de texto
│
├── .env                        # Variáveis de ambiente
├── render.yaml                  # Configuração de Deploy (Render)
└── runtime.txt                  # Versão do Python utilizada
💻 Como Executar o Projeto Localmente
Siga os passos abaixo para configurar e rodar a API e o Dashboard em sua máquina.

1. Pré-requisitos
Python 3.10+

Credenciais AWS: O aplicativo busca dados e modelos do AWS S3 (datathon-recrutai). Certifique-se de configurar as variáveis de ambiente com suas chaves de acesso.

2. Clonagem e Setup
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
A API provê os endpoints para integração com outros serviços (como as rotas /match e /batch_match).

Bash

# O --reload é opcional, usado para desenvolvimento
uvicorn app.main:app --reload --port 8000
Acesse a documentação interativa (Swagger UI) em: http://localhost:8000/docs

B) Iniciar o Dashboard Streamlit
O Dashboard é a interface visual para teste e uso da plataforma. Ele já faz o consumo dos modelos diretamente do S3.

Bash

streamlit run app/streamlit_app.py
O dashboard será aberto automaticamente no seu navegador, geralmente em: http://localhost:8501
