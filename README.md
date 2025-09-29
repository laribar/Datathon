🚀 Projeto Datathon — AI Recruitment
Plataforma de otimização de recrutamento e seleção com Deep Learning e NLP semântico (SBERT) para realizar match inteligente entre currículos e vagas.

📌 Visão Geral
Este projeto busca reduzir o tempo gasto pelo RH em triagens manuais, aplicando modelos de similaridade semântica e classificação de compatibilidade entre candidatos e vagas.

A solução combina:

Processamento e limpeza de dados de CVs/vagas

Embeddings semânticos com Sentence Transformers

Classificação com XGBoost

API em FastAPI e dashboard em Streamlit

Deploy em Render para uso em produção

🔗 PARA ACESSAR O APP: https://recrutai.streamlit.app/

📊 Fluxo Detalhado do Aplicativo RECRUT.AI
text
1. USUÁRIO ABRE O APP
   │
2. 🔐 VERIFICAÇÃO AWS
   ├── Testa credenciais
   ├── Lista arquivos no bucket
   └── Exibe status da conexão
   │
3. 📥 CARREGAMENTO INICIAL
   ├── Modelos ML (XGBoost, Encoder)
   ├── Dados (Candidatos, Vagas)
   └── Embeddings (Cache ou Geração)
   │
4. 🎛️ INTERAÇÃO DO USUÁRIO
   ├── Seleciona vaga
   ├── Ajusta parâmetros
   └── Dispara processamento
   │
5. 🔍 PROCESSAMENTO
   ├── Recupera embedding da vaga
   ├── Calcula matching com todos candidatos
   ├── Ordena por probabilidade
   └── Gera ranking
   │
6. 📊 EXIBIÇÃO
   ├── Mostra Top N candidatos
   ├── Exibe métricas
   └── Disponibiliza download
   │
7. 💾 PERSISTÊNCIA
   └── Salva novos embeddings no S3 (se gerados)
🏗️ Arquitetura do Sistema
Os arquivos mais pesados foram colocados na AWS S3 para facilitar o Deploy via Streamlit.

https://github.com/user-attachments/assets/7ad8d222-8f07-4b8d-8af2-382f7d38b19a

📂 Estrutura de Pastas
text
DATATHON/
│
├── app/                         # Backend da aplicação
│   ├── main.py                   # API principal (FastAPI)
│   ├── requirements.txt          # Dependências
│   └── streamlit_app.py          # Dashboard em Streamlit
│
├── data/                        # Bases de dados
│   └── processed/                # Dados tratados
│       ├── applicants_clean.csv
│       ├── prospects_clean.csv
│       ├── vagas_clean.csv
│       ├── applicants.json
│       ├── prospects.json
│       └── vagas.json
│
├── models/                      # Modelos treinados
│   ├── sbert_encoder/            # Encoder SBERT
│   └── modelo_match_xgboost.pkl  # Modelo XGBoost final
│
├── notebooks/                   # Notebooks de exploração
│   ├── Datathon.ipynb           # Pré-processamento / Feature Engineering / Treinamento / Validação
│
├── utils/                       # Scripts auxiliares
│   ├── json_to_csv_prospects.py
│   ├── json_to_csv_vagas.py
│   ├── map_files.py
│   ├── transformar.py
│   └── treinamento_modelo_match.py
│
├── .env                         # Variáveis de ambiente
├── render.yaml                  # Configuração Render
├── runtime.txt                  # Versão do Python
└── README.md                    # Documentação
⚙️ Fluxo do Projeto
1. 📥 Coleta de Dados
applicants.json → currículos

prospects.json → candidatos prospectados

vagas.json → descrições de vagas

2. 🧹 Processamento
Conversão JSON → CSV (json_to_csv_*)

Limpeza e padronização de texto (transformar.py)

3. 🧠 Modelagem
SBERT Encoder → embeddings de CVs e vagas

XGBoost → classificador de compatibilidade

Treinamento automatizado (Datathon.ipynb)

4. 🖥️ Interface
Streamlit (streamlit_app.py) → upload de CVs/vagas, visualização de matches

FastAPI (main.py) → rotas /match e /batch_match

5. ☁️ Deploy
Configurado no Render via render.yaml e runtime.txt

🛠️ Tecnologias Utilizadas
Python 3.10+

FastAPI (API REST)

Streamlit (Dashboard)

Sentence Transformers (SBERT)

XGBoost

Pandas / NumPy

Scikit-learn

Render

AWS S3

🚀 Como Executar Localmente
1. Clonar o repositório
bash
git clone https://github.com/laribar/datathon.git
cd datathon
2. Criar ambiente virtual
bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
3. Instalar dependências
bash
pip install -r app/requirements.txt
4. Rodar API FastAPI
bash
uvicorn app.main:app --reload --port 8000
5. Rodar Dashboard Streamlit
bash
streamlit run app/streamlit_app.py
📈 Funcionalidades Principais
🔍 Matching Inteligente
Análise semântica de currículos e vagas

Sistema de ranking baseado em múltiplos fatores

Probabilidade de match calculada por XGBoost

📊 Dashboard Interativo
Seleção de vagas via interface amigável

Visualização de top candidatos

Métricas de performance do matching

Exportação de resultados em CSV

⚡ Performance Otimizada
Cache multi-nível (memória, disco, S3)

Processamento em lote de embeddings

Sistema de fallback para alta disponibilidade

🎯 Resultados Esperados
Redução de 80% no tempo de triagem manual

Aumento de 40% na precisão de matches

Escalabilidade para milhares de candidatos/vagas

Interface intuitiva para usuários não técnicos

📞 Contato
Para mais informações sobre o projeto, entre em contato através do repositório GitHub.

