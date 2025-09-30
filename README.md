🚀 RECRUT.AI: Otimização Inteligente de Recrutamento
📌 Visão Geral do Projeto
O RECRUT.AI é uma plataforma de Otimização de Recrutamento e Seleção que visa modernizar a triagem de currículos (CVs).

Em vez de usar a filtragem tradicional por palavras-chave, o projeto aplica Deep Learning e Processamento de Linguagem Natural (NLP) semântico para calcular um match inteligente entre o significado completo das vagas e os currículos dos candidatos.

O objetivo é reduzir drasticamente o tempo de triagem do RH, fornecendo um ranking preciso dos candidatos com maior Potencial de Aderência (Probabilidade de Match).

✨ Arquitetura de Deep Learning
A solução é construída sobre um poderoso pipeline de Machine Learning que combina a compreensão de texto com a classificação robusta:

1. 🧠 Embeddings Semânticos (Deep Learning/NLP)
Detalhe	Informação
Modelo	Sentence Transformers (SBERT)
Arquitetura	all-MiniLM-L6-v2
Função	Converte o texto integral dos CVs e das vagas em vetores numéricos de 384 dimensões (embeddings). Esta codificação captura o significado contextual das sentenças.
Benefício	Permite que o sistema entenda a intenção e o contexto, encontrando match mesmo quando termos diferentes são usados (ex: "Conhecimento em AWS" e "Experiência com Amazon Web Services").


2. 🏆 Classificação Avançada (Machine Learning)
Detalhe	Informação
Modelo	XGBoost (eXtreme Gradient Boosting)
Função	Atua como um Classificador Robusto. Ele recebe a concatenação dos vetores semânticos (Candidato + Vaga) e prevê a Probabilidade de Match (uma pontuação de 0 a 1).
Benefício	Gera um ranking final de candidatos ordenado por sua compatibilidade predita, garantindo alta precisão na pré-seleção.


⚙️ Pilares de Infraestrutura e Implementação
Pilar	Tecnologia/Ferramenta	Descrição
Interface de Usuário	Streamlit	Dashboard interativo e ágil para a equipe de RH, com visualização de ranking e painéis de dados.
Infraestrutura Cloud	AWS S3	Armazenamento persistente e seguro de modelos ML e datasets grandes.
Modelo em Produção	Render / FastAPI	Utilizado para hospedar o modelo e API (FastAPI) de predição, permitindo baixa latência e integração futura com sistemas de terceiros.


🔗 Acesso ao Aplicativo
O dashboard interativo do RECRUT.AI está publicado e pode ser acessado no link abaixo.

Markdown

[Acessar RECRUT.AI](https://recrutai.streamlit.app/)
👉 https://recrutai.streamlit.app/



Os arquivos mais pesados (modelos e dados limpos) foram colocados na AWS S3 para facilitar o deploy e garantir a performance, com o Streamlit consumindo diretamente da nuvem.
<img width="1596" height="548" alt="image" src="https://github.com/user-attachments/assets/e4cab819-6c3c-4688-8d93-f6954fe34f7a" />

📂 Estrutura do Repositório
Aqui está a estrutura de pastas do projeto, garantindo o alinhamento correto com o bloco de código:
```
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
└── runtime.txt                  # Versão do Python utilizada
```


⚙️ Fluxo de Trabalho (Pipeline ML)
Etapa	Ferramentas	Descrição
1. Coleta/Processamento	Pandas, utils/	Conversão de JSON para CSVs limpos (applicants_clean.csv, vagas_clean.csv) e padronização.
2. Modelagem Semântica	SBERT, notebooks/	Geração de embeddings (vetores numéricos) dos CVs e Vagas.
3. Treinamento	XGBoost, Scikit-learn	Treinamento do classificador de Match usando os embeddings como features.
4. Deploy e Serviço	Streamlit, AWS S3	O Streamlit serve o dashboard.

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

````# 1. Clonar o repositório
git clone https://github.com/laribar/datathon.git
cd datathon

# 2. Criar ambiente virtual
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
.venv\Scripts\activate       # Windows

# 3. Instalar dependências
pip install -r app/requirements.txt
````
