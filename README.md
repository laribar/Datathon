🚀 Projeto RECRUT.AI: Otimização Inteligente de Recrutamento com Deep Learning
📌 Visão Geral da Solução
O RECRUT.AI é uma plataforma de Otimização de Recrutamento e Seleção que nasceu com o objetivo de revolucionar a triagem de currículos. Ao invés de depender de filtros manuais e buscas por palavras-chave (métodos tradicionais e demorados do RH), o projeto aplica Deep Learning e Processamento de Linguagem Natural (NLP) semântico para calcular um match inteligente entre o significado completo das vagas e os currículos dos candidatos.

O foco central é reduzir drasticamente o tempo de triagem e aumentar a qualidade das pré-seleções, fornecendo ao time de RH um ranking preciso dos candidatos com maior Potencial de Aderência (Probabilidade de Match).

✨ Arquitetura de Deep Learning e Modelos
A inteligência do RECRUT.AI se apoia em dois pilares de Machine Learning para processar e classificar informações textuais complexas:

1. Deep Learning Semântico (Sentence Transformers - SBERT)
Tipo de Deep Learning/Modelo: Embeddings de Linguagem (NLP), utilizando o modelo Sentence Transformers (SBERT), especificamente a arquitetura all-MiniLM-L6-v2.

Função: O SBERT é o motor semântico. Ele converte o texto integral dos currículos (CVs) e o texto completo das vagas em vetores numéricos de 384 dimensões (embeddings).

Vantagem: Essa codificação captura o significado contextual das palavras e sentenças, não apenas a presença de termos. Isso permite que a plataforma identifique um candidato com "Conhecimento em nuvem AWS" como compatível com uma vaga que pede "Experiência com Amazon Web Services", mesmo que as palavras não sejam idênticas.

2. Classificação Avançada (XGBoost)
Tipo de Modelo: Boosting de Árvores de Decisão, utilizando o algoritmo XGBoost.

Função: Após a fase de embedding, o modelo XGBoost atua como um Classificador Robusto. Ele recebe como input a concatenação dos vetores semânticos do Candidato e da Vaga (além de possíveis features de metadados) e é treinado para prever a Probabilidade de Match (uma pontuação de 0 a 1) de que aquele candidato é o ideal para a vaga.

Vantagem: O XGBoost é conhecido por sua alta precisão e velocidade, fornecendo um ranking final de candidatos ordenado por sua compatibilidade predita.

🌐 Pilares de Infraestrutura e Implementação
Pilar	Tecnologia/Ferramenta	Descrição
Interface de Usuário	Streamlit	Dashboard interativo e ágil para seleção de vagas, visualização do ranking de candidatos, e análise explicativa (Explicabilidade).
Infraestrutura Cloud	AWS S3	Armazenamento persistente de datasets brutos, modelos ML grandes (XGBoost e SBERT) e os embeddings pré-calculados, garantindo escalabilidade.


Export to Sheets
🔗 Acesso ao Aplicativo
O dashboard interativo do RECRUT.AI está publicado e pode ser acessado em:
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
