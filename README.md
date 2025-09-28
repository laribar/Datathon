# 🚀 Projeto Datathon — AI Recruitment

Plataforma de **otimização de recrutamento e seleção** com **Deep Learning** e **NLP semântico (SBERT)** para realizar **match inteligente entre currículos e vagas**.

---

## 📌 Visão Geral
Este projeto busca reduzir o tempo gasto pelo RH em triagens manuais, aplicando modelos de **similaridade semântica** e **classificação de compatibilidade** entre candidatos e vagas.  

A solução combina:
- Processamento e limpeza de dados de CVs/vagas.
- **Embeddings semânticos** com *Sentence Transformers*.
- Classificação com **XGBoost**.
- API em **FastAPI** e dashboard em **Streamlit**.
- Deploy em **Render** para uso em produção.

---

## 📂 Estrutura de Pastas

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
│   ├── Datathon.ipynb
│   └── 01_exploracao_dados.ipynb
│
├── utils/                       # Scripts auxiliares
│   ├── json_to_csv_prospects.py
│   ├── json_to_csv_vagas.py
│   ├── map_files.py
│   ├── transformar.py
│   └── treinamento_modelo_match.py
│
├── .env                         # Variáveis de ambiente
├── render.yaml                   # Configuração Render
├── runtime.txt                   # Versão do Python
└── README.md                     # Documentação


---

## ⚙️ Fluxo do Projeto

1. **📥 Coleta de Dados**  
   - `applicants.json` → currículos  
   - `prospects.json` → candidatos prospectados  
   - `vagas.json` → descrições de vagas  

2. **🧹 Processamento**  
   - Conversão JSON → CSV (`json_to_csv_*`).  
   - Limpeza e padronização de texto (`transformar.py`).  

3. **🔎 Exploração de Dados**  
   - `01_exploracao_dados.ipynb` → análise inicial de skills, vagas e distribuição.  

4. **🧠 Modelagem**  
   - **SBERT Encoder** → embeddings de CVs e vagas.  
   - **XGBoost** → classificador de compatibilidade.  
   - Treinamento automatizado (`treinamento_modelo_match.py`).  

5. **🖥️ Interface**  
   - **Streamlit (`streamlit_app.py`)** → upload de CVs/vagas, visualização de matches.  
   - **FastAPI (`main.py`)** → rotas `/match` e `/batch_match`.  

6. **☁️ Deploy**  
   - Configurado no **Render** via `render.yaml` e `runtime.txt`.  

---

## 📊 Tecnologias

- **Python 3.10+**
- **FastAPI** (API REST)
- **Streamlit** (Dashboard)
- **Sentence Transformers (SBERT)**
- **XGBoost**
- **Pandas / NumPy**
- **Scikit-learn**
- **Render**

---

## 🚀 Como Executar

### 1. Clonar o repositório
```bash
git clone https://github.com/laribar/datathon.git
cd datathon















# Datathon
1. Abra o notebook notebook/Datathon.ipynb
2. Adicione os 3 arquivos disponibilizados no datathon na raiz
   <img width="1747" height="850" alt="image" src="https://github.com/user-attachments/assets/78c960ab-6c31-4742-bdac-7fc82e73252b" />

3. Rode



## Projeto Datathon - Match de Currículos e Vagas

Este projeto utiliza técnicas de Machine Learning e Deep Learning para encontrar o melhor match entre candidatos e vagas, com integração a um backend FastAPI para predição e endpoints de serviço.

Estrutura do projeto
data/ → Contém os arquivos de dados usados no projeto.

processed/ → Dados já tratados e prontos para uso no treino, como pairs.parquet (pares vaga–candidato) e JSONs normalizados (vagas.json, applicants.json, prospects.json).

notebooks/ → Notebooks para exploração, análise e prototipagem do modelo.

models/ → Diretório onde são salvos o modelo treinado (modelo_match_baseline.pkl) e o encoder SBERT (sbert_encoder/) gerados pelo script de treino.

app/main.py → Arquivo principal que inicializa a aplicação FastAPI e registra os routers.

extractor.py → Módulo para leitura e extração de texto de currículos em PDF, usado pelo endpoint /match/pdf.

treinamento_modelo_match.py → Script para treinar o modelo de match usando dados de pares vaga–candidato (pairs.parquet) e salvar o modelo e encoder.

requirements.txt → Lista de dependências necessárias para rodar o projeto.


┌─────────────────────────────────────────────────────────────────────┐
│                           DATA PIPELINE                              │
└─────────────────────────────────────────────────────────────────────┘
      dados brutos (JSON)                      pares prontos p/ treino
  data/vagas.json  applicants.json  ─────────▶ data/processed/pairs.parquet
           prospects.json                           (vaga_text, cv_text, label)

                         ┌──────────────────────────────────────────┐
                         │  treinamento_modelo_match.py             │
                         │  • carrega pairs.parquet                 │
                         │  • gera embeddings (SBERT)               │
                         │  • cria features (concat + |diff| + prod)│
                         │  • treina LogisticRegression             │
                         │  • salva modelo e encoder                │
                         └───────┬───────────────────────┬──────────┘
                                 │                       │
                      models/modelo_match_baseline.pkl   │
                                                         │
                                             models/sbert_encoder/
                                                         │
                                                         ▼

┌─────────────────────────────────────────────────────────────────────┐
│                              BACKEND                                │
└─────────────────────────────────────────────────────────────────────┘
backend/app/main.py
  └── inclui routers:
      - backend/routers/match.py
      - backend/routers/jobs.py, candidates.py, sessions.py ...

backend/services/model.py
  • carrega encoder (models/sbert_encoder/)
  • carrega modelo  (modelo_match_baseline.pkl) [fallback: cosine]
  • gera embeddings e FEATURES idênticas ao treino
  • expõe decisão (score, match, threshold/model info)

backend/routers/match.py
  • POST /match
      - recebe {vaga_text, cv_text}
      - chama service → score/match
  • POST /match/pdf
      - recebe Form(vaga_text) + File(cv_pdf)
      - usa extractor.py para texto do PDF
      - chama service → score/match

extractor.py
  • extrai texto de PDFs (pdfplumber → PyMuPDF)
  • limpeza leve; falha se PDF for imagem (OCR opcional)

┌─────────────────────────────────────────────────────────────────────┐
│                              CONSUMO                                 │
└─────────────────────────────────────────────────────────────────────┘
/docs (Swagger UI)
  • testar POST /match e POST /match/pdf
  • ver payload/resultados
