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

```bash
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
│   ├── Datathon.ipynb    #notebook que faz pré processamnento / feature engineering / treinamento /validação
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
