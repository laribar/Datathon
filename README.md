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
