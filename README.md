# Datathon
1. Abra o notebook notebook/Datathon.ipynb
2. Adicione os 3 arquivos disponibilizados no datathon na raiz
   <img width="1747" height="850" alt="image" src="https://github.com/user-attachments/assets/78c960ab-6c31-4742-bdac-7fc82e73252b" />

3. Rode



#Para utilizar no visual studio.

# Clonar direto no branch datathon-clean
git clone -b datathon-clean https://github.com/laribar/Datathon.git


# 1. Garantir que está dentro da pasta do projeto
cd C:\Datathon

# 2. Criar (opcional, mas recomendado) um ambiente virtual
python -m venv venv

# 3. Ativar o ambiente virtual
# No PowerShell
.\venv\Scripts\Activate

# 4. Instalar as dependências do requirements.txt
pip install -r api/requirements.txt


#para atualizar
git add .
git commit -m "comentario"
git push origin datathon-clean
