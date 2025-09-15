import json
import csv
import os
import tkinter as tk
from tkinter import filedialog
import re

def clean_text(text):
    """
    Remove quebras de linha, tabulações e espaços extras de um texto.
    """
    if not isinstance(text, str):
        return text
    # Remove quebras de linha e tabulações
    text = re.sub(r'[\n\r\t]', ' ', text)
    # Remove múltiplos espaços em branco
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_prospects_json(input_file, output_file):
    """
    Processa o arquivo JSON de prospects e o transforma em um CSV limpo.
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Erro: O arquivo '{input_file}' não foi encontrado.")
        return
    except json.JSONDecodeError:
        print(f"Erro: O arquivo '{input_file}' não é um JSON válido.")
        return

    prospects_data = []
    processed_vacancies = 0
    processed_prospects = 0

    # Itera sobre cada vaga no dicionário
    for vacancy_id, vacancy_info in data.items():
        processed_vacancies += 1
        
        # Pega as informações da vaga (título, modalidade, etc.)
        vacancy_title = clean_text(vacancy_info.get('titulo', ''))
        vacancy_modality = clean_text(vacancy_info.get('modalidade', ''))
        
        # Itera sobre a lista de prospects dentro de cada vaga
        prospects_list = vacancy_info.get('prospects', [])
        
        for prospect in prospects_list:
            processed_prospects += 1
            
            # Extrai os dados do prospect
            prospect_info = {
                'id_vaga': vacancy_id,
                'titulo_vaga': vacancy_title,
                'modalidade_vaga': vacancy_modality,
                'nome': clean_text(prospect.get('nome', '')),
                'codigo_candidato': clean_text(prospect.get('codigo', '')),
                'situacao_candidato': clean_text(prospect.get('situacao_candidado', '')),
                'data_candidatura': clean_text(prospect.get('data_candidatura', '')),
                'ultima_atualizacao': clean_text(prospect.get('ultima_atualizacao', '')),
                'comentario': clean_text(prospect.get('comentario', '')),
                'recrutador': clean_text(prospect.get('recrutador', '')),
            }
            prospects_data.append(prospect_info)

    if not prospects_data:
        print("Nenhum prospect encontrado para processamento.")
        return

    # Define o cabeçalho do CSV
    fieldnames = ['id_vaga', 'titulo_vaga', 'modalidade_vaga', 'nome', 'codigo_candidato', 
                  'situacao_candidato', 'data_candidatura', 'ultima_atualizacao', 
                  'comentario', 'recrutador']

    # Salva os dados no arquivo CSV limpo
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(prospects_data)

    print("\n--- Análise Concluída ---")
    print(f"Total de vagas processadas: {processed_vacancies}")
    print(f"Total de candidatos (prospects) processados: {processed_prospects}")
    print(f"\nDados padronizados salvos com sucesso em: {output_file}")

def main():
    root = tk.Tk()
    root.withdraw()
    
    input_file_path = filedialog.askopenfilename(
        title="Selecione o arquivo JSON de prospects",
        filetypes=[("Arquivos JSON", "*.json")]
    )
    
    if not input_file_path:
        print("Nenhum arquivo selecionado. Encerrando o script.")
        return

    output_file_path = os.path.splitext(input_file_path)[0] + '_clean.csv'
    
    process_prospects_json(input_file_path, output_file_path)

if __name__ == "__main__":
    main()