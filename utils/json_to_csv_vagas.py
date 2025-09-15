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

def flatten_json(y):
    out = {}
    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x
    flatten(y)
    return out

def get_all_keys(data):
    all_keys = set()
    for item in data:
        flattened_item = flatten_json(data[item])
        all_keys.update(flattened_item.keys())
    return sorted(list(all_keys))

def process_data(input_file, output_file, column_mapping):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Erro: O arquivo '{input_file}' não foi encontrado.")
        return

    flat_data = []
    processed_items = 0

    for item_id, item_data in data.items():
        processed_items += 1
        flat_item = flatten_json(item_data)
        flat_item['id'] = item_id
        flat_data.append(flat_item)

    if not flat_data:
        print("Total de itens processados: 0. Verifique se o arquivo JSON possui a estrutura de dicionário esperada.")
        return

    output_columns = ['id'] + list(column_mapping.values())
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=output_columns)
        writer.writeheader()
        
        for item in flat_data:
            row = {}
            for flat_key, csv_name in column_mapping.items():
                row[csv_name] = clean_text(item.get(flat_key, ''))
            row['id'] = item.get('id', '')
            writer.writerow(row)
    
    print(f"Total de itens processados: {processed_items}")
    print(f"O arquivo CSV foi salvo como '{output_file}'")

def main():
    # Mapeamento de colunas para o arquivo vagas.json
    column_mapping = {
        # Informações Básicas
        'informacoes_basicas_data_requicisao': 'data_requisicao',
        'informacoes_basicas_limite_esperado_para_contratacao': 'limite_contratacao',
        'informacoes_basicas_titulo_vaga': 'titulo_vaga',
        'informacoes_basicas_cliente': 'cliente',
        'informacoes_basicas_tipo_contratacao': 'tipo_contratacao',
        'informacoes_basicas_prazo_contratacao': 'prazo_contratacao',
        'informacoes_basicas_objetivo_vaga': 'objetivo_vaga',
        'informacoes_basicas_prioridade_vaga': 'prioridade_vaga',
        'informacoes_basicas_origem_vaga': 'origem_vaga',
        
        # Perfil da Vaga
        'perfil_vaga_pais': 'pais',
        'perfil_vaga_estado': 'estado',
        'perfil_vaga_cidade': 'cidade',
        'perfil_vaga_regiao': 'regiao',
        'perfil_vaga_nivel profissional': 'nivel_profissional',
        'perfil_vaga_nivel_academico': 'nivel_academico',
        'perfil_vaga_nivel_ingles': 'nivel_ingles',
        'perfil_vaga_nivel_espanhol': 'nivel_espanhol',
        'perfil_vaga_outro_idioma': 'outro_idioma',
        'perfil_vaga_areas_atuacao': 'areas_atuacao',
        'perfil_vaga_principais_atividades': 'principais_atividades',
        'perfil_vaga_competencia_tecnicas_e_comportamentais': 'competencias',
        'perfil_vaga_demais_observacoes': 'observacoes',
        'perfil_vaga_viagens_requeridas': 'viagens_requeridas',
        'perfil_vaga_habilidades_comportamentais_necessarias': 'habilidades_comportamentais',
        'beneficios_valor_venda': 'salario',
    }

    root = tk.Tk()
    root.withdraw()
    
    input_file_path = filedialog.askopenfilename(
        title="Selecione o arquivo JSON",
        filetypes=[("Arquivos JSON", "*.json")]
    )
    
    if not input_file_path:
        print("Nenhum arquivo selecionado. Encerrando o script.")
        return
    
    output_file_path = os.path.splitext(input_file_path)[0] + '_clean.csv'
    
    # Processa e salva os dados
    process_data(input_file_path, output_file_path, column_mapping)

if __name__ == "__main__":
    main()