import ijson
import tkinter as tk
from tkinter import filedialog
import csv
import json
import re

def select_json_file():
    """
    Abre uma janela de diálogo para selecionar um arquivo JSON.
    """
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Selecione um arquivo JSON",
        filetypes=[("Arquivos JSON", "*.json")]
    )
    return file_path

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

def standardize_and_transform_json(file_path):
    """
    Padroniza os dados do JSON e os transforma em um CSV limpo para importação no Supabase.
    """
    if not file_path:
        print("Nenhum arquivo selecionado. Encerrando.")
        return

    print(f"Processando o arquivo: {file_path}")

    try:
        with open(file_path, 'rb') as f:
            parser = ijson.kvitems(f, '')

            all_data = []
            processed_count = 0
            
            # Mapeamento para renomear e consolidar colunas
            column_mapping = {
                'infos_basicas_codigo_profissional': 'codigo_profissional',
                'infos_basicas_nome': 'nome',
                'informacoes_pessoais_nome': 'nome_pessoal', # Mantemos para ver se são diferentes
                'infos_basicas_email': 'email',
                'informacoes_pessoais_email': 'email_pessoal',
                'infos_basicas_telefone': 'telefone',
                'informacoes_pessoais_telefone_celular': 'telefone_pessoal',
                'infos_basicas_local': 'local',
                'informacoes_pessoais_endereco': 'endereco',
                'formacao_e_idiomas_nivel_ingles': 'nivel_ingles',
                'formacao_e_idiomas_nivel_espanhol': 'nivel_espanhol',
                'formacao_e_idiomas_outro_idioma': 'outro_idioma',
                'formacao_e_idiomas_nivel_academico': 'nivel_academico',
                'informacoes_profissionais_titulo_profissional': 'titulo_profissional',
                'informacoes_profissionais_area_atuacao': 'area_atuacao',
                'informacoes_profissionais_remuneracao': 'remuneracao',
                'informacoes_profissionais_nivel_profissional': 'nivel_profissional',
                'informacoes_pessoais_data_nascimento': 'data_nascimento',
                'informacoes_pessoais_estado_civil': 'estado_civil',
                'informacoes_pessoais_sexo': 'sexo',
                'cv_pt': 'curriculo_pt',
            }

            for candidate_id, item in parser:
                if not isinstance(item, dict):
                    continue

                processed_count += 1
                
                flat_data = {'id': candidate_id}

                for main_key, sub_data in item.items():
                    if isinstance(sub_data, dict):
                        for sub_key, value in sub_data.items():
                            flat_key = f"{main_key}_{sub_key}"
                            # Verifica se a chave está no mapeamento
                            if flat_key in column_mapping:
                                clean_key = column_mapping[flat_key]
                                flat_data[clean_key] = clean_text(value)
                    else:
                        if main_key in column_mapping:
                            clean_key = column_mapping[main_key]
                            flat_data[clean_key] = clean_text(sub_data)
                
                all_data.append(flat_data)

            print("\n--- Análise Concluída ---")
            print(f"Total de itens (candidatos) processados: {processed_count}")
            
            # Pega as chaves do primeiro item para definir o cabeçalho
            if not all_data:
                print("Nenhum dado encontrado para processamento.")
                return
            
            header = list(all_data[0].keys())
            
            # Garante que o ID é a primeira coluna
            header.remove('id')
            header = ['id'] + sorted(header)

            output_file = file_path.replace('.json', '_clean.csv')
            
            # Salva os dados no arquivo CSV limpo
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=header, quoting=csv.QUOTE_ALL)
                writer.writeheader()
                
                for row in all_data:
                    writer.writerow({k: row.get(k, '') for k in header})

            print(f"\nDados padronizados salvos com sucesso em: {output_file}")
            print("---")
            
    except FileNotFoundError:
        print(f"Erro: O arquivo não foi encontrado em {file_path}")
    except Exception as e:
        print(f"Ocorreu um erro: {e}")

if __name__ == "__main__":
    json_file_path = select_json_file()
    standardize_and_transform_json(json_file_path)