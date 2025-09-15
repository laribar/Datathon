import os

def get_directory_size(path='.'):
    """Calcula o tamanho de um diretório em bytes."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size

def human_readable_size(size, decimal_places=2):
    """Converte bytes para um formato legível."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f} {unit}"

def map_large_files(root_path='.', min_size_mb=10):
    """
    Mapeia arquivos grandes e o tamanho de diretórios.
    
    Args:
        root_path (str): O caminho raiz para iniciar a busca.
        min_size_mb (int): Tamanho mínimo em MB para um arquivo ser considerado "grande".
    """
    min_size_bytes = min_size_mb * 1024 * 1024
    
    print("Mapeando arquivos e diretórios grandes...")
    print("=" * 40)
    
    large_files = []
    
    # Mapear o tamanho de cada subdiretório
    subdirs_info = []
    for entry in os.scandir(root_path):
        if entry.is_dir():
            dir_size = get_directory_size(entry.path)
            subdirs_info.append((entry.path, dir_size))
            
            # Mapear arquivos grandes dentro de cada diretório
            for dirpath, dirnames, filenames in os.walk(entry.path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if not os.path.islink(fp) and os.path.getsize(fp) > min_size_bytes:
                        large_files.append((fp, os.path.getsize(fp)))

    # Exibir o tamanho dos diretórios
    subdirs_info.sort(key=lambda x: x[1], reverse=True)
    print("Tamanho dos Diretórios:")
    for path, size in subdirs_info:
        print(f"  {path}: {human_readable_size(size)}")
    
    print("\n" + "=" * 40)
    
    # Exibir arquivos grandes
    large_files.sort(key=lambda x: x[1], reverse=True)
    if large_files:
        print(f"Arquivos Maiores que {min_size_mb} MB:")
        for path, size in large_files:
            print(f"  {path}: {human_readable_size(size)}")
    else:
        print(f"Nenhum arquivo maior que {min_size_mb} MB encontrado.")

# Execute o script a partir do diretório raiz do seu projeto
if __name__ == "__main__":
    map_large_files(root_path='.')