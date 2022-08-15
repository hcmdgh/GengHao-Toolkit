from .imports import * 

__all__ = [
    'set_cwd', 
    'is_file_exist',
    'deep_walk',
]


def set_cwd(path: str):
    dir_path = os.path.dirname(path)
    
    if dir_path:
        os.chdir(dir_path)


def is_file_exist(path: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) > 0


def deep_walk(path: str) -> tuple[list[str], list[str]]:
    filepath_list = []
    dirpath_list = []

    for root, dirs, files in os.walk(path):
        for filename in files:
            filepath = os.path.join(root, filename)
            filepath_list.append(filepath)
            
        for dirname in dirs:
            dirpath = os.path.join(root, dirname)
            dirpath_list.append(dirpath)
            
    return filepath_list, dirpath_list 
