from .imports import * 

__all__ = [
    'set_cwd',
]


def set_cwd(path: str):
    dir_path = os.path.dirname(path)
    
    if dir_path:
        os.chdir(dir_path)
