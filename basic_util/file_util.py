from .imports import * 

__all__ = [
    'set_cwd', 
    'is_file_exist',
    'deep_walk',
    'get_file_mtime',
    'path_join', 
]


def path_join(*paths: str) -> str:
    assert paths 
    
    res = '/'.join(
        path.replace('\\', '/').rstrip('/')
        for path in paths 
    )
    
    return res 


def set_cwd(path: str):
    dir_path = os.path.dirname(path)
    
    if dir_path:
        os.chdir(dir_path)


def is_file_exist(path: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) > 0


def deep_walk(root_path: str,
              return_absolute_path: bool = True,
              filename_rule: Optional[Callable] = None,
              dirname_rule: Optional[Callable] = None,) -> tuple[list[str], list[str]]:
    filepath_list = []
    dirpath_list = []

    for root, dirs, files in os.walk(root_path):
        for filename in files:
            if not filename_rule or filename_rule(filename):
                filepath = os.path.join(root, filename)
                filepath_list.append(filepath)
            
        _dirs = list(dirs)
        dirs.clear() 
            
        for dirname in _dirs:
            if not dirname_rule or dirname_rule(dirname):
                dirs.append(dirname)

                dirpath = os.path.join(root, dirname)
                dirpath_list.append(dirpath)
                
    if not return_absolute_path:
        filepath_list = [os.path.relpath(x, start=root_path) for x in filepath_list]
        dirpath_list = [os.path.relpath(x, start=root_path) for x in dirpath_list]

    # 化\为/，确保在各个系统上一致
    filepath_list = [x.replace('\\', '/') for x in filepath_list]
    dirpath_list = [x.replace('\\', '/') for x in dirpath_list]
            
    return filepath_list, dirpath_list 


def get_file_mtime(path: str) -> datetime:
    timestamp = os.path.getmtime(path)
    time_obj = time.localtime(timestamp)
    
    dt = datetime(
        year = time_obj.tm_year,
        month = time_obj.tm_mon,
        day = time_obj.tm_mday,
        hour = time_obj.tm_hour,
        minute = time_obj.tm_min,
        second = time_obj.tm_sec,
    )
    
    return dt 
