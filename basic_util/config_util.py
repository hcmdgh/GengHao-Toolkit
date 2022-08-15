from .imports import * 
from .file_util import * 

__all__ = [
    'init_config',
    'get_config',
    'set_config',
]

_config_dict = dict() 

_yaml_path = ''


def init_config(yaml_path: str):
    global _yaml_path 
    assert yaml_path 
    _yaml_path = yaml_path 
    
    if not is_file_exist(yaml_path):
        _config_dict.clear() 
    else:
        with open(yaml_path, 'r', encoding='utf-8') as fp:
            _dict = yaml.safe_load(fp)
            assert isinstance(_dict, dict)
            
            _config_dict.clear()
            _config_dict.update(_dict)


def get_config(config_path: str,
               default: Any = ...) -> Any:
    config_path_list = config_path.strip().split('.')
    
    try:
        val = _config_dict
        
        for item in config_path_list:
            val = val[item]

        return val 
    except Exception:
        if default == ...:
            raise KeyError 
        else:
            return default 


def set_config(config_path: str,
               val: Any):
    config_path_list = config_path.strip().split('.')
    
    _dict = _config_dict
        
    for item in config_path_list[:-1]:
        if item not in _dict:
            _dict[item] = dict() 
            
        _dict = _dict[item]
        
    _dict[config_path_list[-1]] = val 

    assert _yaml_path 
    
    with open(_yaml_path, 'w', encoding='utf-8') as fp:
        yaml.safe_dump(_config_dict, fp, allow_unicode=True)
