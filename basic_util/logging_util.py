from .imports import * 
from .datetime_util import *

__all__ = [
    'init_log', 
    'log_info',
    'log_warning',
    'log_error',
]

_use_stdout = True 
_log_fp = None 


def init_log(log_path: Optional[str] = './log.log',
             stdout: bool = True):
    global _use_stdout, _log_fp
    
    if log_path and _log_fp is None:
        _log_fp = open(log_path, 'w', encoding='utf-8')
        
    _use_stdout = stdout

    
def _log(content: str,
         level: str):
    msg = f"{datetime2str(datetime.now())} [{level}] {content}"
         
    if _use_stdout:
        print(msg, flush=True)
        
    if _log_fp is not None:
        print(msg, file=_log_fp, flush=True)


def log_info(content: str):
    _log(content, 'INFO')
    
    
def log_warning(content: str):
    _log(content, 'WARNING')
    
    
def log_error(content: str):
    _log(content, 'ERROR')
