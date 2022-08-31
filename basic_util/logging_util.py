from .imports import * 

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
    
    if log_path:
        _log_fp = open(log_path, 'w', encoding='utf-8')
    else:
        _log_fp = None 
        
    _use_stdout = stdout

    
def _log(content: str,
         level: str):

         
    if _use_stdout:
        print()


def log_info(content: str):
    pass 
    
    
def log_warning(content: str):
    init_log()
    logging.warning(content)
    
    
def log_error(content: str):
    init_log()
    logging.error(content)
