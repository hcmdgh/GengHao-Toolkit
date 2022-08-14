from .imports import * 

__all__ = [
    'init_log', 
    'log_info',
    'log_warning',
    'log_error',
]

_has_init_log = False 


def init_log(log_path: Optional[str] = './log.log',
             stdout: bool = True):
    global _has_init_log
    
    if _has_init_log:
        return
    
    _has_init_log = True  
             
    handlers = []
             
    if log_path:
        handlers.append(logging.FileHandler(log_path, 'w', encoding='utf-8'))
    
    if stdout:
        handlers.append(logging.StreamHandler())
    
    logging.basicConfig(
        format = '%(asctime)s [%(levelname)s] %(message)s',
        datefmt = '%Y-%m-%d %H:%M:%S',
        handlers = handlers,
        level = logging.INFO,
    )


def log_info(content: str):
    init_log()
    logging.info(content)
    
    
def log_warning(content: str):
    init_log()
    logging.warning(content)
    
    
def log_error(content: str):
    init_log()
    logging.error(content)
