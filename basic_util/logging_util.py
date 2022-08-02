from .imports import * 

__all__ = [
    'init_log', 
]


def init_log(log_path: Optional[str] = './log.log',
             stdout: bool = True):
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
