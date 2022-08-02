from .imports import * 
from .metric import * 

from io import BytesIO

__all__ = [
    'seed_all',
    'auto_set_device',
    'set_device',
    'get_device',
    'is_on_cpu',
    'is_on_gpu',
    'load_model_state',
    'save_model_state',
]

_device = torch.device('cpu')

_model_state_bytes: Optional[bytes] = None 


def is_on_cpu(obj: Any) -> bool:
    device_type = obj.device.type 
    
    if device_type == 'cpu':
        return True 
    elif device_type == 'cuda':
        return False 
    else:
        raise AssertionError 
    
    
def is_on_gpu(obj: Any) -> bool:
    return not is_on_cpu(obj)


def load_model_state(model: nn.Module, file_path: Optional[str] = None):
    if file_path:
        model.load_state_dict(torch.load(file_path))
    else:
        assert _model_state_bytes
        
        bio = BytesIO(_model_state_bytes)
        model.load_state_dict(torch.load(bio))


def save_model_state(model: nn.Module, file_path: Optional[str] = None):
    if file_path:
        torch.save(model.state_dict(), file_path)
    else:
        bio = BytesIO()
        torch.save(model.state_dict(), bio)

        global _model_state_bytes
        _model_state_bytes = bio.getvalue()


def seed_all(seed: Optional[int]):
    if not seed:
        return 
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    dgl.seed(seed)
    dgl.random.seed(seed)


def auto_set_device(use_gpu: bool = True) -> torch.device:
    global _device 

    if not use_gpu:
        _device = torch.device('cpu')
        return _device
    
    exe_res = os.popen('gpustat --json').read() 
    
    state_dict = json.loads(exe_res)
    
    gpu_infos = [] 
    
    for gpu_entry in state_dict['gpus']:
        gpu_id = int(gpu_entry['index'])
        used_mem = int(gpu_entry['memory.used'])

        gpu_infos.append((used_mem, gpu_id))
    
    gpu_infos.sort()
    
    _device = torch.device(f'cuda:{gpu_infos[0][1]}')
    
    return _device 


def set_device(device_name: str):
    global _device
    _device = torch.device(device_name)
    
    
def get_device() -> torch.device:
    return _device 
