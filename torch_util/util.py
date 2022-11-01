from .imports import * 

from io import BytesIO

__all__ = [
    'to_FloatArray',
    'to_IntArray', 
    'to_BoolArray',
    'to_FloatTensor',
    'to_IntTensor', 
    'to_BoolTensor',
    'seed_all',
    'auto_select_gpu',
    'set_device',
    'get_device',
    'is_on_cpu',
    'is_on_gpu',
    'load_model_state',
    'save_model_state',
    'compare_tensor', 
]

_device = torch.device('cpu')


def compare_tensor(v1, v2):
    if isinstance(v1, FloatTensor):
        v1 = v1.detach().cpu().numpy() 
    if isinstance(v2, FloatTensor):
        v2 = v2.detach().cpu().numpy()

    print(f"The shape of v1: {v1.shape}")
    print(f"The shape of v2: {v2.shape}")
    print(f"The difference between v1 and v2: {np.sum(np.power(v1 - v2, 2))}")


def to_ndarray(v, dtype) -> FloatArray:
    if isinstance(v, list):
        return np.array(v, dtype=dtype)
    elif isinstance(v, ndarray):
        return v.astype(dtype)
    elif isinstance(v, Tensor):
        return v.detach().cpu().numpy().astype(dtype) 
    else:
        raise TypeError 


def to_FloatArray(v: Any) -> FloatArray:
    return to_ndarray(v, np.float32)


def to_IntArray(v: Any) -> IntArray:
    return to_ndarray(v, np.int64)


def to_BoolArray(v: Any) -> BoolArray:
    return to_ndarray(v, bool)


def to_FloatTensor(v: Any) -> FloatTensor:
    return torch.from_numpy(to_ndarray(v, np.float32))


def to_IntTensor(v: Any) -> IntTensor:
    return torch.from_numpy(to_ndarray(v, np.int64))


def to_BoolTensor(v: Any) -> BoolTensor:
    return torch.from_numpy(to_ndarray(v, bool))


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


def load_model_state(model: nn.Module, data: bytes):
    bio = BytesIO(data)
    model.load_state_dict(torch.load(bio))


def save_model_state(model: nn.Module) -> bytes:
    bio = BytesIO()
    torch.save(model.state_dict(), bio)
    return bio.getvalue()


def seed_all(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    import dgl 
    dgl.seed(seed)
    dgl.random.seed(seed)
    

def auto_select_gpu(use_gpu: bool = True) -> torch.device:
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
