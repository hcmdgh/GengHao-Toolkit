from .imports import * 
from basic_util import * 
import io 

__all__ = [
    'hash_tensor',
    'hash_model_parameters',
]


def hash_tensor(tensor: Tensor) -> str:
    dtype = str(tensor.dtype)
    data = json.dumps(tensor.tolist())
    
    info = f"dtype: {dtype}, data: {data}"

    digest = hash_by_MD5(info.encode())
    
    return digest
    
    
def hash_model_parameters(model: nn.Module) -> str:
    digest_list = [] 
    
    for p in model.parameters():
        digest = hash_tensor(p)
        digest_list.append(digest)
        
    overall_digest = hash_by_MD5(','.join(digest_list).encode())

    return overall_digest 
