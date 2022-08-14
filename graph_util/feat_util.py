from .imports import * 
from torch_util import * 

__all__ = [
    'normalize_feature', 
]


def normalize_feature(feat: FloatArrayTensor) -> FloatTensor:
    feat = to_FloatTensor(feat)
    
    out = F.normalize(feat, p=1, dim=-1)
    
    return out 
