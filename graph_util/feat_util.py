from .imports import * 

__all__ = [
    'normalize_feature', 
]


def normalize_feature(feat: FloatTensor) -> FloatTensor:
    feat = feat - feat.min()
    feat.div_(feat.sum(dim=-1, keepdim=True).clamp_(min=1.))
    
    return feat 
