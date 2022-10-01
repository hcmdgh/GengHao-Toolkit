from .imports import * 


__all__ = [
    'calc_f1_micro',
    'calc_f1_macro',
    'calc_acc',
    'calc_cosine_similarity',
]


def calc_f1_micro() -> float:
    raise DeprecationWarning


def calc_f1_macro() -> float:
    raise DeprecationWarning


def calc_acc(input: Union[ndarray, Tensor],
             target: Union[ndarray, Tensor]) -> float:
    if isinstance(input, Tensor):
        input = input.detach().cpu().numpy() 
    if isinstance(target, Tensor):
        target = target.detach().cpu().numpy() 
    
    # 第1种情况-多分类单标签：input = int[N], target = int[N]
    # 第2种情况-多分类单标签：input = float[N, D], target = int[N]
    # 第3种情况-多分类多标签：input = int[N, D], target = int[N, D]
    
    # 第1种情况-多分类单标签：input = int[N], target = int[N]
    if input.ndim == 1:
        assert input.dtype == target.dtype == np.int64 
        N = len(input)
        assert input.shape == target.shape == (N,) 
        
        acc = (input == target).mean() 
        
        return float(acc) 
     
    # 第2种情况-多分类单标签：input = float[N, D], target = int[N]
    elif input.ndim == 2 and target.ndim == 1:
        assert input.dtype == np.float32 and target.dtype == np.int64  
        N, D = input.shape 
        assert target.shape == (N,)
        
        input = np.argmax(input, axis=-1) 
        
        acc = (input == target).mean() 
        
        return float(acc) 
    
    # 第3种情况-多分类多标签：input = int[N, D], target = int[N, D]
    elif input.ndim == 2 and target.ndim == 2:
        N, D = input.shape 
        assert input.dtype == target.dtype == np.int64 
        assert target.shape == (N, D)
        
        acc = np.all(input == target, axis=-1).mean() 
        
        return float(acc) 
     
    else:
        raise AssertionError


def calc_cosine_similarity(h1: FloatTensor, 
                           h2: FloatTensor) -> FloatTensor:
    N, D = h1.shape 
    assert h1.shape == h2.shape 
                           
    h1 = F.normalize(h1, p=2, dim=-1)
    h2 = F.normalize(h2, p=2, dim=-1)

    out = torch.mm(h1, h2.T)
    assert out.shape == (N, N)
    
    return out 
