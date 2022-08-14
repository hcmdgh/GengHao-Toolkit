from .imports import * 

__all__ = [
    'get_activation_func', 
]


def get_activation_func(act: Union[str, Callable]) -> Callable:
    if not isinstance(act, str):
        return act 
    
    act = act.lower().strip() 
    
    if act == 'relu':
        return nn.ReLU() 
    elif act == 'sigmoid':
        return nn.Sigmoid() 
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'leakyrelu':
        return nn.LeakyReLU() 
    elif act == 'prelu':
        return nn.PReLU()
    elif act == 'rrelu':
        return nn.RReLU()
    elif act == 'elu':
        return nn.ELU()
    elif act == 'none':
        return nn.Identity()
    else:
        raise AssertionError 
