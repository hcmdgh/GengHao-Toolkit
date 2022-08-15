import random 

__all__ = [
    'generate_random_str',
]


def generate_random_str(length: int,
                        lowercase: bool = False,
                        uppercase: bool = False,
                        digit: bool = False) -> str:
    candidate = ''
    
    if lowercase:
        candidate += 'abcdefghijklmnopqrstuvwxyz'
    if uppercase:
        candidate += 'abcdefghijklmnopqrstuvwxyz'.upper() 
    if digit:
        candidate += '0123456789'
        
    assert candidate 
    
    res = ''.join(random.choices(candidate, k=length)) 
    
    return res 
