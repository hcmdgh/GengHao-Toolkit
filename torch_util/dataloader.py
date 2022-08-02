from .imports import * 

__all__ = [
    'get_dataloader',
    'get_dual_dataloader', 
]


def get_dataloader(num_samples: int,
                   batch_size: int,
                   shuffle: bool = True,
                   drop_last: bool = False) -> Iterator[IntArray]:
    if shuffle:              
        perm = np.random.permutation(num_samples)
    else:
        perm = np.arange(num_samples)
    
    if num_samples % batch_size == 0:
        batch_cnt = num_samples // batch_size 
    else:
        if drop_last:
            batch_cnt = num_samples // batch_size
        else:
            batch_cnt = num_samples // batch_size + 1 
            
    for i in range(batch_cnt):
        yield perm[i * batch_size: (i + 1) * batch_size]
        
        
def get_dual_dataloader(num_samples_1: int,
                        num_samples_2: int,
                        batch_size: int,
                        shuffle: bool = True) -> Iterator[tuple[IntArray, IntArray]]:
    assert shuffle 
                      
    perm_1 = np.random.permutation(num_samples_1)
    perm_2 = np.random.permutation(num_samples_2)

    max_len = max(len(perm_1), len(perm_2))
    batch_cnt = math.ceil(max_len / batch_size)
    N = batch_cnt * batch_size 
    
    extended_1 = perm_1
    extended_2 = perm_2
    
    while len(extended_1) < N:
        extended_1 = np.concatenate([extended_1, perm_1])

    while len(extended_2) < N:
        extended_2 = np.concatenate([extended_2, perm_2])

    for i in range(batch_cnt):
        yield extended_1[i * batch_size: (i + 1) * batch_size], extended_2[i * batch_size: (i + 1) * batch_size]
