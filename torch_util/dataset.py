from .imports import * 
from numpy.random import default_rng 

__all__ = [
    'random_split_dataset',
]

DEFAULT_SEED = 14285


def random_split_dataset(total_cnt: int,
                         *, 
                         train_ratio: float,
                         val_ratio: float,
                         test_ratio: float, 
                         seed: int = DEFAULT_SEED,
                         return_numpy: bool = False) -> tuple:
    assert math.isclose(train_ratio + val_ratio + test_ratio, 1.) 

    rng = default_rng(seed)
    perm = rng.permutation(total_cnt)
    
    train_cnt = int(total_cnt * train_ratio)
    val_cnt = int(total_cnt * val_ratio)
    
    train_mask = np.zeros(total_cnt, dtype=bool)
    val_mask = np.zeros(total_cnt, dtype=bool)
    test_mask = np.zeros(total_cnt, dtype=bool)

    train_mask[perm[:train_cnt]] = True
    val_mask[perm[train_cnt : train_cnt + val_cnt]] = True
    test_mask[perm[train_cnt + val_cnt:]] = True
    
    assert np.all(train_mask | val_mask | test_mask)
    assert np.all(~(train_mask & val_mask & test_mask))
    assert np.sum(train_mask) == train_cnt and np.sum(val_mask) == val_cnt 

    if return_numpy:
        return train_mask, val_mask, test_mask 
    else:
        return (
            torch.tensor(train_mask, dtype=torch.bool),
            torch.tensor(val_mask, dtype=torch.bool),
            torch.tensor(test_mask, dtype=torch.bool),
        )
