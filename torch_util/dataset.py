from .imports import * 
from numpy.random import default_rng 

__all__ = [
    'split_train_val_set',
    'split_train_val_test_set',
]

DEFAULT_SEED = 1428


def split_train_val_set(*,
                        total_cnt: int,
                        train_ratio: float,
                        val_ratio: float,
                        seed: int = DEFAULT_SEED) -> tuple[BoolArray, BoolArray]:
    assert train_ratio + val_ratio == 1. 

    rng = default_rng(seed)
    
    rand_idxs = rng.permutation(total_cnt)
    
    num_train = int(total_cnt * train_ratio)
    
    train_mask = np.zeros(total_cnt, dtype=bool)
    val_mask = np.zeros(total_cnt, dtype=bool)

    train_mask[rand_idxs[:num_train]] = True
    val_mask[rand_idxs[num_train:]] = True
    
    assert np.all(train_mask | val_mask)
    assert np.all(~(train_mask & val_mask))

    return train_mask, val_mask


def split_train_val_test_set(*,
                             total_cnt: int,
                             train_ratio: float,
                             val_ratio: float,
                             test_ratio: float,
                             seed: int = DEFAULT_SEED) -> tuple[BoolArray, BoolArray, BoolArray]:
    assert train_ratio + val_ratio + test_ratio == 1. 

    rng = default_rng(seed)
    
    rand_idxs = rng.permutation(total_cnt)
    
    num_train = int(total_cnt * train_ratio)
    num_val = int(total_cnt * val_ratio)
    
    train_mask = np.zeros(total_cnt, dtype=bool)
    val_mask = np.zeros(total_cnt, dtype=bool)
    test_mask = np.zeros(total_cnt, dtype=bool)

    train_mask[rand_idxs[:num_train]] = True
    val_mask[rand_idxs[num_train: num_train + num_val]] = True
    test_mask[rand_idxs[num_train + num_val:]] = True
    
    assert np.all(train_mask | val_mask | test_mask)
    assert np.all(~(train_mask & val_mask & test_mask))

    return train_mask, val_mask, test_mask 
