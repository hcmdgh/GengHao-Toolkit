from .imports import * 

__all__ = [
    'GraphDataLoader',
]


class GraphDataLoader:
    def __init__(self,
                 g_list: list[dgl.DGLGraph],
                 label_list: list[int],
                 batch_size: int,
                 shuffle: bool):
        self.g_list = np.array(g_list, dtype=object)
        self.label_list = np.array(label_list, dtype=np.int64)
        self.N = len(g_list)
        assert self.g_list.shape == self.label_list.shape == (self.N,)
        self.batch_size = batch_size
        self.shuffle = shuffle 
        
    def __iter__(self):
        if self.shuffle:
            perm = np.random.permutation(self.N)
        else:
            perm = np.arange(self.N)
            
        for i in range(0, self.N, self.batch_size):
            batch_idx = perm[i : i + self.batch_size]
            
            batch_g_list = self.g_list[batch_idx]
            batch_label_list = self.label_list[batch_idx]

            yield batch_g_list, batch_label_list 
