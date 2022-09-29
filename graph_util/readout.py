from .imports import * 

__all__ = [
    'readout_nodes', 
]


def readout_nodes(batch_g: dgl.DGLGraph,
                  batch_num_nodes: IntArray,
                  feat: FloatTensor, 
                  op: Literal['sum', 'max', 'mean']) -> FloatTensor:
    num_batch, = batch_num_nodes.shape 
    num_nodes, feat_dim = feat.shape 
    assert np.sum(batch_num_nodes) == batch_g.num_nodes() == num_nodes
    
    readout_list = [] 
    pos = 0 
    
    for i, n in enumerate(batch_num_nodes):
        single_feat = feat[pos : pos + n]
        pos += n 
        
        if op == 'sum':
            single_readout = single_feat.sum(dim=0)
        elif op == 'mean':
            single_readout = single_feat.mean(dim=0)
        elif op == 'max':
            raise NotImplementedError 
        else:
            raise AssertionError 

        readout_list.append(single_readout)
        
    readout = torch.stack(readout_list)
    
    return readout 
