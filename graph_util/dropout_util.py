from .imports import * 
from .dgl_util import * 

from torch_geometric.utils.dropout import dropout_adj

__all__ = [
    'dropout_edge',
    'dropout_feature',
]


def dropout_edge(g: dgl.DGLGraph,
                 dropout_ratio: float,
                 force_undirected: bool = False,
                 add_self_loop: bool = True) -> dgl.DGLGraph:
    num_nodes = g.num_nodes() 
    edge_index = get_edge_index(g, format='pyg')
    
    dropped_edge_index, _ = dropout_adj(
        edge_index = edge_index,
        p = dropout_ratio,
        force_undirected = force_undirected, 
    )
    dropped_edge_index = tuple(dropped_edge_index)
    
    dropped_g = dgl.graph(dropped_edge_index, num_nodes=num_nodes)

    if add_self_loop:
        dropped_g = dgl.add_self_loop(dgl.remove_self_loop(dropped_g))
    
    return dropped_g 


def dropout_feature(feat: FloatTensor,
                    dropout_ratio: float) -> FloatTensor:
    device = feat.device 
    num_nodes, feat_dim = feat.shape 
    
    keep_ratio = 1 - dropout_ratio 
    keep_mask = torch.bernoulli(torch.ones(feat_dim, device=device) * keep_ratio).view(1, feat_dim)
    
    dropped_feat = feat * keep_mask 
    
    return dropped_feat 
