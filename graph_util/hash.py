from .imports import * 
from basic_util import * 
from torch_util import * 

__all__ = [
    'hash_g', 
    'hash_g_list', 
]


def hash_g(g: dgl.DGLGraph) -> str:
    num_nodes, num_edges = g.num_nodes(), g.num_edges() 
    
    edge_index = g.edges()
    edge_index = torch.stack(edge_index)
    assert edge_index.shape == (2, num_edges)
    
    info = f"num_nodes: {num_nodes}, edge_index_digest: {hash_tensor(edge_index)}"
    
    digest = hash_by_MD5(info.encode())
    
    return digest 


def hash_g_list(g_list: list[dgl.DGLGraph]) -> str:
    digest_list = [
        hash_g(g)
        for g in g_list 
    ]
    
    overall_digest = hash_by_MD5(','.join(digest_list).encode())
    
    return overall_digest 
