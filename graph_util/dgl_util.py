from .imports import * 
from .bean import * 
from basic_util import * 

__all__ = [
    'save_dgl_graph',
    'load_dgl_graph', 
    'get_edge_index',
    'get_adj_mat', 
]


def get_adj_mat(g: dgl.DGLGraph,
                return_tensor: bool,
                return_sparse: bool):
    adj_mat = g.adj()
    
    if return_tensor and return_sparse:
        return adj_mat  
    elif return_tensor and not return_sparse:
        return adj_mat.to_dense() 
    elif not return_tensor and return_sparse:
        raise NotImplementedError 
    elif not return_tensor and not return_sparse:
        return adj_mat.to_dense().numpy() 
    else:
        raise AssertionError


def hash_graph(g: dgl.DGLGraph) -> str:
    raise NotImplementedError
    edge_index = get_edge_index(g, format='pyg', return_numpy=True)
    
    if isinstance(edge_index, IntArray):
        _bytes = edge_index.tobytes()
    elif isinstance(edge_index, dict):
        items = sorted(edge_index.items()) 
        _bytes = bytes() 

        for _, _edge_index in items:
            _bytes += _edge_index.tobytes() 
    else:
        raise AssertionError 
    
    return hash_by_SHA1(_bytes)


def get_edge_index(g: dgl.DGLGraph,
                   format: str = 'dgl',
                   return_numpy: bool = False):
    format = format.lower().strip() 
                   
    if g.is_homogeneous:
        edge_index = g.edges() 
        
        if format == 'dgl':
            pass  
        elif format == 'pyg':
            edge_index = torch.stack(edge_index)
        else:
            raise AssertionError 

        if return_numpy:
            if isinstance(edge_index, tuple):
                edge_index = (edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy())
            elif isinstance(edge_index, IntTensor):
                edge_index = edge_index.cpu().numpy() 
            else:
                raise AssertionError 
        
        return edge_index
    else:
        edge_index_dict = {} 
        
        for etype in g.canonical_etypes:
            edge_index = g.edges(etype=etype)

            if format == 'dgl':
                edge_index_dict[etype] = edge_index
            elif format == 'pyg':
                edge_index_dict[etype] = torch.stack(edge_index)
            else:
                raise AssertionError 

        if return_numpy:
            for etype, edge_index in edge_index_dict.items():
                if isinstance(edge_index, tuple):
                    edge_index = (edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy())
                elif isinstance(edge_index, IntTensor):
                    edge_index = edge_index.cpu().numpy() 
                else:
                    raise AssertionError 
                
                edge_index_dict[etype] = edge_index 
                
        # 二分图
        if len(edge_index_dict) == 1:
            return next(iter(edge_index_dict.values()))
            
        return edge_index_dict


def homo_graph_2_coo_mat(graph: dgl.DGLGraph) -> sp.coo_matrix:
    raise DeprecationWarning
    num_nodes = graph.num_nodes() 
    edge_index = graph.edges() 
    row = edge_index[0].cpu().numpy().astype(np.int64)
    col = edge_index[1].cpu().numpy().astype(np.int64)
    ones = np.ones_like(row, dtype=np.float32)
    
    coo_mat = sp.coo_matrix((ones, (row, col)), shape=[num_nodes, num_nodes])
    
    return coo_mat


def coo_mat_2_homo_graph(coo_mat: sp.coo_matrix) -> dgl.DGLGraph:
    raise DeprecationWarning 
    num_nodes = coo_mat.shape[0]
    assert coo_mat.shape == (num_nodes, num_nodes)
    
    edge_index = (coo_mat.row, coo_mat.col)
    
    graph = dgl.graph(edge_index, num_nodes=num_nodes)
    
    return graph 


def save_dgl_graph(graph: dgl.DGLGraph,
                   file_path: str):
    if graph.is_homogeneous:
        HomoGraph.from_dgl(graph).save_to_file(file_path)
    else:
        HeteroGraph.from_dgl(graph).save_to_file(file_path)
        
        
def load_dgl_graph(file_path: str) -> dgl.DGLGraph:
    try:
        return HomoGraph.load_from_file(file_path).to_dgl()
    except TypeError:
        return HeteroGraph.load_from_file(file_path).to_dgl()
