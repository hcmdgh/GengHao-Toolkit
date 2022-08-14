from .imports import * 
from .bean import * 

__all__ = [
    'save_dgl_graph',
    'load_dgl_graph', 
    'get_edge_index',
]


def get_edge_index(g: dgl.DGLGraph,
                   format: str):
    format = format.lower().strip() 
                   
    if g.is_homogeneous:
        edge_index = g.edges() 
        
        if format == 'dgl':
            return edge_index 
        elif format == 'pyg':
            return torch.stack(edge_index)
        else:
            raise AssertionError 
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
