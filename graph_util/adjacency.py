from .imports import * 

__all__ = [
    'convert_edge_index_to_adj_list',
    'convert_g_to_adj_list',
    'convert_hg_to_adj_list',
]


def convert_edge_index_to_adj_list(edge_index: EdgeIndex) -> dict[int, list[int]]:
    src_edge_index, dest_edge_index = edge_index 

    adj_list: dict[int, list[int]] = defaultdict(list)
    
    for src_nid, dest_nid in zip(src_edge_index.tolist(), dest_edge_index.tolist()):
        adj_list[src_nid].append(dest_nid)
        
    return adj_list 


def convert_g_to_adj_list(g: dgl.DGLGraph) -> dict[int, list[int]]:
    edge_index = g.edges() 
    
    return convert_edge_index_to_adj_list(edge_index)


def convert_hg_to_adj_list(hg: dgl.DGLHeteroGraph) -> dict[EdgeType, dict[int, list[int]]]:
    adj_list_dict = dict() 
    
    for etype in hg.canonical_etypes:
        adj_list_dict[etype] = convert_edge_index_to_adj_list(hg.edges(etype=etype))
        
    return adj_list_dict
