from .imports import * 
from .adjacency import * 

__all__ = [
    'walk_along_metapath',
    'sample_metapath_subgraph', 
]


def walk_along_metapath(adj_list_dict: dict[EdgeType, dict[int, list[int]]],
                        metapath: list[str]) -> list[list[int]]:
    """
    对异构图中的结点，沿着元路径行走，返回所有轨迹。
    """
    
    etypes = set(adj_list_dict.keys())
    etype_map = { etype[1]: etype for etype in etypes }

    src_traces = [] 
    dest_traces = [] 
    
    for origin_nid in adj_list_dict[etype_map[metapath[0]]].keys():
        src_traces.append([origin_nid])
        
    for etype in metapath:
        etype = etype_map[etype]
        adj_list = adj_list_dict[etype] 
        
        for trace in src_traces:
            last_nid = trace[-1]
            
            for next_nid in adj_list[last_nid]:
                new_trace = trace.copy() 
                new_trace.append(next_nid)
                dest_traces.append(new_trace)
                
        src_traces.clear() 
        src_traces, dest_traces = dest_traces, src_traces 
        
    return src_traces


def sample_metapath_subgraph(adj_list_dict: dict[EdgeType, dict[int, list[int]]],
                             metapath: list[str],
                             is_bipartite: bool = False) -> dgl.DGLGraph:
    traces = walk_along_metapath(
        adj_list_dict = adj_list_dict,
        metapath = metapath, 
    )
    
    subgraph_adj_list: dict[int, set[int]] = defaultdict(set) 
    
    for trace in traces:
        src_nid = trace[0]    
        dest_nid = trace[-1]  
        
        subgraph_adj_list[src_nid].add(dest_nid)
    
    src_edge_index = []    
    dest_edge_index = []
    
    for src_nid in subgraph_adj_list:
        for dest_nid in subgraph_adj_list[src_nid]:
            src_edge_index.append(src_nid)
            dest_edge_index.append(dest_nid)
        
    if not is_bipartite:
        g = dgl.graph((src_edge_index, dest_edge_index))
    else:
        raise NotImplementedError
    
    return g 
