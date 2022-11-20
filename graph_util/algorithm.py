from .imports import * 

__all__ = [
    'calc_PageRank', 
    'get_k_hop_neighbor_nodes', 
]


def calc_PageRank(g: dgl.DGLGraph,
                  alpha: float = 0.85,
                  personalization = None) -> FloatArray:
    num_nodes = g.num_nodes() 
    net = dgl.to_networkx(g)
    
    pr = nx.pagerank(net, alpha=alpha, personalization=personalization)
    
    pr_arr = np.full(fill_value=-1., shape=[num_nodes])
    
    for nid, val in pr.items():
        pr_arr[nid] = val 
        
    assert np.all(pr_arr >= 0.)

    return pr_arr 


def get_k_hop_neighbor_nodes(*, 
                             seed_nids: Iterable[int],
                             hop: int = 1,
                             adj: dict[int, list[int]]) -> set[int]:
    assert hop >= 1 
                    
    visited_nids = set(seed_nids)          
    src_nids = set(seed_nids) 
    dest_nids = set() 
    
    for h in range(hop):
        dest_nids.clear() 
        
        for src_nid in src_nids:
            if src_nid in adj:
                for dest_nid in adj[src_nid]:
                    if dest_nid not in visited_nids:
                        visited_nids.add(dest_nid)
                        dest_nids.add(dest_nid)
                        
        src_nids, dest_nids = dest_nids, src_nids

    return visited_nids 
