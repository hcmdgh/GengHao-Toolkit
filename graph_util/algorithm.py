from .imports import * 

__all__ = [
    'calc_PageRank', 
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


def calc_PageRank_dgl(g: dgl.DGLGraph,
                      k: int = 10,
                      damp: float = 0.85) -> FloatTensor:
    raise DeprecationWarning
    num_nodes = g.num_nodes()
    g.ndata['pv'] = torch.ones(num_nodes) / num_nodes
    degrees = g.out_degrees().float() 

    for _ in range(k):
        g.ndata['pv'] = g.ndata['pv'] / degrees
        g.update_all(
            message_func = dglfn.copy_src(src='pv', out='_'),
            reduce_func = dglfn.sum(msg='_', out='pv'),
        )
        g.ndata['pv'] = (1 - damp) / num_nodes + damp * g.ndata['pv']

    pagerank = g.ndata.pop('pv') 
    
    return pagerank
