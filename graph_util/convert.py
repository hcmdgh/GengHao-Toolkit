from .imports import * 

__all__ = [
    'convert_dgl_g_to_dict', 
    'convert_dict_to_dgl_g', 
    'convert_dgl_hg_to_dict', 
    'convert_dict_to_dgl_hg', 
]


def convert_dict_to_dgl_g(d: dict[str, Any],
                          to_bidirected: bool = True,
                          add_self_loop: bool = True) -> dgl.DGLGraph:
    num_nodes = d['num_nodes']
    
    if to_bidirected and add_self_loop:
        edge_index = d['edge_index_selfloop_bidirected']
    elif to_bidirected and not add_self_loop:
        edge_index = d['edge_index_bidirected']
    elif not to_bidirected and add_self_loop:
        edge_index = d['edge_index_selfloop']
    elif not to_bidirected and not add_self_loop:
        edge_index = d['edge_index']
    else:
        raise AssertionError
    
    g = dgl.graph(edge_index, num_nodes=num_nodes)

    for key, val in d['ndata_dict'].items():
        g.ndata[key] = torch.tensor(val)  
        
    for key, val in d['edata_dict'].items():
        g.edata[key] = torch.tensor(val)  
    
    return g 


def convert_dgl_g_to_dict(g: dgl.DGLGraph) -> dict[str, Any]:
    g = dgl.remove_self_loop(g)
    
    num_nodes = g.num_nodes() 
    
    edge_index = g.edges()
    edge_index = (edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy())
    
    ndata_dict = {
        key: val.cpu().numpy() 
        for key, val in g.ndata.items() 
    } 
    
    edata_dict = {
        key: val.cpu().numpy() 
        for key, val in g.edata.items() 
    } 
    
    # [BEGIN] 额外存储增加自环、无向图的edge_index
    g_bidirected = dgl.to_bidirected(g)
    edge_index_bidirected = g_bidirected.edges()
    edge_index_bidirected = (edge_index_bidirected[0].cpu().numpy(), edge_index_bidirected[1].cpu().numpy())
    
    g_selfloop = dgl.add_self_loop(g)
    edge_index_selfloop = g_selfloop.edges()
    edge_index_selfloop = (edge_index_selfloop[0].cpu().numpy(), edge_index_selfloop[1].cpu().numpy())
    
    g_selfloop_bidirected = dgl.add_self_loop(dgl.to_bidirected(g)) 
    edge_index_selfloop_bidirected = g_selfloop_bidirected.edges()
    edge_index_selfloop_bidirected = (edge_index_selfloop_bidirected[0].cpu().numpy(), edge_index_selfloop_bidirected[1].cpu().numpy())
    # [END]

    return dict(
        num_nodes = num_nodes,
        edge_index = edge_index, 
        edge_index_bidirected = edge_index_bidirected,
        edge_index_selfloop = edge_index_selfloop,
        edge_index_selfloop_bidirected = edge_index_selfloop_bidirected,
        ndata_dict = ndata_dict, 
        edata_dict = edata_dict, 
    )


def convert_dict_to_dgl_hg(d: dict[str, Any]) -> dgl.DGLHeteroGraph:
    num_nodes_dict = d['num_nodes_dict']
    edge_index_dict = d['edge_index_dict']
    ndata_dict = d['ndata_dict']
    edata_dict = d['edata_dict']
    
    hg = dgl.heterograph(edge_index_dict, num_nodes_dict=num_nodes_dict)

    for key in ndata_dict:
        for ntype in ndata_dict[key]:
            val = ndata_dict[key][ntype]
            hg.nodes[ntype].data[key] = torch.tensor(val)
            
    for key in edata_dict:
        for etype in edata_dict[key]:
            val = edata_dict[key][etype]
            hg.edges[etype].data[key] = torch.tensor(val)
    
    return hg 


def convert_dgl_hg_to_dict(hg: dgl.DGLHeteroGraph) -> dict[str, Any]:
    ntypes = set(hg.ntypes)
    etypes = set(hg.canonical_etypes)
    
    num_nodes_dict = {
        ntype: hg.num_nodes(ntype)
        for ntype in ntypes 
    } 
    
    edge_index_dict = {} 
    
    for etype in etypes:
        edge_index = hg.edges(etype=etype)
        edge_index = (edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy())

        edge_index_dict[etype] = edge_index
    
    ndata_dict = {} 
    
    for key in hg.ndata:
        ndata_dict[key] = dict() 
        
        for ntype in hg.ndata[key]:
            val = hg.ndata[key][ntype]
            val = val.cpu().numpy()

            ndata_dict[key][ntype] = val  
    
    edata_dict = {} 
    
    for key in hg.edata:
        edata_dict[key] = dict() 
        
        for etype in hg.edata[key]:
            val = hg.edata[key][etype]
            val = val.cpu().numpy()

            edata_dict[key][etype] = val  

    return dict(
        num_nodes_dict = num_nodes_dict,
        edge_index_dict = edge_index_dict, 
        ndata_dict = ndata_dict, 
        edata_dict = edata_dict, 
    )
