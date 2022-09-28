from .imports import * 

__all__ = [
    'convert_hg_to_triplets', 
]


def convert_hg_to_triplets(hg: dgl.DGLHeteroGraph,
                           output_path: str):
    nid_offset_dict: dict[NodeType, int] = dict()
    
    offset = 0 
    
    for ntype in hg.ntypes:
        num_nodes = hg.num_nodes(ntype)
        nid_offset_dict[ntype] = offset 
        offset += num_nodes 
        
    with open(output_path, 'w', encoding='utf-8') as fp:
        for etype in hg.canonical_etypes:
            _etype = '__'.join(etype)
            src_ntype, _, dest_ntype = etype 
            
            src_edge_index, dest_edge_index = hg.edges(etype=etype)
            src_edge_index, dest_edge_index = src_edge_index.cpu().numpy(), dest_edge_index.cpu().numpy() 
            
            src_edge_index = src_edge_index + nid_offset_dict[src_ntype]
            dest_edge_index = dest_edge_index + nid_offset_dict[dest_ntype]

            for src_nid, dest_nid in zip(src_edge_index, dest_edge_index):
                print(f"{src_nid}\t{_etype}\t{dest_nid}", file=fp)
