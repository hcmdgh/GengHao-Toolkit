# from .imports import * 
# from torch_util import * 

# __all__ = [
#     'L1_normalize', 
#     'generate_hg_feat', 
# ]


# def L1_normalize(inp: FloatTensor) -> FloatTensor:
#     out = F.normalize(inp, p=1, dim=-1)
    
#     return out 


# def generate_hg_feat(hg: dgl.DGLHeteroGraph,
#                      method: str,
#                      dim: int):
#     device = hg.device 
    
#     for ntype in hg.ntypes:
#         try:
#             hg.nodes[ntype].data['feat']
#         except KeyError:
#             if method == 'randn':
#                 feat = torch.randn(hg.num_nodes(ntype), dim, device=device)
#             elif method == 'onehot':
#                 feat = torch.eye(hg.num_nodes(ntype), device=device)
#             else:
#                 raise AssertionError
            
#             hg.nodes[ntype].data['feat'] = feat 
