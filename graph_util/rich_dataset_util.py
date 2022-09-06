from .imports import * 
from .dataset_util import * 

__all__ = [
    'RichHeteroGraphDataset',
    'load_rich_graph_dataset', 
]


class RichHeteroGraphDataset:
    def __init__(self,
                 device, 
                 hg: dgl.DGLHeteroGraph,
                 label_ntype: NodeType,
                 metapath_list: list[list[str]],
                 num_classes: Optional[int] = None):
        self.device = device 
                 
        self.hg = hg = hg.to(device)
        self.label_ntype = label_ntype 
        self.train_mask = hg.nodes[label_ntype].data['train_mask']
        self.val_mask = hg.nodes[label_ntype].data['val_mask']
        self.test_mask = hg.nodes[label_ntype].data['test_mask']
        self.label = hg.nodes[label_ntype].data['label']
        self.num_classes = num_classes if num_classes else len(self.label.unique())
        self.metapath_list = metapath_list
        
    def sample_metapath_subgraph(self,
                                 hg: Optional[dgl.DGLHeteroGraph] = None) -> list[dgl.DGLGraph]:
        if hg is None:
            hg = self.hg 
            
        subgraph_list = [
            dgl.add_self_loop(
                dgl.remove_self_loop(
                    dgl.metapath_reachable_graph(hg, metapath)
                )
            )
            for metapath in self.metapath_list 
        ]
        
        return subgraph_list
        
        
def load_rich_graph_dataset(dataset_name: str,
                            device) -> RichHeteroGraphDataset:
    dataset_name = dataset_name.lower().strip() 
    
    if dataset_name == 'ogbn-mag_transe':
        return RichHeteroGraphDataset(
            device = device,
            hg = load_graph_dataset('ogbn-mag_transe'),
            label_ntype = 'paper',
            metapath_list = [
                ['pp', 'pp_rev'],
                ['pf', 'fp'],
                ['pa', 'ap'],
            ],
        )
        
    elif dataset_name == 'dblp-hetero':
        hg = load_graph_dataset('dblp-hetero')

        # 生成onehot结点特征
        hg.nodes['conference'].data['feat'] = torch.eye(hg.num_nodes('conference'))
        
        return RichHeteroGraphDataset(
            device = device,
            hg = hg,
            label_ntype = 'author',
            metapath_list = [
                ['ap', 'pa'],
                ['ap', 'pt', 'tp', 'pa'],
                ['ap', 'pc', 'cp', 'pa'],
            ],
        )
        
    elif dataset_name == 'imdb':
        return RichHeteroGraphDataset(
            device = device,
            hg = load_graph_dataset('imdb'),
            label_ntype = 'movie',
            metapath_list = [
                ['md', 'dm'],
                ['ma', 'am'],
            ],
        )
        
    else:
        raise AssertionError 
