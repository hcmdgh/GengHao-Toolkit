from .imports import * 
from .dgl_util import * 

from basic_util import * 

__all__ = [
    'HeteroGraphSampler', 
]


class HeteroGraphSampler:
    def __init__(self, 
                 hg: dgl.DGLHeteroGraph,
                 use_tqdm: bool = True):
        self.hg = hg.cpu() 
        self.edge_index_dict = get_edge_index(hg, return_numpy=True)

        # [BEGIN] 构建邻接表
        digest = hash_graph(hg)
        os.makedirs('./cache/adj_list_table', exist_ok=True)
        cache_path = f'./cache/adj_list_table/{digest}.pkl'
        
        if is_file_exist(cache_path):
            self.adj_list_table = pickle_load(cache_path)
        else:
            self.adj_list_table: dict[EdgeType, dict[int, list[int]]] = dict()
            
            for etype, (src_edge_index, dest_edge_index) in tqdm(self.edge_index_dict.items(), desc='构建异构图邻接表', disable=not use_tqdm):
                self.adj_list_table[etype] = defaultdict(list)
                
                for src_nid, dest_nid in zip(src_edge_index, tqdm(dest_edge_index, disable=not use_tqdm)):
                    src_nid, dest_nid = int(src_nid), int(dest_nid)
                    
                    self.adj_list_table[etype][src_nid].append(dest_nid)

            pickle_dump(self.adj_list_table, cache_path)
        # [END]
        
        
    def sample_relation_neighbors(self,
                                  etype: EdgeType,
                                  src_nids: Union[IntArray, set[int]]) -> tuple[list[int], list[int]]:
        adj_list = self.adj_list_table[etype]
        edge_index = ([], [])
        
        if isinstance(src_nids, IntArray):
            src_nids = set(src_nids.tolist())
        
        for src_nid in src_nids:
            for dest_nid in adj_list[src_nid]:
                edge_index[0].append(src_nid)
                edge_index[1].append(dest_nid)

        return edge_index
        
        
    def sample_relation_subgraph(self,
                                 ntype: str,
                                 nids: Union[IntArray, set[int]]) -> tuple[list[int], list[int]]:
        if isinstance(nids, IntArray):
            nids = set(nids.tolist())

        edge_index = ([], [])
        
        for etype, adj_list in self.adj_list_table.items():
            src_ntype, _, dest_ntype = etype 
            
            if src_ntype == dest_ntype == ntype:
                for src_nid in nids:
                    for dest_nid in adj_list[src_nid]:
                        if dest_nid in nids:
                            edge_index[0].append(src_nid)
                            edge_index[1].append(dest_nid)
                            
        return edge_index 

        
    def sample_subgraph_ogbn_mag(self,
                                 paper_nids: IntArray,
                                 copy_ndata: bool = True) -> dgl.DGLHeteroGraph:
        paper_paper_edge_index = self.sample_relation_subgraph(ntype='paper', nids=paper_nids)
        paper_author_edge_index = self.sample_relation_neighbors(etype=('paper', 'pa', 'author'), src_nids=paper_nids)
        paper_field_edge_index = self.sample_relation_neighbors(etype=('paper', 'pf', 'field'), src_nids=paper_nids)
        paper_nid_set = set(paper_nids.tolist())
        author_nid_set = set(paper_author_edge_index[1])
        field_nid_set = set(paper_field_edge_index[1])
        author_institution_edge_index = self.sample_relation_neighbors(etype=('author', 'ai', 'institution'), src_nids=author_nid_set)
        institution_nid_set = set(author_institution_edge_index[1])

        paper_reindex_table = {nid: i for i, nid in enumerate(paper_nid_set)}
        author_reindex_table = {nid: i for i, nid in enumerate(author_nid_set)}
        field_reindex_table = {nid: i for i, nid in enumerate(field_nid_set)}
        institution_reindex_table = {nid: i for i, nid in enumerate(institution_nid_set)}
        
        reindexed_paper_paper_edge_index = (
            [paper_reindex_table[nid] for nid in paper_paper_edge_index[0]],
            [paper_reindex_table[nid] for nid in paper_paper_edge_index[1]],
        )
        reindexed_paper_author_edge_index = (
            [paper_reindex_table[nid] for nid in paper_author_edge_index[0]],
            [author_reindex_table[nid] for nid in paper_author_edge_index[1]],
        )
        reindexed_paper_field_edge_index = (
            [paper_reindex_table[nid] for nid in paper_field_edge_index[0]],
            [field_reindex_table[nid] for nid in paper_field_edge_index[1]],
        )
        reindexed_author_institution_edge_index = (
            [author_reindex_table[nid] for nid in author_institution_edge_index[0]],
            [institution_reindex_table[nid] for nid in author_institution_edge_index[1]],
        )
        
        hg = dgl.heterograph({
            ('author', 'ai', 'institution'): reindexed_author_institution_edge_index, 
            ('author', 'ap', 'paper'): reindexed_paper_author_edge_index[::-1], 
            ('field', 'fp', 'paper'): reindexed_paper_field_edge_index[::-1], 
            ('institution', 'ia', 'author'): reindexed_author_institution_edge_index[::-1], 
            ('paper', 'pa', 'author'): reindexed_paper_author_edge_index, 
            ('paper', 'pf', 'field'): reindexed_paper_field_edge_index, 
            ('paper', 'pp', 'paper'): reindexed_paper_paper_edge_index, 
        })
        
        # [BEGIN] 子图结点下标对应的原始下标
        hg.nodes['paper'].data['raw_nid'] = torch.tensor(list(paper_nid_set), dtype=torch.int64)
        hg.nodes['author'].data['raw_nid'] = torch.tensor(list(author_nid_set), dtype=torch.int64)
        hg.nodes['field'].data['raw_nid'] = torch.tensor(list(field_nid_set), dtype=torch.int64)
        hg.nodes['institution'].data['raw_nid'] = torch.tensor(list(institution_nid_set), dtype=torch.int64)
        # [END]
        
        if copy_ndata:
            for attr_name in self.hg.ndata:
                for ntype in self.hg.ndata[attr_name]:
                    attr_val = self.hg.ndata[attr_name][ntype]
                    mask = hg.nodes[ntype].data['raw_nid']
                    hg.nodes[ntype].data[attr_name] = attr_val[mask]
        
        return hg 
