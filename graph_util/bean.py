from .imports import * 
from basic_util import * 

__all__ = [
    'HeteroGraph',
    'HomoGraph',
]


@dataclass
class HeteroGraph:
    num_nodes_dict: dict[NodeType, int]
    edge_index_dict: dict[EdgeType, EdgeIndex]
    node_attr_dict: dict[str, dict[NodeType, Tensor]]
    edge_attr_dict: dict[str, dict[EdgeType, Tensor]]
    
    def to_dgl(self) -> dgl.DGLHeteroGraph:
        hg = dgl.heterograph(
            data_dict = self.edge_index_dict,
            num_nodes_dict = self.num_nodes_dict, 
        )
        
        for k1 in self.node_attr_dict:
            for k2, v in self.node_attr_dict[k1].items():
                hg.nodes[k2].data[k1] = v 
                
        for k1 in self.edge_attr_dict:
            for k2, v in self.edge_attr_dict[k1].items():
                hg.edges[k2].data[k1] = v
                
        return hg  
    
    @classmethod
    def from_dgl(cls, 
                 hg: dgl.DGLHeteroGraph) -> 'HeteroGraph':
        num_nodes_dict = {
            node_type: hg.num_nodes(node_type)
            for node_type in hg.ntypes
        }
        
        edge_index_dict = {
            edge_type: hg.edges(etype=edge_type)
            for edge_type in hg.canonical_etypes
        }
        
        return cls(
            num_nodes_dict = num_nodes_dict,
            edge_index_dict = edge_index_dict,
            node_attr_dict = dict(hg.ndata),
            edge_attr_dict = dict(hg.edata),
        )
        
    def to_pyg(self) -> pygdata.HeteroData:
        data = pygdata.HeteroData()
        
        for ntype, num_nodes in self.num_nodes_dict.items():
            data[ntype].num_nodes = num_nodes
        
        for etype, edge_index in self.edge_index_dict.items():
            data[etype].edge_index = torch.stack(edge_index)
        
        for k1 in self.node_attr_dict:
            for k2, v in self.node_attr_dict[k1].items():
                data[k2][k1] = v 
                
        for k1 in self.edge_attr_dict:
            for k2, v in self.edge_attr_dict[k1].items():
                data[k2][k1] = v
                
        return data 

    def save_to_file(self, file_path: str):
        torch_dump(asdict(self), file_path)
        
    @classmethod
    def load_from_file(cls, file_path: str) -> 'HeteroGraph':
        return cls(**torch_load(file_path))
        

@dataclass
class HomoGraph:
    num_nodes: int
    edge_index: EdgeIndex
    node_attr_dict: dict[str, Tensor]
    edge_attr_dict: dict[str, Tensor]

    def to_dgl(self) -> dgl.DGLGraph:
        g = dgl.graph(
            data = self.edge_index,
            num_nodes = self.num_nodes, 
        )
        
        for k, v in self.node_attr_dict.items():
            g.ndata[k] = v 
            
        for k, v in self.edge_attr_dict.items():
            g.edata[k] = v 

        return g 
    
    @classmethod
    def from_dgl(cls, 
                 g: dgl.DGLGraph) -> 'HomoGraph':
        return cls(
            num_nodes = g.num_nodes(), 
            edge_index = tuple(g.edges()),
            node_attr_dict = dict(g.ndata),
            edge_attr_dict = dict(g.edata),
        )
        
    def to_pyg(self) -> pygdata.Data:
        data = pygdata.Data()
        
        data.num_nodes = self.num_nodes
        
        data.edge_index = torch.stack(self.edge_index)
        
        for k, v in self.node_attr_dict.items():
            data[k] = v 
            
        for k, v in self.edge_attr_dict.items():
            data[k] = v 
                
        return data 

    def save_to_file(self, file_path: str):
        torch_dump(asdict(self), file_path)
        
    @classmethod
    def load_from_file(cls, file_path: str) -> 'HomoGraph':
        return cls(**torch_load(file_path))
