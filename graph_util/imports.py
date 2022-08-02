from torch_util.imports import * 

import dgl
import dgl.function as dglfn
import dgl.nn.pytorch as dglnn
import dgl.nn.functional as dglF
import dgl.data.utils as dglutil

import torch_geometric as pyg
import torch_geometric.data as pygdata 
import torch_geometric.nn as pygnn 
import torch_geometric.nn.conv as pygconv 
import torch_geometric.loader as pygloader 
import torch_geometric.utils as pygutil

NodeType = str 
EdgeType = tuple[str, str, str]
EdgeIndex = tuple[IntTensor, IntTensor]
