from .imports import * 
from .dgl_util import * 
from .bean import * 

from basic_util import * 
from torch_geometric.datasets import CitationFull, DBLP, IMDB, AMiner, Yelp, CoraFull, Coauthor, MovieLens, HGBDataset, Planetoid, Reddit, OGB_MAG
from ogb.nodeproppred import PygNodePropPredDataset

__all__ = ['load_graph_dataset']

ROOT_CANDIDATES = [
    '/Dataset',
    '/home/Dataset',
    '~/Dataset'
]

for _root in ROOT_CANDIDATES:
    _root = os.path.expanduser(_root)

    if os.path.exists(_root):
        root = _root 
        break 
else:
    raise RuntimeError('数据集根目录不存在！')


def load_ACM_TransE_dataset() -> dgl.DGLHeteroGraph:
    hg_info = torch.load(os.path.join(root, 'NARS/ACM.pt'))
    
    hg = dgl.heterograph(
        hg_info['edge_index_dict'],
        num_nodes_dict = {
            'paper': hg_info['num_paper_nodes'],
            'author': hg_info['num_author_nodes'],
            'field': hg_info['num_field_nodes'],
        },
    )
    
    hg.nodes['paper'].data['feat'] = hg_info['paper_feat']
    hg.nodes['author'].data['feat'] = hg_info['author_feat']
    hg.nodes['field'].data['feat'] = hg_info['field_feat']
    hg.nodes['paper'].data['label'] = hg_info['paper_label']
    hg.nodes['paper'].data['train_mask'] = hg_info['paper_train_mask']
    hg.nodes['paper'].data['val_mask'] = hg_info['paper_val_mask']
    hg.nodes['paper'].data['test_mask'] = hg_info['paper_test_mask']

    return hg 


def load_HeCo_dataset(name: str) -> dgl.DGLHeteroGraph:
    if name == 'ACM':
        return load_dgl_graph(os.path.join(root, 'HeCo/ACM/ACM.pt'))
    elif name == 'Freebase':
        return load_dgl_graph(os.path.join(root, 'HeCo/Freebase/Freebase.pt'))
    else:
        raise AssertionError 


def load_Planetoid_dataset(name: str) -> dgl.DGLGraph:
    dataset = Planetoid(
        root = os.path.join(root, 'PyG/Planetoid'),
        name = name,
    )
    
    _g = dataset[0]
    feat = _g.x 
    label = _g.y 
    edge_index = tuple(_g.edge_index)
    train_mask = _g.train_mask
    val_mask = _g.val_mask
    test_mask = _g.test_mask
    
    g = dgl.graph(edge_index)
    g.ndata['feat'] = feat 
    g.ndata['label'] = label
    g.ndata['train_mask'] = train_mask
    g.ndata['val_mask'] = val_mask
    g.ndata['test_mask'] = test_mask
    
    return g 


def load_Reddit_dataset() -> dgl.DGLGraph:
    dataset = Reddit(
        root = os.path.join(root, 'PyG/Reddit'),
    )
    
    _g = dataset[0]
    feat = _g.x 
    label = _g.y 
    edge_index = tuple(_g.edge_index)
    train_mask = _g.train_mask
    val_mask = _g.val_mask
    test_mask = _g.test_mask
    
    g = dgl.graph(edge_index)
    g.ndata['feat'] = feat 
    g.ndata['label'] = label
    g.ndata['train_mask'] = train_mask
    g.ndata['val_mask'] = val_mask
    g.ndata['test_mask'] = test_mask
    
    return g 


def load_ACM_dataset() -> dgl.DGLHeteroGraph:
    url = "https://data.dgl.ai/dataset/ACM.mat"
    mat_path = os.path.join(root, "DGL/ACM/ACM.mat")
    
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(f"Please download the file manually!\nFrom: {url}\nTo: {mat_path}")
    
    data = sio.loadmat(mat_path)
    p_vs_l = data['PvsL']       # paper-field?
    p_vs_a = data['PvsA']       # paper-author
    p_vs_t = data['PvsT']       # paper-term, bag of words
    p_vs_c = data['PvsC']       # paper-conference, labels come from that

    # We assign
    # (1) KDD papers as class 0 (data mining),
    # (2) SIGMOD and VLDB papers as class 1 (database),
    # (3) SIGCOMM and MOBICOMM papers as class 2 (communication)
    conf_ids = [0, 1, 9, 10, 13]
    label_ids = [0, 1, 2, 2, 1]

    p_vs_c_filter = p_vs_c[:, conf_ids]
    p_selected = (p_vs_c_filter.sum(1) != 0).A1.nonzero()[0]
    p_vs_l = p_vs_l[p_selected]
    p_vs_a = p_vs_a[p_selected]
    p_vs_t = p_vs_t[p_selected]
    p_vs_c = p_vs_c[p_selected]

    hg = dgl.heterograph({
        ('paper', 'pa', 'author'): p_vs_a.nonzero(),
        ('author', 'ap', 'paper'): p_vs_a.transpose().nonzero(),
        ('paper', 'pf', 'field'): p_vs_l.nonzero(),
        ('field', 'fp', 'paper'): p_vs_l.transpose().nonzero()
    })

    feat = torch.FloatTensor(p_vs_t.toarray())

    pc_p, pc_c = p_vs_c.nonzero()
    label = np.zeros(len(p_selected), dtype=np.int64)
    for conf_id, label_id in zip(conf_ids, label_ids):
        label[pc_p[pc_c == conf_id]] = label_id
    label = torch.LongTensor(label)

    num_classes = 3

    float_mask = np.zeros(len(pc_p))
    for conf_id in conf_ids:
        pc_c_mask = (pc_c == conf_id)
        float_mask[pc_c_mask] = np.random.permutation(np.linspace(0, 1, pc_c_mask.sum()))
    train_idx = np.where(float_mask <= 0.2)[0]
    val_idx = np.where((float_mask > 0.2) & (float_mask <= 0.3))[0]
    test_idx = np.where(float_mask > 0.3)[0]

    num_nodes = hg.num_nodes('paper')
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True 
    val_mask[val_idx] = True 
    test_mask[test_idx] = True 
    
    hg.nodes['paper'].data['feat'] = feat 
    hg.nodes['paper'].data['label'] = label 
    hg.nodes['paper'].data['train_mask'] = train_mask 
    hg.nodes['paper'].data['val_mask'] = val_mask 
    hg.nodes['paper'].data['test_mask'] = test_mask 

    return hg 


def load_HGB_dataset(name: str) -> dgl.DGLHeteroGraph:
    dataset = HGBDataset(
        root = os.path.join(root, 'PyG/HGB'),
        name = name, 
    )
    _hg = dataset[0]

    if name == 'ACM':
        hg = dgl.heterograph({
            ('paper', 'pp', 'paper'): tuple(_hg.edge_index_dict[('paper', 'cite', 'paper')]),
            ('paper', 'pp_rev', 'paper'): tuple(_hg.edge_index_dict[('paper', 'ref', 'paper')]),
            ('paper', 'pa', 'author'): tuple(_hg.edge_index_dict[('paper', 'to', 'author')]),
            ('author', 'ap', 'paper'): tuple(_hg.edge_index_dict[('author', 'to', 'paper')]),
            ('paper', 'ps', 'subject'): tuple(_hg.edge_index_dict[('paper', 'to', 'subject')]),
            ('subject', 'sp', 'paper'): tuple(_hg.edge_index_dict[('subject', 'to', 'paper')]),
            ('paper', 'pt', 'term'): tuple(_hg.edge_index_dict[('paper', 'to', 'term')]),
            ('term', 'tp', 'paper'): tuple(_hg.edge_index_dict[('term', 'to', 'paper')]),
        })
        
        hg.nodes['paper'].data['feat'] = _hg['paper'].x 
        hg.nodes['paper'].data['label'] = _hg['paper'].y
        hg.nodes['paper'].data['label_mask'] = _hg['paper'].train_mask
        assert (hg.nodes['paper'].data['label_mask'] == (hg.nodes['paper'].data['label'] >= 0)).all() 
        hg.nodes['author'].data['feat'] = _hg['author'].x
        hg.nodes['subject'].data['feat'] = _hg['subject'].x

    elif name == 'DBLP':
        hg = dgl.heterograph({
            ('author', 'ap', 'paper'): tuple(_hg.edge_index_dict[('author', 'to', 'paper')]),
            ('paper', 'pt', 'term'): tuple(_hg.edge_index_dict[('paper', 'to', 'term')]),
            ('paper', 'pv', 'venue'): tuple(_hg.edge_index_dict[('paper', 'to', 'venue')]),
            ('paper', 'pa', 'author'): tuple(_hg.edge_index_dict[('paper', 'to', 'author')]),
            ('term', 'tp', 'paper'): tuple(_hg.edge_index_dict[('term', 'to', 'paper')]),
            ('venue', 'vp', 'paper'): tuple(_hg.edge_index_dict[('venue', 'to', 'paper')]),
        })
        
        hg.nodes['author'].data['feat'] = _hg['author'].x 
        hg.nodes['author'].data['label'] = _hg['author'].y
        hg.nodes['author'].data['label_mask'] = _hg['author'].train_mask
        assert (hg.nodes['author'].data['label_mask'] == (hg.nodes['author'].data['label'] >= 0)).all() 
        hg.nodes['paper'].data['feat'] = _hg['paper'].x
        hg.nodes['term'].data['feat'] = _hg['term'].x
        
    elif name == 'IMDB':
        hg = dgl.heterograph({
            ('movie', 'md', 'director'): tuple(_hg.edge_index_dict[('movie', 'to', 'director')]),
            ('director', 'dm', 'movie'): tuple(_hg.edge_index_dict[('director', 'to', 'movie')]),
            ('movie', 'ma', 'actor'): tuple(_hg.edge_index_dict[('movie', '>actorh', 'actor')]),
            ('actor', 'am', 'movie'): tuple(_hg.edge_index_dict[('actor', 'to', 'movie')]),
            ('movie', 'mk', 'keyword'): tuple(_hg.edge_index_dict[('movie', 'to', 'keyword')]),
            ('keyword', 'km', 'movie'): tuple(_hg.edge_index_dict[('keyword', 'to', 'movie')]),
        })
        
        hg.nodes['movie'].data['feat'] = _hg['movie'].x 
        hg.nodes['movie'].data['label'] = _hg['movie'].y
        hg.nodes['movie'].data['label_mask'] = _hg['movie'].train_mask
        assert hg.nodes['movie'].data['label'][~ hg.nodes['movie'].data['label_mask']].sum() == 0.  
        hg.nodes['director'].data['feat'] = _hg['director'].x
        hg.nodes['actor'].data['feat'] = _hg['actor'].x
        
    elif name == 'Freebase':
        print(_hg)
        raise NotImplementedError 
    
    return hg 


def load_ogbn_mag_with_TransE() -> dgl.DGLHeteroGraph:
    hg = load_OGB_dataset('ogbn-mag')
    
    hg_pyg = OGB_MAG(
        root = os.path.join(root, 'PyG/ogbn-mag'), 
        preprocess = 'TransE',
    ).data 
    
    hg.nodes['paper'].data['feat'] = hg_pyg['paper'].x
    hg.nodes['author'].data['feat'] = hg_pyg['author'].x
    hg.nodes['field'].data['feat'] = hg_pyg['field_of_study'].x
    hg.nodes['institution'].data['feat'] = hg_pyg['institution'].x

    return hg 


def load_ogbn_mag_with_Metapath2Vec() -> dgl.DGLHeteroGraph:
    hg = load_OGB_dataset('ogbn-mag')
    
    hg_pyg = OGB_MAG(
        root = os.path.join(root, 'PyG/ogbn-mag'), 
        preprocess = 'metapath2vec',
    ).data 
    
    hg.nodes['paper'].data['feat'] = hg_pyg['paper'].x
    hg.nodes['author'].data['feat'] = hg_pyg['author'].x
    hg.nodes['field'].data['feat'] = hg_pyg['field_of_study'].x
    hg.nodes['institution'].data['feat'] = hg_pyg['institution'].x

    return hg 


def load_OGB_dataset(name: str) -> dgl.DGLGraph:
    if name == 'ogbn-mag':
        dataset = PygNodePropPredDataset(name='ogbn-mag', root=os.path.join(root, 'OGB/ogbn-mag')) 

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"]['paper'], split_idx["valid"]['paper'], split_idx["test"]['paper']

        _hg = dataset[0]
        paper_feat = _hg.x_dict['paper']
        paper_year = _hg.node_year['paper'].squeeze(dim=-1)
        paper_label = _hg.y_dict['paper'].squeeze(dim=-1)
        paper_num_nodes = len(paper_feat)
        train_mask = torch.zeros(paper_num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(paper_num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(paper_num_nodes, dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask[valid_idx] = True
        test_mask[test_idx] = True
        edge_index_dict = {
            ('author', 'ai', 'institution'): tuple(_hg.edge_index_dict[('author', 'affiliated_with', 'institution')]),
            ('author', 'ap', 'paper'): tuple(_hg.edge_index_dict[('author', 'writes', 'paper')]),
            ('paper', 'pp', 'paper'): tuple(_hg.edge_index_dict[('paper', 'cites', 'paper')]),
            ('paper', 'pf', 'field'): tuple(_hg.edge_index_dict[('paper', 'has_topic', 'field_of_study')]),
        }
        
        # 增加反向边
        edge_index_dict.update({
            ('institution', 'ia', 'author'): edge_index_dict[('author', 'ai', 'institution')][::-1],
            ('paper', 'pa', 'author'): edge_index_dict[('author', 'ap', 'paper')][::-1],
            ('paper', 'pp_rev', 'paper'): edge_index_dict[('paper', 'pp', 'paper')][::-1],
            ('field', 'fp', 'paper'): edge_index_dict[('paper', 'pf', 'field')][::-1],
        })
        
        hg = dgl.heterograph(edge_index_dict)
        hg.nodes['paper'].data['feat'] = paper_feat
        hg.nodes['paper'].data['year'] = paper_year
        hg.nodes['paper'].data['label'] = paper_label 
        hg.nodes['paper'].data['train_mask'] = train_mask 
        hg.nodes['paper'].data['val_mask'] = val_mask 
        hg.nodes['paper'].data['test_mask'] = test_mask 
        
        return hg 
    
    elif name == 'ogbn-arxiv':
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=os.path.join(root, 'OGB/ogbn-arxiv')) 

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

        _g = dataset[0]
        num_nodes = _g.num_nodes 
        feat = _g.x
        year = _g.node_year.squeeze(dim=-1)
        label = _g.y.squeeze(dim=-1)
        edge_index = tuple(_g.edge_index)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask[valid_idx] = True
        test_mask[test_idx] = True
        
        g = dgl.graph(edge_index)
        g.ndata['feat'] = feat
        g.ndata['year'] = year
        g.ndata['label'] = label
        g.ndata['train_mask'] = train_mask
        g.ndata['val_mask'] = val_mask
        g.ndata['test_mask'] = test_mask

        return g 
    
    elif name == 'ogbn-products':
        dataset = PygNodePropPredDataset(name='ogbn-products', root=os.path.join(root, 'OGB/ogbn-products')) 

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

        _g = dataset[0]
        num_nodes = _g.num_nodes 
        feat = _g.x
        label = _g.y.squeeze(dim=-1)
        edge_index = tuple(_g.edge_index)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask[valid_idx] = True
        test_mask[test_idx] = True
        
        g = dgl.graph(edge_index)
        g.ndata['feat'] = feat
        g.ndata['label'] = label
        g.ndata['train_mask'] = train_mask
        g.ndata['val_mask'] = val_mask
        g.ndata['test_mask'] = test_mask

        return g 
    
    else:
        raise AssertionError


def load_MovieLens_dataset() -> dgl.DGLHeteroGraph:
    raise NotImplementedError


def load_Coauthor_dataset(name: str) -> dgl.DGLGraph:
    dataset = Coauthor(
        root = os.path.join(root, 'PyG/Coauthor'),
        name = name,
    )
    
    _g = dataset[0]
    feat = _g.x 
    label = _g.y 
    edge_index = tuple(_g.edge_index)
    
    g = dgl.graph(edge_index)
    g.ndata['feat'] = feat 
    g.ndata['label'] = label
    
    return g 


def load_CoraFull_dataset() -> dgl.DGLGraph:
    dataset = CoraFull(
        root = os.path.join(root, 'PyG/CoraFull'),
    )
    
    _g = dataset[0]
    feat = _g.x 
    label = _g.y 
    edge_index = tuple(_g.edge_index)
    
    g = dgl.graph(edge_index)
    g.ndata['feat'] = feat 
    g.ndata['label'] = label
    
    return g 


def load_Yelp_dataset() -> dgl.DGLGraph:
    dataset = Yelp(
        root = os.path.join(root, 'PyG/Yelp'),
    )
    
    _g = dataset[0]
    feat = _g.x 
    label = _g.y 
    edge_index = tuple(_g.edge_index)
    train_mask = _g.train_mask
    val_mask = _g.val_mask
    test_mask = _g.test_mask
    
    g = dgl.graph(edge_index)
    g.ndata['feat'] = feat 
    g.ndata['label'] = label
    g.ndata['train_mask'] = train_mask 
    g.ndata['val_mask'] = val_mask 
    g.ndata['test_mask'] = test_mask 
    
    return g 


def load_AMiner_dataset() -> dgl.DGLGraph:
    raise NotImplementedError


def load_DBLP_dataset(homo_or_hetero: bool) -> dgl.DGLGraph:
    if homo_or_hetero:
        dataset = CitationFull(
            root = os.path.join(root, 'PyG/DBLP_homo'),
            name = 'DBLP',
        )
        
        _g = dataset[0]
        feat = _g.x 
        label = _g.y 
        edge_index = tuple(_g.edge_index)
        
        g = dgl.graph(edge_index)
        g.ndata['feat'] = feat 
        g.ndata['label'] = label 
        
        return g 
    
    else:
        dataset = DBLP(
            root = os.path.join(root, 'PyG/DBLP_hetero'),
        )

        _hg = dataset[0]
        author_feat = _hg.x_dict['author']
        author_label = _hg.y_dict['author']
        author_train_mask = _hg.train_mask_dict['author']
        author_val_mask = _hg.val_mask_dict['author']
        author_test_mask = _hg.test_mask_dict['author']
        paper_feat = _hg.x_dict['paper']
        term_feat = _hg.x_dict['term']
        edge_index_dict = {
            ('author', 'ap', 'paper'): tuple(_hg.edge_index_dict[('author', 'to', 'paper')]), 
            ('paper', 'pa', 'author'): tuple(_hg.edge_index_dict[('paper', 'to', 'author')]), 
            ('paper', 'pt', 'term'): tuple(_hg.edge_index_dict[('paper', 'to', 'term')]), 
            ('paper', 'pc', 'conference'): tuple(_hg.edge_index_dict[('paper', 'to', 'conference')]), 
            ('term', 'tp', 'paper'): tuple(_hg.edge_index_dict[('term', 'to', 'paper')]), 
            ('conference', 'cp', 'paper'): tuple(_hg.edge_index_dict[('conference', 'to', 'paper')]), 
        }
        
        hg = dgl.heterograph(edge_index_dict)
        hg.nodes['author'].data['feat'] = author_feat
        hg.nodes['author'].data['label'] = author_label
        hg.nodes['author'].data['train_mask'] = author_train_mask
        hg.nodes['author'].data['val_mask'] = author_val_mask
        hg.nodes['author'].data['test_mask'] = author_test_mask
        hg.nodes['paper'].data['feat'] = paper_feat
        hg.nodes['term'].data['feat'] = term_feat

        return hg 
    
    
def load_IMDB_dataset() -> dgl.DGLHeteroGraph:
    dataset = IMDB(root=os.path.join(root, 'PyG/IMDB'))

    _hg = dataset[0] 
    movie_feat = _hg['movie'].x 
    movie_label = _hg['movie'].y
    movie_train_mask = _hg['movie'].train_mask
    movie_val_mask = _hg['movie'].val_mask
    movie_test_mask = _hg['movie'].test_mask
    director_feat = _hg['director'].x 
    actor_feat = _hg['actor'].x 
    
    md_edge_index = tuple(_hg.edge_index_dict[('movie', 'to', 'director')])
    ma_edge_index = tuple(_hg.edge_index_dict[('movie', 'to', 'actor')])
    dm_edge_index = tuple(_hg.edge_index_dict[('director', 'to', 'movie')])
    am_edge_index = tuple(_hg.edge_index_dict[('actor', 'to', 'movie')])
    
    hg = dgl.heterograph({
        ('movie', 'md', 'director'): md_edge_index,
        ('movie', 'ma', 'actor'): ma_edge_index,
        ('director', 'dm', 'movie'): dm_edge_index,
        ('actor', 'am', 'movie'): am_edge_index,
    })
    
    hg.nodes['movie'].data['feat'] = movie_feat
    hg.nodes['movie'].data['label'] = movie_label
    hg.nodes['movie'].data['train_mask'] = movie_train_mask
    hg.nodes['movie'].data['val_mask'] = movie_val_mask
    hg.nodes['movie'].data['test_mask'] = movie_test_mask
    hg.nodes['director'].data['feat'] = director_feat 
    hg.nodes['actor'].data['feat'] = actor_feat 
    
    return hg 


def load_AMiner_dataset() -> dgl.DGLHeteroGraph:
    dataset = AMiner(root=os.path.join(root, 'PyG/AMiner'))

    _hg = dataset[0] 
    
    num_author_nodes = _hg['author'].num_nodes
    num_venue_nodes = _hg['venue'].num_nodes
    num_paper_nodes = _hg['paper'].num_nodes

    _author_label = _hg['author'].y 
    _venue_label = _hg['venue'].y 
    author_label_nids = _hg['author'].y_index 
    venue_label_nids = _hg['venue'].y_index 
    
    author_label = torch.full(size=[num_author_nodes], fill_value=-1, dtype=torch.int64)
    author_label[author_label_nids] = _author_label
    venue_label = torch.full(size=[num_venue_nodes], fill_value=-1, dtype=torch.int64)
    venue_label[venue_label_nids] = _venue_label
    
    pa_edge_index = tuple(_hg.edge_index_dict[('paper', 'written_by', 'author')])
    ap_edge_index = tuple(_hg.edge_index_dict[('author', 'writes', 'paper')])
    pv_edge_index = tuple(_hg.edge_index_dict[('paper', 'published_in', 'venue')])
    vp_edge_index = tuple(_hg.edge_index_dict[('venue', 'publishes', 'paper')])
    
    hg = dgl.heterograph(
        {
            ('paper', 'pa', 'author'): pa_edge_index,
            ('author', 'ap', 'paper'): ap_edge_index,
            ('paper', 'pv', 'venue'): pv_edge_index,
            ('venue', 'vp', 'paper'): vp_edge_index,
        },
        num_nodes_dict = {'author': num_author_nodes, 'venue': num_venue_nodes, 'paper': num_paper_nodes},
    )
    
    hg.nodes['author'].data['label'] = author_label
    hg.nodes['venue'].data['label'] = venue_label
    
    return hg 


def load_graph_dataset(dataset_name: str,
                       format: str = 'dgl') -> Union[dgl.DGLGraph, pygdata.HeteroData]:
    dataset_name = dataset_name.lower().strip() 
    format = format.lower().strip() 
    
    if dataset_name == 'dblp-homo':
        g = load_DBLP_dataset(homo_or_hetero=True)
    elif dataset_name == 'dblp-hetero':
        g = load_DBLP_dataset(homo_or_hetero=False)
    elif dataset_name == 'imdb':
        g = load_IMDB_dataset()
    elif dataset_name == 'yelp':
        g = load_Yelp_dataset()
    elif dataset_name == 'corafull':
        g = load_CoraFull_dataset()
    elif dataset_name == 'coauthor-cs':
        g = load_Coauthor_dataset('CS')
    elif dataset_name == 'coauthor-physics':
        g = load_Coauthor_dataset('Physics')
    elif dataset_name == 'movielens':
        g = load_MovieLens_dataset()
    elif dataset_name == 'ogbn-mag':
        g = load_OGB_dataset('ogbn-mag')
    elif dataset_name == 'ogbn-mag_transe':
        g = load_ogbn_mag_with_TransE()
    elif dataset_name == 'ogbn-mag_metapath2vec':
        g = load_ogbn_mag_with_Metapath2Vec()
    elif dataset_name == 'ogbn-arxiv':
        g = load_OGB_dataset('ogbn-arxiv')
    elif dataset_name == 'ogbn-products':
        g = load_OGB_dataset('ogbn-products')
    elif dataset_name == 'hgb-acm':
        g = load_HGB_dataset('ACM')
    elif dataset_name == 'hgb-dblp':
        g = load_HGB_dataset('DBLP')
    elif dataset_name == 'hgb-freebase':
        g = load_HGB_dataset('Freebase')
    elif dataset_name == 'hgb-imdb':
        g = load_HGB_dataset('IMDB')
    elif dataset_name == 'acm':
        g = load_ACM_dataset()
    elif dataset_name == 'cora':
        g = load_Planetoid_dataset('Cora')
    elif dataset_name == 'citeseer':
        g = load_Planetoid_dataset('CiteSeer')
    elif dataset_name == 'pubmed':
        g = load_Planetoid_dataset('PubMed')
    elif dataset_name == 'reddit':
        g = load_Reddit_dataset()
    elif dataset_name == 'heco-acm':
        g = load_HeCo_dataset('ACM')
    elif dataset_name == 'heco-freebase':
        g = load_HeCo_dataset('Freebase')
    elif dataset_name == 'acm_transe':
        g = load_ACM_TransE_dataset()
    elif dataset_name == 'aminer':
        g = load_AMiner_dataset()
    else:
        raise AssertionError 

    if format == 'dgl':
        return g 
    elif format == 'pyg':
        if g.is_homogeneous:
            return HomoGraph.from_dgl(g).to_pyg() 
        else: 
            return HeteroGraph.from_dgl(g).to_pyg()
    else:
        raise AssertionError 
