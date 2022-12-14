from .imports import * 
from .metric import * 
from .recorder import *
from .util import * 
 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

__all__ = [
    'mlp_multiclass_classification',
]


def mlp_multiclass_classification(*,
                                  feat: Union[FloatArray, FloatTensor],
                                  label: Union[IntArray, IntTensor],
                                  num_layers: Literal[1, 2] = 2, 
                                  train_mask: BoolArrayTensor,    
                                  val_mask: BoolArrayTensor,    
                                  test_mask: BoolArrayTensor,
                                  lr: float = 0.001,
                                  num_epochs: int = 300,
                                  use_tqdm: bool = True) -> tuple[float, float]:
    device = get_device()

    feat = to_FloatTensor(feat).to(device)
    label = to_IntTensor(label).to(device)
    train_mask = to_BoolArray(train_mask)
    val_mask = to_BoolArray(val_mask)
    test_mask = to_BoolArray(test_mask)

    N, feat_dim = feat.shape 
    assert label.shape == train_mask.shape == val_mask.shape == test_mask.shape == (N,)

    num_classes = int(torch.max(label)) + 1 
    assert int(torch.min(label)) == 0

    if num_layers == 1:
        model = nn.Linear(feat_dim, num_classes)
    elif num_layers == 2:
        hidden_dim = (feat_dim + num_classes) // 2 

        model = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes), 
        )
    else:
        raise AssertionError 
    
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)

    recorder = ClassificationRecorder(log=False, wandb_log=False)
    
    for epoch in tqdm(range(1, num_epochs + 1), disable=not use_tqdm, desc='mlp_multiclass_classification', unit='epoch'):
        model.train() 
        
        pred = model(feat[train_mask])
        
        loss = F.cross_entropy(input=pred, target=label[train_mask])
        
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step() 
        
        model.eval() 
        
        with torch.no_grad():
            val_pred = model(feat[val_mask])
            test_pred = model(feat[test_mask])

        val_acc = calc_acc(pred=val_pred, target=label[val_mask])
        test_acc = calc_acc(pred=test_pred, target=label[test_mask])
        
        recorder.eval(epoch=epoch, val_acc=val_acc, test_acc=test_acc)    

    best_val_acc, best_val_acc_epoch, test_acc = recorder.summary()
        
    return best_val_acc, test_acc


def mlp_multilabel_classification(
    feat: FloatArray,
    label: BoolArray,
    train_mask: BoolArray,    
    val_mask: BoolArray,    
    test_mask: Optional[BoolArray] = None,
    use_gpu: bool = True,
    lr: float = 0.001,
    early_stopping_epochs: int = 50,
    use_tqdm: bool = True,
) -> dict[str, Any]:
    if use_gpu:
        device = get_device()
    else:
        device = torch.device('cpu')

    label = label.astype(np.int64)
    feat_th = torch.from_numpy(feat).to(torch.float32).to(device)
    label_th = torch.from_numpy(label).to(torch.float32).to(device)
    
    in_dim = feat.shape[-1]
    out_dim = label.shape[-1]
    hidden_dim = (in_dim + out_dim) // 2 
        
    model = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim), 
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_dict = {
        'epoch': 0,
        'val_f1_micro': 0.,
        'val_f1_macro': 0.,
        'test_f1_micro': 0.,
        'test_f1_macro': 0.,
    }
    
    for epoch in tqdm(itertools.count(1), disable=not use_tqdm, desc='mlp_multilabel_classification', unit='epoch'):
        model.train() 
        
        logits = model(feat_th[train_mask])
        
        loss = F.binary_cross_entropy_with_logits(input=logits, target=label_th[train_mask])
        
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step() 
        
        
        model.eval() 
        
        with torch.no_grad():
            logits = model(feat_th[val_mask])

        y_pred = (logits.detach().cpu().numpy() > 0).astype(np.int64)
        y_true = label[val_mask]
        val_f1_micro = calc_f1_micro(y_pred=y_pred, y_true=y_true)
        val_f1_macro = calc_f1_macro(y_pred=y_pred, y_true=y_true)
        
        if test_mask is not None:
            with torch.no_grad():
                logits = model(feat_th[test_mask])

            y_pred = (logits.detach().cpu().numpy() > 0).astype(np.int64)
            y_true = label[test_mask]
            test_f1_micro = calc_f1_micro(y_pred=y_pred, y_true=y_true)
            test_f1_macro = calc_f1_macro(y_pred=y_pred, y_true=y_true)
        else:
            test_f1_micro = test_f1_macro = 0.

        if val_f1_micro > best_dict['val_f1_micro']:
            best_dict['epoch'] = epoch 
            best_dict['val_f1_micro'] = val_f1_micro
            best_dict['val_f1_macro'] = val_f1_macro 
            best_dict['test_f1_micro'] = test_f1_micro 
            best_dict['test_f1_macro'] = test_f1_macro 

        if epoch - best_dict['epoch'] > early_stopping_epochs:
            break 
        
    return best_dict 


def KNeighbors_multiclass_classification(
    feat: FloatArray,
    label: IntArray,
    train_mask: BoolArray,    
    val_mask: BoolArray,    
    test_mask: Optional[BoolArray] = None,
    num_neighbors: int = 1,
) -> tuple[float, float, Optional[float], Optional[float]]:
    assert np.min(label) == 0 

    if test_mask is None:
        assert len(feat) == len(label) == len(train_mask) == len(val_mask)
        assert np.all(train_mask | val_mask)
        assert np.all(~(train_mask & val_mask))
    else:
        assert len(feat) == len(label) == len(train_mask) == len(val_mask) == len(test_mask)
        assert np.all(train_mask | val_mask | test_mask)
        assert np.all(~(train_mask & val_mask & test_mask))
    
    feat_dim = feat.shape[-1]
    num_classes = np.max(label) + 1 
    
    clf = KNeighborsClassifier(n_neighbors=num_neighbors)
    clf.fit(feat[train_mask], label[train_mask])

    val_y_pred = clf.predict(feat[val_mask])
    if test_mask is not None:
        test_y_pred = clf.predict(feat[test_mask])
    
    val_f1_micro = calc_f1_micro(y_pred=val_y_pred, y_true=label[val_mask])
    val_f1_macro = calc_f1_macro(y_pred=val_y_pred, y_true=label[val_mask])
    if test_mask is not None:
        test_f1_micro = calc_f1_micro(y_pred=test_y_pred, y_true=label[test_mask])
        test_f1_macro = calc_f1_macro(y_pred=test_y_pred, y_true=label[test_mask])
    
    if test_mask is None:
        return val_f1_micro, val_f1_macro, None, None 
    else:
        return val_f1_micro, val_f1_macro, test_f1_micro, test_f1_macro 


def xgb_multiclass_classification(
    feat: Union[FloatArray, FloatTensor],
    label: Union[IntArray, IntTensor],
    train_mask: Union[BoolArray, BoolTensor],    
    val_mask: Union[BoolArray, BoolTensor],    
    test_mask: Optional[Union[BoolArray, BoolTensor]] = None,
    verbose: bool = True, 
    check_mask: bool = False,
) -> dict[str, float]:
    if isinstance(feat, Tensor):
        feat = feat.cpu().numpy() 
    if isinstance(label, Tensor):
        label = label.cpu().numpy() 
    if isinstance(train_mask, Tensor):
        train_mask = train_mask.cpu().numpy() 
    if isinstance(val_mask, Tensor):
        val_mask = val_mask.cpu().numpy() 
    if isinstance(test_mask, Tensor):
        test_mask = test_mask.cpu().numpy() 
        
    assert np.min(label) == 0 

    if test_mask is None:
        assert len(feat) == len(label) == len(train_mask) == len(val_mask)
        
        if check_mask:
            assert np.all(train_mask | val_mask)
            assert np.all(~(train_mask & val_mask))
    else:
        assert len(feat) == len(label) == len(train_mask) == len(val_mask) == len(test_mask)
        
        if check_mask:
            assert np.all(train_mask | val_mask | test_mask)
            assert np.all(~(train_mask & val_mask & test_mask))
    
    feat_dim = feat.shape[-1]
    num_classes = np.max(label) + 1 
    
    xg_train = xgb.DMatrix(feat[train_mask], label=label[train_mask])
    xg_val = xgb.DMatrix(feat[val_mask], label=label[val_mask])
    if test_mask is not None:
        xg_test = xgb.DMatrix(feat[test_mask], label=label[test_mask])
    
    param = {
        'objective': 'multi:softmax',
        'eta': 0.1,
        'max_depth': 6, 
        'nthread': 4, 
        'num_class': num_classes, 
    }

    if test_mask is None:
        watch_list = [(xg_train, 'train'), (xg_val, 'val')]
    else:
        watch_list = [(xg_train, 'train'), (xg_val, 'val'), (xg_test, 'test')]
        
    num_round = 5
    
    if verbose:
        bst = xgb.train(param, xg_train, num_round, watch_list)
    else:
        bst = xgb.train(param, xg_train, num_round)
    
    val_y_pred = bst.predict(xg_val).astype(np.int64)
    if test_mask is not None:
        test_y_pred = bst.predict(xg_test).astype(np.int64)
    
    val_f1_micro = calc_f1_micro(y_pred=val_y_pred, y_true=label[val_mask])
    val_f1_macro = calc_f1_macro(y_pred=val_y_pred, y_true=label[val_mask])
    if test_mask is not None:
        test_f1_micro = calc_f1_micro(y_pred=test_y_pred, y_true=label[test_mask])
        test_f1_macro = calc_f1_macro(y_pred=test_y_pred, y_true=label[test_mask])
    else:
        test_f1_micro = test_f1_macro = 0.
    
    return {
        'val_f1_micro': val_f1_micro,
        'val_f1_macro': val_f1_macro,
        'test_f1_micro': test_f1_micro,
        'test_f1_macro': test_f1_macro,
    }


def sklearn_multiclass_classification(
    feat: Union[FloatArray, FloatTensor],
    label: Union[IntArray, IntTensor],
    train_mask: Union[BoolArray, BoolTensor],    
    val_mask: Union[BoolArray, BoolTensor],    
    test_mask: Optional[Union[BoolArray, BoolTensor]] = None,
    check_mask: bool = False,
    max_epochs: int = 200, 
) -> dict[str, float]:
    if isinstance(feat, Tensor):
        feat = feat.cpu().numpy() 
    if isinstance(label, Tensor):
        label = label.cpu().numpy() 
    if isinstance(train_mask, Tensor):
        train_mask = train_mask.cpu().numpy() 
    if isinstance(val_mask, Tensor):
        val_mask = val_mask.cpu().numpy() 
    if isinstance(test_mask, Tensor):
        test_mask = test_mask.cpu().numpy() 
        
    assert np.min(label) == 0 

    if test_mask is None:
        assert len(feat) == len(label) == len(train_mask) == len(val_mask)
        
        if check_mask:
            assert np.all(train_mask | val_mask)
            assert np.all(~(train_mask & val_mask))
    else:
        assert len(feat) == len(label) == len(train_mask) == len(val_mask) == len(test_mask)
        
        if check_mask:
            assert np.all(train_mask | val_mask | test_mask)
            assert np.all(~(train_mask & val_mask & test_mask))
    
    feat_dim = feat.shape[-1]
    num_classes = np.max(label) + 1 
    
    clf = MLPClassifier(max_iter=max_epochs)

    clf.fit(X=feat[train_mask], y=label[train_mask])    
    
    val_y_pred = clf.predict(feat[val_mask])
    if test_mask is not None:
        test_y_pred = clf.predict(feat[test_mask])
    
    val_f1_micro = calc_f1_micro(y_pred=val_y_pred, y_true=label[val_mask])
    val_f1_macro = calc_f1_macro(y_pred=val_y_pred, y_true=label[val_mask])
    if test_mask is not None:
        test_f1_micro = calc_f1_micro(y_pred=test_y_pred, y_true=label[test_mask])
        test_f1_macro = calc_f1_macro(y_pred=test_y_pred, y_true=label[test_mask])
    else:
        test_f1_micro = test_f1_macro = 0.
    
    return {
        'val_f1_micro': val_f1_micro,
        'val_f1_macro': val_f1_macro,
        'test_f1_micro': test_f1_micro,
        'test_f1_macro': test_f1_macro,
    }
