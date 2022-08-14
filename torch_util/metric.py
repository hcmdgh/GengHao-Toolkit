from .imports import * 
from .util import * 

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

__all__ = [
    'calc_f1_micro',
    'calc_f1_macro',
    'calc_acc',
    'calc_cosine_similarity',
]


def calc_f1_micro(pred: FloatArrayTensor,
                  target: IntArrayTensor) -> float:
    pred = to_FloatArray(pred)
    target = to_IntArray(target)

    if target.ndim == 1:
        N, num_classes = pred.shape 
        assert target.shape == (N,)
        assert np.max(target) < num_classes
        
        pred = np.argmax(pred, axis=-1)
                    
        return f1_score(y_true=target, y_pred=pred, average='micro')
    elif target.ndim == 2:
        N, num_classes = pred.shape 
        assert target.shape == (N, num_classes)
        assert np.min(target) == 0 and np.max(target) == 1 

        _pred = np.zeros([N, num_classes], dtype=np.int64)
        _pred[pred > 0.] = 1 
        
        return f1_score(y_true=target, y_pred=_pred, average='micro')
    else:
        raise AssertionError


def calc_f1_macro(pred: FloatArrayTensor,
                  target: IntArrayTensor) -> float:
    pred = to_FloatArray(pred)
    target = to_IntArray(target)

    if target.ndim == 1:
        N, num_classes = pred.shape 
        assert target.shape == (N,)
        assert np.max(target) < num_classes
        
        pred = np.argmax(pred, axis=-1)
                    
        return f1_score(y_true=target, y_pred=pred, average='macro')
    elif target.ndim == 2:
        N, num_classes = pred.shape 
        assert target.shape == (N, num_classes)
        assert np.min(target) == 0 and np.max(target) == 1 

        _pred = np.zeros([N, num_classes], dtype=np.int64)
        _pred[pred > 0.] = 1 
        
        return f1_score(y_true=target, y_pred=_pred, average='macro')
    else:
        raise AssertionError


def calc_acc(pred, target) -> float:
    pred = to_FloatArray(pred)
    target = to_IntArray(target)

    if target.ndim == 1:
        N, num_classes = pred.shape 
        assert target.shape == (N,)
        assert np.max(target) < num_classes
        
        pred = np.argmax(pred, axis=-1)
                    
        return accuracy_score(y_true=target, y_pred=pred)
    elif target.ndim == 2:
        N, num_classes = pred.shape 
        assert target.shape == (N, num_classes)
        assert np.min(target) == 0 and np.max(target) == 1 

        _pred = np.zeros([N, num_classes], dtype=np.int64)
        _pred[pred > 0.] = 1 
        
        return accuracy_score(y_true=target, y_pred=_pred)
    else:
        raise AssertionError


def calc_cosine_similarity(h1: FloatTensor, 
                           h2: FloatTensor) -> FloatTensor:
    N, emb_dim = h1.shape 
    assert h2.shape == (N, emb_dim)
                           
    h1 = F.normalize(h1, p=2, dim=-1)
    h2 = F.normalize(h2, p=2, dim=-1)

    out = torch.mm(h1, h2.T)
    
    return out 
