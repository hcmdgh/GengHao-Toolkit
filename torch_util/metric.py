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
    target = to_FloatArray(target)

    N, num_classes = pred.shape 
    assert target.shape == (N,)
    assert np.max(target) < num_classes
    
    pred = np.argmax(pred, axis=-1)
                  
    return f1_score(y_true=target, y_pred=pred, average='micro')


def calc_f1_macro(pred: FloatArrayTensor,
                  target: IntArrayTensor) -> float:
    pred = to_FloatArray(pred)
    target = to_FloatArray(target)

    N, num_classes = pred.shape 
    assert target.shape == (N,)
    assert np.max(target) < num_classes
    
    pred = np.argmax(pred, axis=-1)
                  
    return f1_score(y_true=target, y_pred=pred, average='macro')


def calc_acc(pred: FloatArrayTensor,
             target: IntArrayTensor) -> float:
    pred = to_FloatArray(pred)
    target = to_FloatArray(target)

    N, num_classes = pred.shape 
    assert target.shape == (N,)
    assert np.max(target) < num_classes
    
    pred = np.argmax(pred, axis=-1)
                  
    return accuracy_score(y_true=target, y_pred=pred)


def calc_cosine_similarity(h1: FloatTensor, 
                           h2: FloatTensor) -> FloatTensor:
    N, emb_dim = h1.shape 
    assert h2.shape == (N, emb_dim)
                           
    h1 = F.normalize(h1, p=2, dim=-1)
    h2 = F.normalize(h2, p=2, dim=-1)

    out = torch.mm(h1, h2.T)
    
    return out 
