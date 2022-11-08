from .imports import * 
from .util import * 
from .metric import * 

from basic_util import *

__all__ = [
    'linear_classify'
]


def linear_classify(*,
                    train_feat: FloatTensor,
                    train_label: IntTensor,
                    val_feat: Optional[FloatTensor] = None,
                    val_label: Optional[IntTensor] = None,
                    test_feat: FloatTensor,
                    test_label: IntTensor,
                    use_gpu: bool = True, 
                    lr: float = 0.001,
                    num_epochs: int = 300,
                    use_tqdm: bool = True) -> dict[str, Any]:
    raise DeprecationWarning
    device = auto_select_gpu(use_gpu=use_gpu)

    train_feat = train_feat.to(device)
    train_label = train_label.to(device)
    if val_feat is not None and val_label is not None:
        val_feat = val_feat.to(device)
        val_label = val_label.to(device)
    test_feat = test_feat.to(device)
    test_label = test_label.to(device)

    feat_dim = train_feat.shape[-1]

    if val_label is not None:
        total_label = torch.concat([train_label, val_label, test_label])
    else:
        total_label = torch.concat([train_label, test_label])
    num_classes = len(total_label.unique())

    model = nn.Linear(feat_dim, num_classes)
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    epoch_to_val_acc: dict[int, float] = dict() 
    epoch_to_test_acc: dict[int, float] = dict()

    for epoch in tqdm(range(1, num_epochs + 1), disable=not use_tqdm, desc='Linear Classify', unit='epoch'):
        model.train() 
        
        pred = model(train_feat)
        
        loss = F.cross_entropy(input=pred, target=train_label)
        
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step() 
        
        model.eval() 
        
        with torch.no_grad():
            val_pred = model(val_feat)
            test_pred = model(test_feat)

        val_acc = calc_acc(pred=val_pred, target=val_label)
        test_acc = calc_acc(pred=test_pred, target=test_label)
        
        epoch_to_val_acc[epoch] = val_acc
        epoch_to_test_acc[epoch] = test_acc

    best_val_acc_epoch, best_val_acc = max(epoch_to_val_acc.items(), key=lambda x: (x[1], -x[0]))
    best_test_acc_epoch, best_test_acc = max(epoch_to_test_acc.items(), key=lambda x: (x[1], -x[0]))

    if best_val_acc_epoch >= num_epochs - 10 \
    or best_test_acc_epoch >= num_epochs - 10:
        log_warning(f"Linear模型尚未完全收敛！")

    # 释放显存
    del train_feat
    del train_label
    del val_feat
    del val_label
    del test_feat
    del test_label
    del model 
        
    return dict(
        best_val_acc = best_val_acc,
        best_val_acc_epoch = best_val_acc_epoch,
        best_test_acc = best_test_acc,
        best_test_acc_epoch = best_test_acc_epoch,
    )
