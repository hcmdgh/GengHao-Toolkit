from .imports import * 
from basic_util import * 

__all__ = [
    'ClassificationRecorder',
]


class ClassificationRecorder:
    def __init__(self,
                 log: bool = True,
                 wandb_log: bool = True):
        self.use_log = log 
        self.use_wandb_log = wandb_log 
        
        self.val_acc_dict: dict[int, float] = dict()
        self.test_acc_dict: dict[int, float] = dict()
        
    def _get_best_val_acc(self) -> tuple[int, float]:
        val_acc_list = sorted(self.val_acc_dict.items(), key=lambda x: (-x[1], x[0]))
        best_val_acc_epoch, best_val_acc = val_acc_list[0]

        return best_val_acc_epoch, best_val_acc

    def train(self,
              epoch: int,
              loss: Any,
              mute: bool = False,
              **other_loss_dict):
        loss = float(loss)
        
        other_loss_dict = { k: float(v) for k, v in other_loss_dict.items() }
        
        if self.use_log and not mute:
            if not other_loss_dict:
                log_info(f"epoch: {epoch}, loss: {loss:.4f}")
            else:
                other_loss_info = ', '.join(f"{k}: {v:.4f}" for k, v in other_loss_dict.items())
                log_info(f"epoch: {epoch}, loss: {loss:.4f}, {other_loss_info}")
            
        all_loss_dict = { 'loss': loss }
        all_loss_dict.update(other_loss_dict)
            
        if self.use_wandb_log:
            wandb.log(
                all_loss_dict, 
                step = epoch,
            )
    
    def eval(self,
             epoch: int, 
             val_acc: float,
             test_acc: float):
        self.val_acc_dict[epoch] = val_acc 
        self.test_acc_dict[epoch] = test_acc 

        best_val_acc_epoch, best_val_acc = self._get_best_val_acc()
        
        if self.use_log:
            log_info(f"epoch: {epoch}, val_acc: {val_acc:.4f} (best: {best_val_acc:.4f} in epoch {best_val_acc_epoch}), test_acc: {test_acc:.4f}")
            
        if self.use_wandb_log:
            wandb.log(
                { 
                    'val_acc': val_acc,
                    'test_acc': test_acc,
                }, 
                step = epoch,
            )

    def summary(self) -> tuple[float, int, float]:
        best_val_acc_epoch, best_val_acc = self._get_best_val_acc()
        test_acc = self.test_acc_dict[best_val_acc_epoch]
        
        if self.use_wandb_log:
            wandb.summary['best_val_acc'] = best_val_acc 
            wandb.summary['best_val_acc_epoch'] = best_val_acc_epoch
            wandb.summary['final_test_acc'] = test_acc  

        return best_val_acc, best_val_acc_epoch, test_acc
