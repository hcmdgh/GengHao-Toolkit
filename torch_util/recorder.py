from .imports import * 
from basic_util import * 

__all__ = [
    'ClassificationRecorder',
]


class ClassificationRecorder:
    def __init__(self,
                 multiple_run: bool = False, 
                 log: bool = True,
                 wandb_log: bool = True):
        self.use_log = log 
        self.use_wandb_log = wandb_log 
        self.multiple_run = multiple_run 
        
        self.val_acc_dict: dict[int, float] = dict()
        self.test_acc_dict: dict[int, float] = dict()
        
    def get_best_acc(self) -> dict[str, Any]:
        val_acc_list = sorted(self.val_acc_dict.items(), key=lambda x: (-x[1], x[0]))
        test_acc_list = sorted(self.test_acc_dict.items(), key=lambda x: (-x[1], x[0]))
        best_val_acc_epoch, best_val_acc = val_acc_list[0]
        best_test_acc_epoch, best_test_acc = test_acc_list[0]

        return dict(
            best_val_acc = best_val_acc,
            best_val_acc_epoch = best_val_acc_epoch,
            best_test_acc = best_test_acc,
            best_test_acc_epoch = best_test_acc_epoch, 
        )

    def train(self,
              epoch: int,
              loss: Any,
              mute: bool = False):
        loss = float(loss)
        
        if self.use_log and not mute:
            log_info(f"epoch: {epoch}, loss: {loss:.4f}")
            
        if self.use_wandb_log:
            wandb.log(
                dict(loss=loss), 
                step = epoch,
            )
            
    def eval_step(self,
                  epoch: int,
                  step: int,
                  val_acc: float,
                  test_acc: float):
        raise DeprecationWarning
        if self.use_log:
            log_info(f"epoch: {epoch}, step: {step}, val_acc: {val_acc:.4f}, test_acc: {test_acc:.4f}")
    
    def eval(self,
             epoch: int, 
             val_acc: float,
             test_acc: float):
        self.val_acc_dict[epoch] = val_acc 
        self.test_acc_dict[epoch] = test_acc 

        best_acc_info = self.get_best_acc()
        
        if self.use_log:
            log_info(f"Epoch: {epoch}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")
            log_info(f"Best Val Acc: {best_acc_info['best_val_acc']:.4f} in Epoch {best_acc_info['best_val_acc_epoch']}, Best Test Acc: {best_acc_info['best_test_acc']:.4f} in Epoch {best_acc_info['best_test_acc_epoch']}")
            
        if self.use_wandb_log:
            wandb.log(
                dict( 
                    val_acc = val_acc,
                    test_acc = test_acc,
                ), 
                step = epoch,
            )

    def summary(self) -> dict[str, Any]:
        best_acc_info = self.get_best_acc()
        
        if self.use_wandb_log:
            wandb.summary['best_val_acc'] = best_acc_info['best_val_acc']
            wandb.summary['best_val_acc_epoch'] = best_acc_info['best_val_acc_epoch']
            wandb.summary['best_test_acc'] = best_acc_info['best_test_acc']
            wandb.summary['best_test_acc_epoch'] = best_acc_info['best_test_acc_epoch']

        if self.use_log:
            log_info("[Summary]")
            log_info(f"    Best Val Acc: {best_acc_info['best_val_acc']:.4f} in Epoch {best_acc_info['best_val_acc_epoch']}")
            log_info(f"    Best Test Acc: {best_acc_info['best_test_acc']:.4f} in Epoch {best_acc_info['best_test_acc_epoch']}")

        return best_acc_info 
