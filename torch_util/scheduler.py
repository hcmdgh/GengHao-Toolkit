import numpy as np 

__all__ = [
    'CosineDecayScheduler', 
]


class CosineDecayScheduler:
    def __init__(self, 
                 max_val: float, 
                 warmup_steps: int, 
                 total_steps: int):
        self.max_val = max_val
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps 

    def get_val(self, step: int) -> float:
        assert 0 < step <= self.total_steps
        
        if step < self.warmup_steps:
            return self.max_val * step / self.warmup_steps
        elif self.warmup_steps <= step <= self.total_steps:
            return self.max_val * (1 + np.cos((step - self.warmup_steps) * np.pi / (self.total_steps + 1 - self.warmup_steps))) / 2
        else:
            raise AssertionError
