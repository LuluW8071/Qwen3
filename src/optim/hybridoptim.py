import torch
import torch.optim as optim


class HybridOptimizer(optim.Optimizer):
    """Single Optimizer facade over Muon + AdamW. Shares param_groups so LR schedulers hit both."""

    def __init__(self, *optimizers: optim.Optimizer):
        self.optimizers = optimizers
        all_params = [p for o in optimizers for g in o.param_groups for p in g["params"]]
        super().__init__(all_params, defaults={})
        # replace groups with the inner optimizers' own dicts
        self.param_groups = [g for o in optimizers for g in o.param_groups]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for o in self.optimizers:
            o.step()
        return loss

    def zero_grad(self, set_to_none: bool = True):
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {f"opt_{i}": o.state_dict() for i, o in enumerate(self.optimizers)}

    def load_state_dict(self, state_dict):
        for i, o in enumerate(self.optimizers):
            o.load_state_dict(state_dict[f"opt_{i}"])