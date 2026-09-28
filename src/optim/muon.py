import torch

@torch.compile()
def zeropower_via_newtonschulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    ---

    Basically squash huge updates like normalization so huge wright updates become normalized enough to support stable training.

    #### Works on the principle of Frobenius Normalization.
    It puts matrix at controlled overall scale before Newton-Schulz.

    ```
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    ```

    means: Take entire matrix, calculate its overall magnitude, then divide every element by that magnitude.
    """

    assert G.ndim >= 2

    # Coefficients for Newton-Schulz polynomial.
    # Repeated iterations push singular values toward a similar magnitude.
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()            # Faster/Lower-memory matrix operations

    # Newton-Schulz is easier/stabler when rows <= columns.
    # Transpose back after orthogonalization.
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Normalize entire matrix by its Frobenius norm.
    # Prevents excessively large input from destabilizing the iteration.
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    for _ in range(steps):
        # Compute X Xᵀ to capture relationships between matrix directions.
        A = X @ X.mT

        # Build Newton-Schulz correction term.
        B = b * A + c * A @ A

        # Update X toward its zeroth-power / orthogonalized form.
        X = a * X + B @ X

    # Restore original matrix orientation.
    if G.size(-2) > G.size(-1):
        X = X.mT

    return X


class Muon(torch.optim.Optimizer):
    """Muon - MomentUm Orthogonalized by Newton-schulz"""
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                # Initialize momentum buffer if first time
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]

                # Update momentum buffer: buf = momentum * buf + (1-momentum) * grad
                buf.lerp_(g, 1 - group["momentum"])

                # Apply Nesterov momentum if enabled, otherwise use standard momentum
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf

                # Apply zero-power normalization via Newton-Schulz iterations (make it close to orthonormal)
                g = zeropower_via_newtonschulz(g, steps=group["ns_steps"])

                # Update parameters with adaptive scaling based on parameter shape
                p.add_(g.view_as(p), alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)
                # Updates parameters with an adaptive learning rate that scales based on the parameter tensor's aspect ratio (height/width).
                # For matrices where height > width, it increases the effective learning rate by √(height/widt
