import math
import torch
from collections.abc import Iterable


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError(f"Invalid betas: {betas}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                # 遇到稀疏梯度就立即报错，避免后面的更新计算出错
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0  # 这个参数已更新的次数
                    state["m"] = torch.zeros_like(p)  # 梯度的一阶移动平均
                    state["v"] = torch.zeros_like(p)  # 梯度平方的二阶移动平均
                state["step"] += 1
                t = state["step"]
                m, v = state["m"], state["v"]

                # AdamW 的解耦权重衰减：θ ← θ - lr · weight_decay · θ
                # 等价于：θ ← (1 - lr · weight_decay) · θ
                p.mul_(1 - lr * weight_decay)

                # 梯度的一阶矩、二阶矩
                # m = beta1 * m + (1 - beta1) * grad
                # v = beta2 * v + (1 - beta2) * grad * grad
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 偏差修正后的步长
                adjusted_lr = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                p.addcdiv_(m, v.sqrt().add_(eps), value=-adjusted_lr)
        # 不传 closure 时，loss 一直是 None，step() 就返回 None
        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    # 学习率 η(it) 分三段：
    # 1. 线性预热（it < warmup_iters）：
    #    η = max_learning_rate * it / warmup_iters
    #
    # 2. 余弦下降（warmup_iters <= it <= cosine_cycle_iters）：
    #    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    #    η = min_learning_rate
    #        + 0.5 * (1 + cos(π * progress))
    #        * (max_learning_rate - min_learning_rate)
    #
    # 3. 保持最小学习率（it > cosine_cycle_iters）：
    #    η = min_learning_rate
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters

    if it > cosine_cycle_iters:
        return min_learning_rate

    # 从最大学习率沿半个余弦周期下降到最小值
    if warmup_iters <= it <= cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + (1 + math.cos(math.pi * progress)) * 0.5 * (
            max_learning_rate - min_learning_rate
        )


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
) -> None:
    # 先保存梯度，避免 parameters 是只能遍历一次的生成器
    # 这里保存的是原梯度的引用，故最后的 grad.mul_() 会改动参数上的 p.grad
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    # 所有梯度元素共同组成一个向量，计算它的 L2 范数
    total_norm = torch.sqrt(sum(grad.detach().float().square().sum() for grad in grads))
    if total_norm <= max_l2_norm:
        return

    # 计算缩放系数 s
    scale = max_l2_norm / (total_norm + 1e-6)
    with torch.no_grad():
        for grad in grads:
            grad.mul_(scale)
