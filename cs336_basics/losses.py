import torch


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # logits: (..., vocab_size)；targets: (...)
    # 减去每行最大值，避免 exp 溢出
    shifted = logits - logits.max(dim=-1, keepdim=True).values
    # 取出每个位置对应正确 token 的 logit
    target_logits = shifted.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    losses = torch.log(torch.exp(shifted).sum(dim=-1)) - target_logits
    return losses.mean()
