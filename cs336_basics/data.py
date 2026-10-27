import numpy as np
import torch


def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    # 起点 i 必须留出 context_length 个输入和最后一个目标 token
    start = np.random.randint(0, len(dataset) - context_length, size=batch_size)

    offsets = np.arange(context_length)
    positions = start[:, None] + offsets[None, :]  # (batch_size, context_length)

    inputs = dataset[positions]
    # 语言模型的下一个 token 预测
    targets = dataset[positions + 1]

    return (
        torch.tensor(inputs, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),
    )
