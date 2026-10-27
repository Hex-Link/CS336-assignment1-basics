import torch
import wandb  # 记录训练指标
import time  # 统计训练已经运行的时间
import argparse  # 解析命令行参数，例如 --batch-size 32
from pathlib import Path  # 表示文件路径
import numpy as np

from cs336_basics.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.losses import cross_entropy
from cs336_basics.module import TransformerLM
from cs336_basics.data import get_batch
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule, gradient_clipping


# 推理模式
@torch.inference_mode()
def evaluate(
    model: TransformerLM,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    num_batches: int,
) -> float:
    model.eval()
    total_loss = 0.0
    for _ in range(num_batches):

        # 从验证集随机取一批连续 token
        # inputs、targets 的形状都是 (B, S)：B = batch_size，S = context_length
        # targets[b, s] 是 inputs[b, s] 后面的那个 token
        inputs, targets = get_batch(dataset, batch_size, context_length, device)

        # 模型对每个位置预测词表中每个 token 的原始分数
        # V = vocab_size；logits 形状为 (B, S, V)
        logits = model(inputs)

        # cross_entropy 对每个位置的正确 token 计算损失
        # 再取平均，得到一个标量 Tensor
        # .item() 把它转换成 Python 数值，便于累计
        total_loss += cross_entropy(logits, targets).item()

    # 当前 evaluate 只在训练循环中调用，因此结束后切回训练模式
    model.train()
    # 返回这些验证 batch 的平均损失
    return total_loss / num_batches


def parse_args() -> argparse.Namespace:
    # 让数据路径、模型规模、学习率等都能在运行命令中指定
    parser = argparse.ArgumentParser(description="Train a CS336 Transformer LM")
    # 输入是已经编码成 token ID 的一维 .npy 数组，不是原始文本
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--val-data", type=Path, required=True)

    # checkpoint 是本次训练保存到的路径
    # resume 是可选的、启动时要读取的旧 checkpoint
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resume", type=Path)

    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)

    # 构造 TransformerLM 所需的模型配置
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    # batch-size：每次更新使用多少条长度为 context_length 的序列
    # max-steps：总共要完成多少次 optimizer.step()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, required=True)

    # 余弦学习率调度，以及 AdamW、梯度裁剪的超参数
    parser.add_argument("--max-lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    # 分别控制 W&B 记录、验证和 checkpoint 保存的频率
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--wandb-project", default="cs336")
    parser.add_argument("--run-name")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # 训练循环的 it 是 0, 1, ..., max_steps-1
    # 正常情况下让最后一个 it 对应余弦调度的终点 min_lr
    cosine_cycle_iters = max(args.max_steps - 1, 1)
    if args.max_steps < 1 or not 0 <= args.warmup_steps < cosine_cycle_iters:
        raise ValueError(
            "需要 max_steps >= 1 且 0 <= warmup_steps < cosine_cycle_iters"
        )
    # 固定新训练的随机初始化与 NumPy 随机采样序列
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # mmap_mode="r"：数据仍保存在磁盘，抽取 batch 时按需读取
    train_data = np.load(args.train_data, mmap_mode="r")
    val_data = np.load(args.val_data, mmap_mode="r")

    # 用命令行配置创建模型；device 决定参数所在设备
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=torch.device(args.device),
    )
    # optimizer 保存每个参数的一阶矩、二阶矩等 AdamW 状态
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    # start_it 是已完成的更新次数
    # 例如 checkpoint 保存 iteration=500，下一次循环从 it=500 开始
    start_it = (
        load_checkpoint(args.resume, model, optimizer) if args.resume is not None else 0
    )
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    # argparse 中的路径是 Path；转成字符串后写入 W&B 配置
    wandb_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    start_time = time.perf_counter()

    # 一个 W&B run 对应这次训练；退出 with 时结束 run
    with wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        config=wandb_config,
    ) as run:
        # 图表以实际训练步数 global_step 为横轴
        run.define_metric("global_step")
        for metric in ("train/loss", "val/loss", "lr", "elapsed_s"):
            run.define_metric(metric, step_metric="global_step")

        for it in range(start_it, args.max_steps):
            # it 是本次更新前的迭代编号，用它查询本次要用的学习率
            lr = get_lr_cosine_schedule(
                it,
                args.max_lr,
                args.min_lr,
                args.warmup_steps,
                cosine_cycle_iters,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr

            # 从训练 token 数组中随机抽取输入与下一 token 目标
            # inputs、targets: (B, S)
            inputs, targets = get_batch(
                train_data,
                args.batch_size,
                args.context_length,
                args.device,
            )
            optimizer.zero_grad()

            # 前向：(B, S) -> (B, S, V)；loss 是标量
            logits = model(inputs)
            loss = cross_entropy(logits, targets)

            # 反向计算各参数的 p.grad；在更新参数前限制总梯度范数
            loss.backward()
            gradient_clipping(model.parameters(), args.max_grad_norm)
            optimizer.step()

            # 更新已完成，所以 step = it + 1；保存 checkpoint 时用 step
            step = it + 1
            should_eval = step % args.eval_interval == 0 or step == args.max_steps

            # train/loss 是刚才这一个训练 batch 的 loss ,不是整个训练集的平均 loss
            metrics = {
                "global_step": step,
                "train/loss": loss.item(),
                "lr": lr,
                "elapsed_s": time.perf_counter() - start_time,
            }

            if should_eval:
                # 从验证集随机抽取 eval_batches 个 batch，
                # 以其平均 loss 估计当前验证表现
                val_loss = evaluate(
                    model,
                    val_data,
                    args.batch_size,
                    args.context_length,
                    args.device,
                    args.eval_batches,
                )
                metrics["val/loss"] = val_loss
                print(
                    f"step={step} train_loss={metrics['train/loss']:.4f} "
                    f"val_loss={val_loss:.4f} lr={lr:.3e}",
                    flush=True,
                )
            # 到记录间隔，或刚完成验证时，写入一次 W&B
            if step % args.log_interval == 0 or should_eval:
                run.log(metrics)

            # 定期保存，并保证最后一步一定保存
            if step % args.save_interval == 0 or step == args.max_steps:
                save_checkpoint(model, optimizer, step, args.checkpoint)


if __name__ == "__main__":
    main()
