import torch
from cs336_basics.module import TransformerLM, softmax
from cs336_basics.tokenizer import Tokenizer


@torch.inference_mode()
def decode(
    tokenizer: Tokenizer,
    model: TransformerLM,
    prompt: str,
    max_new_tokens: int,
    context_length: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> str:
    if temperature <= 0:
        raise ValueError("temperature 必须大于 0")
    if not 0 < top_p <= 1:
        raise ValueError("top_p 必须在 (0, 1] 内")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens 不能为负数")

    generated_ids = tokenizer.encode(prompt)
    if not generated_ids:
        raise ValueError("prompt 至少需要编码出一个 token")

    eos_id = tokenizer.token2id(b"<|endoftext|>")
    # next(...)：从迭代器中取出第一个参数
    device = next(model.parameters()).device
    model.eval()

    for _ in range(max_new_tokens):
        # 模型一次最多接收 context_length 个 token
        input_ids = torch.tensor(
            [generated_ids[-context_length:]],
            dtype=torch.long,
            device=device,
        )

        # 只取最后一个位置预测下一个 token 的 logits：(1, vocab_size)
        logits = model(input_ids)[:, -1, :]
        probs = softmax(logits / temperature, dim=-1)
        if top_p < 1.0:
            sorted_probs, sorted_ids = probs.sort(dim=-1, descending=True)
            cumulative_probs = sorted_probs.cumsum(dim=-1)

            # 保留累计概率首次达到 top_p 的那个 token
            keep = cumulative_probs - sorted_probs < top_p
            sorted_probs = sorted_probs * keep
            # 重新归一化
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

            sampled_position = torch.multinomial(sorted_probs, num_samples=1)
            next_id = sorted_probs.gather(dim=-1, index=sampled_position).item()
        else:
            next_id = torch.multinomial(probs, num_samples=1).item()

        if next_id == eos_id:
            break
        generated_ids.append(next_id)

    return tokenizer.decode(generated_ids)


def main() -> None:

    tokenizer = Tokenizer.from_files(
        "data/TinyStory_vocab.pkl",
        "data/TinyStory_merges.pkl",
        special_tokens=["<|endoftext|>"],
    )

    model = TransformerLM(
        vocab_size=len(tokenizer.vocab),
        context_length=256,
        d_model=512,
        num_heads=16,
        num_layers=4,
        d_ff=1344,
        rope_theta=10000.0,
        device=torch.device("cpu"),
    )
    checkpoint = torch.load(
        "checkpoints/latest.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model"])

    text = decode(
        tokenizer,
        model,
        prompt="Once upon a time",
        max_new_tokens=100,
        context_length=256,
        temperature=0.8,
        top_p=0.9,
    )
    print(text)


if __name__ == "__main__":
    main()
