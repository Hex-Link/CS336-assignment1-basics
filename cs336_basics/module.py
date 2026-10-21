import torch
import torch.nn as nn
import math
from einops import einsum, rearrange


class Linear(nn.Module):
    """不带 bias 的线性变换层。"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        # 创建权重矩阵，形状是 (out_features, in_features)
        # nn.Parameter 将 W 注册为可训练参数, 使它能出现在 model.parameters() 和 state_dict() 中
        self.weight = nn.Parameter(
            torch.empty(
                (out_features, in_features),
                device=device,
                dtype=dtype,
            )
        )
        # 指定的线性层权重方差：std² = 2 / (in_features + out_features)
        std = math.sqrt(2.0 / (out_features + in_features))
        # 使用均值为 0、标准差为 std 的截断正态分布初始化权重
        # 采样值限制在 [-3 * std, 3 * std]
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a=-3.0 * std,
            b=3.0 * std,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in->... d_out")


class Embedding(nn.Module):
    """将整数 token ID 映射为可学习的向量"""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        # 创建 embedding 矩阵，形状为: (词表大小, 每个 token 的向量维度)
        self.weight = nn.Parameter(
            torch.empty(
                (num_embeddings, embedding_dim),
                device=device,
                dtype=dtype,
            )
        )
        # 作业要求 embedding 权重服从 N(0, 1)，
        # 并截断在 [-3, 3] 范围内
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids 中每个整数是词表里的一个索引。
        # 用这些索引从 weight 中取出对应的 embedding 向量。

        # token_ids.shape = (batch_size, sequence_length)
        # 输出 shape = (batch_size, sequence_length, embedding_dim)

        # token_ids 也可以是其他形状 (... )，
        # 输出形状会是 (..., embedding_dim)。
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """沿最后一维进行 RMS 归一化，并乘以可学习的缩放参数"""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.eps = eps
        # RMSNorm 的可学习缩放参数 g，形状为 (d_model,)
        # 初始化为全 1，初始时不会额外缩放归一化后的值
        self.weight = nn.Parameter(
            torch.ones(
                d_model,
                device=device,
                dtype=dtype,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 记录输入类型，计算结束后要转回这个类型
        in_type = x.dtype
        # 平方运算可能造成低精度浮点数溢出；
        # 按作业要求，先将输入升为 float32 再计算归一化
        x = x.to(torch.float32)
        # 对最后一维计算均方值
        # keepdim=True 让结果保留长度为 1 的维度
        # RMS(x) = sqrt(mean(x²) + eps)
        # rsqrt(y) 等价于 1 / sqrt(y)
        r_rsm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        # weight.shape 为 (d_model,)，会广播到所有前置维度
        result = x * r_rsm * self.weight.to(torch.float32)
        # 恢复输入 dtype，输出形状与输入相同
        return result.to(in_type)


def SiLU(x):
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """由 SiLU 门控和线性变换组成的前馈网络。"""

    def __init__(
        self,
        d_model: int,
        d_ff: int,  # 中间维度
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        # W1 和 W3 是并行升维层: d_model -> d_ff
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)
        # W2 是降维层: d_ff -> d_model
        self.w2 = Linear(d_ff, d_model, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU(x) = W2(SiLU(W1 x) ⊙ W3 x)
        gate = SiLU(self.w1(x))
        signal = self.w3(x)
        return self.w2(gate * signal)


class RoPE(nn.Module):
    def __init__(
        self,
        theta: float,  # RoPE 的 θ 值，通常为 10000
        d_k: int,  # query 和 key 向量的维度
        max_seq_len: int,  # 将被输入的最大序列⻓度
        device: torch.device | None = None,
    ):
        super().__init__()
        # 生成位置编号：0, 1, 2, ..., max_seq_len - 1， 形状：(max_seq_len,)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        # 为每一对特征生成频率编号：0, 2, 4, ..., d_k - 2, 形状：(d_k // 2,)
        freq_indices = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
        # 每个位置与每个角频率相乘，得到旋转角度 , 形状：(max_seq_len, d_k // 2)
        angles = einsum(
            positions,
            torch.pow(theta, (-freq_indices) / d_k),
            "seq, freq -> seq freq",
        )
        """等价于 positions.unsqueeze(1) 或者 positions.reshape(-1, 1)"""
        # 预先计算每个位置、每对特征对应的 cos 和 sin
        # register_buffer 表示它们是模型需要携带的张量,但不是通过梯度学习的 Parameter
        # 使用 persistent=False 确保这些缓存不会被保存在 state_dict 中 (因为可以随时重新生成)
        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:
        # x: (..., seq_len, d_k)
        # token_positions: (..., seq_len)
        # cos.shape = (..., seq_len, d_k // 2)
        cos = self.cos_cached[token_positions].to(dtype=x.dtype)
        sin = self.sin_cached[token_positions].to(dtype=x.dtype)
        # 如果 x 为 (batch, heads, seq, d_k)
        # token_positions 为 (batch, seq) 时，在 head 维插入长度为 1 的维度
        while cos.ndim < x.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)
        # 把 x 最后一维拆成偶数下标和奇数下标的元素
        # 两者形状均为 (..., seq_len, d_k // 2)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        output = torch.empty_like(x)
        output[..., 0::2] = x_even * cos - x_odd * sin
        output[..., 1::2] = x_even * sin + x_odd * cos
        return output


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x = x - x.max(dim, keepdim=True).values
    exp = x.exp()
    return exp / exp.sum(dim, keepdim=True)


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Q: (batch_size, ..., n, d_k)
    K: (batch_size, ..., m, d_k)
    V: (batch_size, ..., m, d_v)
    mask: (..., n, m) 或者是可以广播到该形状的布尔张量 (True 表示关注, False 表示屏蔽)
    """
    # 计算分数: Q @ K^T / sqrt(d_k) 每个 query 对每个 key 的分数
    scores = einsum(
        Q, K, "... queries d_k, ... keys d_k -> ... queries keys"
    ) / math.sqrt(Q.shape[-1])
    if mask is not None:
        # 作业要求: 把 mask 为 False 的地方填入 -inf
        scores = scores.masked_fill(mask == False, float("-inf"))
    # 沿 key 维归一化，再对 V 加权求和
    attention_weights = softmax(scores, dim=-1)
    return einsum(
        attention_weights, V, "... queries keys, ... keys d_v -> ... queries d_v"
    )


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        pos_encoder: nn.Module | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model 必须能被 num_heads 整除")
        self.num_heads = num_heads
        self.pos_encoder = pos_encoder
        # 定义 Q, K, V 的投影层
        self.W_q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_k = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_v = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_o = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: (..., seq_len, d_model)
        seq_len = x.shape[-2]
        # 投影后拆分 head：(..., seq_len, d_model) -> (..., num_heads, seq_len, d_k)
        Q = rearrange(
            self.W_q(x), "... seq (head d) -> ... head seq d", head=self.num_heads
        )
        K = rearrange(
            self.W_k(x), "... seq (head d) -> ... head seq d", head=self.num_heads
        )
        V = rearrange(
            self.W_v(x), "... seq (head d) -> ... head seq d", head=self.num_heads
        )
        # RoPE 作用于每个 head 的 Q、K，不作用于 V
        if self.pos_encoder is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            Q = self.pos_encoder(Q, token_positions)
            K = self.pos_encoder(K, token_positions)
        # mask[i, j] 为 True 表示位置 i 可以关注位置 j
        # 下三角使每个位置只能关注自己和之前的 token
        mask = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
        )
        out = scaled_dot_product_attention(Q, K, V, mask=mask)
        out = rearrange(out, "... head seq d -> ... seq (head d)")
        return self.W_o(out)


class Transformer_Block(nn.Module):
    """使用 RoPE 的 pre-norm Transformer block"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,  # 中间维度
        max_seq_len: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        d_head = d_model // num_heads
        if d_model % num_heads != 0 or d_head % 2 != 0:
            raise ValueError(
                "d_model 必须能被 num_heads 整除，且每个 head 的维度必须是偶数"
            )
        self.rope = RoPE(
            theta=rope_theta, d_k=d_head, max_seq_len=max_seq_len, device=device
        )
        self.atten = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            pos_encoder=self.rope,
            device=device,
            dtype=dtype,
        )
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        self.norm1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        token_position: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)
        x = x + self.atten(self.norm1(x), token_position)
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)
        self.layers = nn.ModuleList(
            [
                Transformer_Block(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    rope_theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.output_embedding = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # (batch, seq) -> (batch, seq, d_model)
        x = self.token_embeddings(token_ids)
        for block in self.layers:
            x = block(x)
        # (batch, seq, d_model) -> (batch, seq, vocab_size)
        x = self.final_norm(x)
        return self.output_embedding(x)
