from __future__ import annotations

import os
import pickle
from collections.abc import Iterable, Iterator
import regex as re
from .train_bpe import GPT2_PRETOKENIZATION_PATTERN

BytePair = tuple[bytes, bytes]


class Tokenizer:
    """使用给定词表和 merges 的 byte-level BPE tokenizer。"""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[BytePair],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        # 建立 bytes -> ID 的查表，以及 pair -> merge rank 的查表。
        self.token2id: dict[bytes, int] = {
            token_bytes: token_id for token_id, token_bytes in vocab.items()
        }
        self.merge2rank: dict[BytePair, int] = {
            pair: rank for rank, pair in enumerate(merges)
        }

        # special token 不在词表时追加进去。
        for special_token in self.special_tokens:
            special_bytes = special_token.encode("UTF-8")
            if special_bytes not in self.token2id:
                token_id = len(self.vocab)
                self.vocab[token_id] = special_bytes
                self.token2id[special_bytes] = token_id

        # 捕获组让 re.split 把 special token 本身也放进结果。
        if self.special_tokens:
            special_tokens_pattern = (
                "("
                + "|".join(
                    re.escape(token)
                    for token in sorted(self.special_tokens, key=len, reverse=True)
                )
                + ")"
            )
            self.special_pattern = re.compile(special_tokens_pattern)
        else:
            self.special_pattern = None
        self.pretoken_pattern = re.compile(GPT2_PRETOKENIZATION_PATTERN)

    @classmethod
    def form_file(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None,
    ) -> Tokenizer:
        with open(vocab_filepath, "rb") as vocab_file:
            # 从二进制文件对象 vocab_file 中读取并反序列化一个 Python 对象
            vocab = pickle.load(vocab_file)
        with open(merges_filepath, "rb") as merges_file:
            merges = pickle.load(merges_file)

        return cls(vocab, merges, special_tokens)

    def _encode_token(self, word: str) -> list[int]:
        # UTF-8 字节是 BPE 合并的初始 token
        tokens = [bytes([byte]) for byte in word.encode("UTF-8")]
        while len(tokens) > 1:
            best_pair = None
            best_rank = None

            # 当前相邻 pair 中，rank 最小的优先合并
            for left, right in zip(tokens, tokens[1:]):
                pair = (left, right)
                rank = self.merge2rank.get(pair)
                if rank is not None and (best_pair is None or rank < best_rank):
                    best_pair = pair
                    best_rank = rank
            # 当前没有任何可用的 merge
            if best_pair is None:
                break
            # 从左向右合并当前最佳 pair 的不重叠出现
            merged_tokens = []
            index = 0
            while index < len(tokens):
                if (
                    index + 1 < len(tokens)
                    and tokens[index] == best_pair[0]
                    and tokens[index + 1] == best_pair[1]
                ):
                    merged_tokens.append(best_pair[0] + best_pair[1])
                    index += 2
                else:
                    merged_tokens.append(tokens[index])
                    index += 1
            # 用合并后的新列表进入下一轮
            tokens = merged_tokens
        #  最终的 bytes token 转成 token ID
        return [self.token2id[token] for token in tokens]

    def _encode_text_segment(self, text: str) -> list[int]:
        """编码不含 special token 的普通文本片段。"""
        token_ids = []
        # 每个 match 是一个 pre-token，例如单词、数字或标点片段
        for match in self.pretoken_pattern.finditer(text):
            token_ids.extend(self._encode_token(match.group(0)))
        return token_ids

    def encode(self, text: str) -> list[int]:
        """将整段文本编码成 token ID 列表。"""
        token_ids = []
        # 没有 special token 规则时，整段文本都按普通文本处理
        if self.special_pattern is None:
            return self._encode_text_segment(text)

        # split 使用捕获组时，会把 special token 本身也保留在结果中
        for segment in self.special_pattern.split(text):
            if segment in self.special_tokens:
                token_ids.append(self.token2id[segment.encode("UTF-8")])
            else:
                # 其他片段是普通文本，交给预分词和 BPE 流程处理
                token_ids.extend(self._encode_text_segment(segment))

        return token_ids

    def decode(self, ids: list[int]) -> str:
        # 根据每个 ID 找回对应的 token bytes，再按原顺序拼接
        byte_sequence = b"".join(self.vocab[token_id] for token_id in ids)
        # 先拼接再统一解码，因为一个 UTF-8 字符的字节可能分散在多个 token 中
        return byte_sequence.decode("UTF-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """逐段编码输入，并惰性地产出 token IDs"""
        # iterable 可以是文件对象等；每次只处理当前文本片段
        for text_piece in iterable:
            yield from self.encode(text_piece)
