import argparse
import os
from typing import BinaryIO
from collections import Counter
import regex as re
import multiprocessing
from functools import partial
import pickle
from dataclasses import dataclass
from heapq import heapify, heappop, heappush

GPT2_PRETOKENIZATION_PATTERN = (
    # 撇号后匹配常见英文缩写后缀；(?:...) 是不捕获分组，| 表示“或”，[sdmt] 匹配其中一个字符。
    r"'(?:[sdmt]|ll|ve|re)"
    # ? 表示前面的普通空格可出现 0 或 1 次；\p{L} 匹配一个或多个 Unicode 字母。
    r"| ?\p{L}+"
    # 同样允许前面带一个普通空格；\p{N} 匹配一个或多个 Unicode 数字。
    r"| ?\p{N}+"
    # [^...] 中位于开头的 ^ 表示取反：匹配非空白、非字母、非数字字符（如标点或符号）。
    r"| ?[^\s\p{L}\p{N}]+"
    # \s+ 匹配一段空白；(?!\S) 是负向前瞻，要求后面不是非空白字符，但不消耗后面的字符。
    r"|\s+(?!\S)"
    # 匹配其他空白序列，作为空白字符的兜底分支。
    r"|\s+"
)

BytePair = tuple[bytes, bytes]


@dataclass(frozen=True, slots=True)
class PairPriority:
    count: int
    pair: BytePair

    def __lt__(self, other: "PairPriority") -> bool:
        # heapq 是最小堆；这里让频次大、平局时字节序大的项先出堆
        return (self.count, self.pair) > (other.count, other.pair)


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,  ##作为多进程数目使用
    split_special_token: bytes,
) -> list[int]:
    """
    将文件切分成可以独立计数的若干部分。
    如果边界最终发生重叠，可能会返回更少的分块。
    """
    assert isinstance(
        split_special_token, bytes
    ), "必须将特殊 token 表示为字节串(bytestring)"

    # 获取文件总字节数
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # 分块边界位置的初始猜测值，均匀间隔分布
    # 分块从上一个索引处开始，不包含最后一个索引（左闭右开）
    # chunk_boundaries 存的是每块的起始位置，以及整个文件的末尾位置
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


##预分词函数，使用GPT-2正则化方法
def pre_tokenization(
    text: str,
    special_tokens: list[str],
    pattern: str = GPT2_PRETOKENIZATION_PATTERN,
) -> Counter[str]:
    """切开特殊 token，再统计每个正则预分词单元出现的次数。"""
    if special_tokens:
        special_tokens_pattern = "|".join(
            re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)
        )
        segments = re.split(special_tokens_pattern, text)
    else:
        segments = [text]

    word_counts: Counter[str] = Counter()

    for segment in segments:
        for match in re.finditer(pattern, segment):
            word_counts[match.group(0)] += 1

    return word_counts


# 读取语料分块，并用多个进程统计各块的预分词频次。
def count_pretokens_parallel(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    num_process: int,
) -> Counter[str]:
    if num_process < 1:
        raise ValueError("num_process至少为1")
    if any(not token for token in special_tokens):
        raise ValueError("special_tokens不能包含空字符串")
    # 分词计数
    word_counts: Counter[str] = Counter()
    with open(input_path, "rb") as file:
        # 边界查找函数一次按一个特殊 token 对齐。
        # 通常这里传入文档分隔符，例如 <|endoftext|>。
        boundaries = find_chunk_boundaries(
            file, num_process, special_tokens[0].encode("UTF-8")
        )

        # 返回一个chunk大小文本字符串的生成器
        def iter_chunks():
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                file.seek(start)
                chunk_bytes = file.read(end - start)
                yield chunk_bytes.decode("UTF-8")

        chunk_num = len(boundaries) - 1
        # 只有一个 chunk 时直接在当前进程处理。
        if chunk_num <= 1 or num_process == 1:
            for chunk in iter_chunks():
                word_counts.update(pre_tokenization(chunk, special_tokens))
            return word_counts

        # 多chunk多进程
        worker_count = min(chunk_num, num_process)

        # 用 partial 预先绑定 special_tokens。
        # 之后调用 worker_function(chunk) 时，等价于：
        # pre_tokenization(chunk, special_tokens=special_tokens)
        worker_function = partial(
            pre_tokenization,
            special_tokens=special_tokens,
        )
        # 创建包含 worker_count 个子进程的进程池。
        # with 代码块结束时会清理进程池资源。
        with multiprocessing.Pool(processes=worker_count) as pool:
            # iter_chunks() 逐个提供文本块；
            # 进程池把每个文本块交给 worker_function 并行处理。
            # imap_unordered 会按任务完成的先后返回结果，不保证原始顺序。
            for chunk_counts in pool.imap_unordered(
                worker_function,
                iter_chunks(),
            ):
                # 这句在主进程中执行：
                # 把当前文本块的词频累加到总词频 word_counts 中。
                word_counts.update(chunk_counts)

    return word_counts


def initialize_bpe_state(
    word_counts: Counter[str],
    vocab: list[bytes],
) -> tuple[
    dict[str, list[bytes]],
    Counter[BytePair],
    dict[BytePair, Counter[str]],
]:
    """初始化每个预分词的字节序列、pair 频次和反向索引。"""
    word_tokens: dict[str, list[bytes]] = {}
    pair_counts: Counter[BytePair] = Counter()
    """反向索引是一个字节对出现在哪些词里"""
    pair2word: dict[BytePair, Counter[str]] = {}
    for word, word_frequency in word_counts.items():
        word_bytes = word.encode("utf-8")

        if word_bytes in vocab:
            continue

        tokens: list[bytes] = [bytes([byte]) for byte in word_bytes]
        word_tokens[word] = tokens

        for left, right in zip(tokens, tokens[1:]):
            pair = (left, right)
            # 语料级 pair 频次要乘上这个预分词的出现次数。for循环实现乘法
            pair_counts[pair] += word_frequency

            # 反向索引记录 pair 在这个预分词内部出现几次。
            pair2word.setdefault(pair, Counter())[word] += 1

    return word_tokens, pair_counts, pair2word


def find_best_pair(
    pair_counts: Counter[BytePair],
    pair_heap: list[PairPriority],
    changed_pairs: set[BytePair],
) -> BytePair | None:
    """选择频次最高的 pair；平局时选择字典序更大的 pair。"""
    # 一整轮更新结束后，每个涉及的 pair 只按最终频次入堆一次
    for pair in changed_pairs:
        count = pair_counts.get(pair, 0)
        if count > 0:
            heappush(pair_heap, PairPriority(count, pair))

    # 控制惰性删除留下的过期记录数量
    if len(pair_heap) > 4 * len(pair_counts) + 1024:
        pair_heap[:] = [
            PairPriority(count, pair) for pair, count in pair_counts.items()
        ]
        heapify(pair_heap)

    while pair_heap:
        entry = heappop(pair_heap)
        if pair_counts.get(entry.pair, 0) == entry.count:
            return entry.pair

    return None


def merge_pair_in_word(
    tokens: list[bytes],
    left: bytes,
    right: bytes,
) -> list[bytes]:
    """在一个预分词的 token 序列中合并所有不重叠的 (left, right)。"""
    merged_token = left + right
    merged_tokens: list[bytes] = []

    index = 0
    while index < len(tokens) - 1:
        if tokens[index] == left and tokens[index + 1] == right:
            merged_tokens.append(merged_token)
            index += 2
        else:
            merged_tokens.append(tokens[index])
            index += 1

    merged_tokens.extend(tokens[index:])
    return merged_tokens


def update_pair_stats(
    word: str,
    old_tokens: list[bytes],
    new_tokens: list[bytes],
    word_frequency: int,
    pair_counts: Counter[BytePair],
    pair2word: dict[BytePair, Counter[str]],
    changed_pairs: set[BytePair],
) -> None:
    """移除该预分词合并前的 pair 统计，再加入合并后的统计。"""
    for left, right in zip(old_tokens, old_tokens[1:]):
        pair = (left, right)
        changed_pairs.add(pair)
        pair_counts[pair] -= word_frequency
        if pair_counts[pair] <= 0:
            del pair_counts[pair]

        words_with_pair = pair2word[pair]
        words_with_pair[word] -= 1
        if words_with_pair[word] <= 0:
            del words_with_pair[word]

        if not words_with_pair:
            del pair2word[pair]

    for left, right in zip(new_tokens, new_tokens[1:]):
        pair = (left, right)
        changed_pairs.add(pair)
        pair_counts[pair] += word_frequency
        pair2word.setdefault(pair, Counter())[word] += 1


def train_bpe_from_counts(
    word_counts: Counter[str],
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[BytePair]]:
    """根据已统计的预分词词频训练 byte-level BPE。"""
    #  * 是解包：把每个生成列表中的元素逐个放进外层列表。
    #  也可以把这段理解为“先收集特殊 token 的字节串，再接上 256 个单字节串”
    vocab: list[bytes] = [
        *[token.encode("UTF-8") for token in special_tokens],
        *[bytes([i]) for i in range(256)],
    ]
    if vocab_size < len(vocab):
        raise ValueError(f"vocab_size={vocab_size}小于初始词表大小{len(vocab)}")

    # 将每个预分词拆成初始字节 token，并统计相邻对。
    word_tokens, pair_counts, pair_to_word = initialize_bpe_state(word_counts, vocab)

    merges: list[BytePair] = []

    pair_heap = [PairPriority(count, pair) for pair, count in pair_counts.items()]
    heapify(pair_heap)
    changed_pairs: set[BytePair] = set()
    # 3. 重复合并最佳 pair，直到词表达到指定大小或没有 pair 可合并
    while len(vocab) < vocab_size:
        best_pair = find_best_pair(pair_counts, pair_heap, changed_pairs)
        changed_pairs.clear()
        if best_pair is None:
            break

        left, right = best_pair
        vocab.append(left + right)
        merges.append(best_pair)

        # 只需更新含有本轮最佳 pair 的预分词
        # 先复制键，避免更新 pair_to_word 时影响遍历
        affected_words = list(pair_to_word[best_pair])
        for word in affected_words:
            old_tokens = word_tokens[word]
            new_tokens = merge_pair_in_word(old_tokens, left, right)

            update_pair_stats(
                word=word,
                old_tokens=old_tokens,
                new_tokens=new_tokens,
                word_frequency=word_counts[word],
                pair_counts=pair_counts,
                pair2word=pair_to_word,
                changed_pairs=changed_pairs,
            )

            word_tokens[word] = new_tokens

    vocab_dict = {token_id: token for token_id, token in enumerate(vocab)}
    return vocab_dict, merges


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_process: int = 8,
) -> tuple[dict[int, bytes], list[BytePair]]:
    """先统计预分词词频，再训练 byte-level BPE。"""
    word_counts = count_pretokens_parallel(input_path, special_tokens, num_process)
    return train_bpe_from_counts(word_counts, vocab_size, special_tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 CS336 byte-level BPE tokenizer")
    parser.add_argument(
        "--stage",
        choices=("all", "pretokenize", "merge"),
        default="all",
        help=(
            "all: 连续完成两步；pretokenize: 统计并保存词频；"
            "merge: 读取词频并训练合并规则"
        ),
    )
    parser.add_argument("--input-path", default="data/TinyStoriesV2-GPT4-train.txt")
    parser.add_argument("--word-counts-path", default="data/owt_word_counts.pkl")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--num-process", type=int, default=8)

    args = parser.parse_args()
    special_tokens = ["<|endoftext|>"]
    os.makedirs("data", exist_ok=True)

    if args.stage == "pretokenize":
        word_counts = count_pretokens_parallel(
            args.input_path, special_tokens, args.num_process
        )
        with open(args.word_counts_path, "wb") as f:
            pickle.dump(word_counts, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(
            f"预分词统计完成：{len(word_counts)} 个不同的预分词，"
            f"已保存到 {args.word_counts_path}"
        )
        return

    if args.stage == "merge":
        with open(args.word_counts_path, "rb") as f:
            word_counts = pickle.load(f)
        vocab, merges = train_bpe_from_counts(
            word_counts, args.vocab_size, special_tokens
        )
    else:
        vocab, merges = train_bpe(
            input_path=args.input_path,
            vocab_size=args.vocab_size,
            special_tokens=special_tokens,
            num_process=args.num_process,
        )

    print(f"词表大小：{len(vocab)}")
    print(f"合并次数：{len(merges)}")
    with open("data/owt_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    with open("data/owt_merges.pkl", "wb") as f:
        pickle.dump(merges, f)
    for left, right in merges:
        print(left, "+", right, "->", left + right)


if __name__ == "__main__":
    main()
