from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
import os

import numpy as np

from cs336_basics.tokenizer import Tokenizer

EOS = b"<|endoftext|>"
_WORKER_TOKENIZER = None


def _init_worker(vocab_path: Path, merges_path: Path) -> None:
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = Tokenizer.from_files(
        vocab_path,
        merges_path,
        special_tokens=[EOS.decode("ascii")],
    )
    if len(_WORKER_TOKENIZER.vocab) > 65536:
        raise ValueError("词表超过 uint16 的范围")


def _next_eos_end(file, offset: int) -> int:
    """从指定位置向后找 EOS，返回 EOS 结束后的字节位置。"""
    file.seek(offset)
    overlap = b""

    while block := file.read(1024 * 1024):
        data = overlap + block
        position = data.find(EOS)
        if position != -1:
            return file.tell() - len(data) + position + len(EOS)
        overlap = data[-(len(EOS) - 1) :]

    return file.tell()


def _ranges(path: Path, desired_shards: int) -> list[tuple[int, int]]:
    size = path.stat().st_size
    boundaries = [0]

    with path.open("rb") as file:
        for i in range(1, desired_shards):
            target = size * i // desired_shards
            boundary = _next_eos_end(file, target)
            if boundaries[-1] < boundary < size:
                boundaries.append(boundary)

    boundaries.append(size)
    return list(zip(boundaries, boundaries[1:]))


def _documents_in_range(path: Path, start: int, end: int):
    """只读取分配给当前 worker 的区间，并逐篇产出文本。"""
    with path.open("rb") as file:
        file.seek(start)
        remaining = end - start
        pending = b""

        while remaining:
            block = file.read(min(4 * 1024 * 1024, remaining))
            if not block:
                raise IOError("读取数据时遇到意外的文件末尾")

            remaining -= len(block)
            pieces = (pending + block).split(EOS)

            for piece in pieces[:-1]:
                yield (piece + EOS).decode("utf-8")

            pending = pieces[-1]

        if pending:
            yield pending.decode("utf-8")


def _encode_shard(
    task: tuple[int, Path, int, int, Path],
) -> tuple[int, Path, int]:
    index, input_path, start, end, shard_path = task
    tokenizer = _WORKER_TOKENIZER
    if tokenizer is None:
        raise RuntimeError("worker tokenizer 尚未初始化")

    ids = []
    count = 0

    with shard_path.open("wb") as output:
        for document in _documents_in_range(input_path, start, end):
            ids.extend(tokenizer.encode(document))

            if len(ids) >= 250_000:
                np.asarray(ids, dtype=np.uint16).tofile(output)
                count += len(ids)
                ids.clear()

        if ids:
            np.asarray(ids, dtype=np.uint16).tofile(output)
            count += len(ids)

    return index, shard_path, count


def encode_parallel(
    input_path: Path,
    output_path: Path,
    vocab_path: Path,
    merges_path: Path,
    workers: int = 4,
) -> int:
    if workers < 1:
        raise ValueError("workers 至少为 1")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ranges = _ranges(input_path, workers * 2)

    with TemporaryDirectory(
        prefix="token_shards_",
        dir=output_path.parent,
    ) as temp:
        temp_dir = Path(temp)
        tasks = [
            (index, input_path, start, end, temp_dir / f"{index:04d}.u16")
            for index, (start, end) in enumerate(ranges)
        ]

        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(vocab_path, merges_path),
        ) as pool:
            # map 按 tasks 的顺序返回，保证最终 token 顺序不变。
            results = list(pool.map(_encode_shard, tasks))

        total = sum(count for _, _, count in results)
        if total == 0:
            raise ValueError(f"没有编码出 token：{input_path}")

        temporary_npy = temp_dir / "tokens.npy"
        output = np.lib.format.open_memmap(
            temporary_npy,
            mode="w+",
            dtype=np.uint16,
            shape=(total,),
        )

        position = 0
        for _, shard_path, count in results:
            if count == 0:
                continue

            shard = np.memmap(shard_path, dtype=np.uint16, mode="r")
            for offset in range(0, count, 1_000_000):
                stop = min(offset + 1_000_000, count)
                output[position + offset : position + stop] = shard[offset:stop]

            position += count
            del shard

        output.flush()
        del output
        os.replace(temporary_npy, output_path)

    return total


def main() -> None:
    data = Path("data")
    vocab = data / "TinyStoriesV2_vocab.pkl"
    merges = data / "TinyStoriesV2_merges.pkl"

    for name, workers in (("train", 8), ("valid", 2)):
        input_path = data / f"TinyStoriesV2-GPT4-{name}.txt"
        output_path = data / f"tinystories_{name}_tokens.npy"

        count = encode_parallel(
            input_path,
            output_path,
            vocab,
            merges,
            workers=workers,
        )
        print(f"{input_path} -> {output_path}: {count:,} tokens", flush=True)


if __name__ == "__main__":
    main()
