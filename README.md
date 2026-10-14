# CS336 Spring 2025 Assignment 1: Basics

## BPE 训练实现记录

本仓库的 `cs336_basics/train_bpe.py` 实现了 byte-level BPE 训练流程：先统计 GPT-2 风格预分词单元的频次，再从 UTF-8 字节序列开始，反复合并语料中频次最高的相邻字节对。

### 实现流程

1. **按特殊 token 切分文件块。** `find_chunk_boundaries()` 根据文件大小生成近似均匀的候选边界，再向后查找指定的特殊 token，使文本块尽量在文档分隔 token 处开始。读取和查找都按字节进行。
2. **预分词并统计词频。** `pre_tokenization()` 使用 `GPT2_PRETOKENIZATION_PATTERN` 切分普通文本，并先用特殊 token 的正则模式拆开文本。`count_pretokens_parallel()` 将各文本块交给进程池统计，再用 `Counter.update()` 合并各块结果。
3. **初始化字节级状态。** `initialize_bpe_state()` 将每个预分词编码为 UTF-8，再拆成单字节 token。它维护三份数据：`word_tokens` 保存每个预分词当前的 token 序列；`pair_counts` 保存按语料词频加权的相邻 pair 总频次；`pair_to_word` 是反向索引，记录每个 pair 在哪些预分词中出现及其词内出现次数。
4. **选择并合并 pair。** `find_best_pair()` 选择频次最高的 pair；频次相同时，按 pair 的字节序取较大的 pair。`merge_pair_in_word()` 从左到右合并所有不重叠匹配。
5. **增量更新统计。** 每次合并后，只遍历含有该 pair 的预分词；`update_pair_stats()` 从旧 token 序列中移除旧 pair 统计，再将新序列产生的 pair 统计加入全局计数。训练持续到达到 `vocab_size` 或没有可合并 pair。

### 训练入口与结果

当前脚本末尾的 `__main__` 示例使用 TinyStories 训练集、`vocab_size=10000`、特殊 token `<|endoftext|>` 和 `num_process=8`。运行前需确认该入口中的 `input_path` 指向本机实际数据文件。

`train_bpe()` 返回两个对象：

- `vocab`: `dict[int, bytes]`，token ID 到字节串的映射。初始词表由特殊 token 和全部 256 种单字节 token 组成。
- `merges`: `list[tuple[bytes, bytes]]`，按训练顺序记录每一步被合并的左右字节 token。

## Tokenizer 编码与解码实现记录

`cs336_basics/tokenizer.py` 使用训练得到的 `vocab` 和 `merges` 将文本编码为 token ID，并支持从 token ID 还原文本。实现按特殊 token、GPT-2 预分词、byte-level BPE 和 ID 映射分段处理。

### 编码与解码流程

1. **初始化查表。** `Tokenizer.__init__()` 保存 `vocab`、`merges` 和 `special_tokens`，建立 `token2id`（字节串到 ID）以及 `merge2rank`（相邻字节 token 对到合并顺序）的查表。若传入的特殊 token 尚未出现在词表中，则将其 UTF-8 字节串追加到词表。
2. **优先识别特殊 token。** `encode()` 使用按长度从长到短排列、并经过 `re.escape()` 转义的正则表达式切分文本。正则的捕获组会把匹配到的特殊 token 保留在分段结果中；特殊 token 直接查表，不进入普通 BPE 流程。
3. **普通文本预分词。** `_encode_text_segment()` 使用 `GPT2_PRETOKENIZATION_PATTERN` 将普通文本切成 pre-token，再逐个调用 `_encode_token()`。
4. **应用 BPE merges。** `_encode_token()` 将一个 pre-token 编码为 UTF-8 字节并拆为单字节 token。每轮从当前相邻 pair 中选择 rank 最小的 pair，从左向右合并其不重叠出现；没有可合并 pair 后，再将最终字节 token 映射为 ID。
5. **解码。** `decode()` 根据 ID 找回词表中的字节串，按顺序拼接后统一执行 UTF-8 解码，并用 `errors="replace"` 替换不合法的 UTF-8 序列。
6. **流式编码。** `encode_iterable()` 每次读取 iterable 提供的一段文本，调用 `encode()` 并通过 `yield from` 逐个产出 ID，不需要一次性读入整个文件。

### 文件加载与测试记录

作业接口要求类方法名为 `from_files()`，并从序列化的 `vocab`、`merges` 文件构造 tokenizer。当前 `tokenizer.py` 中的方法名是 `form_file()`，且使用 `pickle.load()` 读取文件；使用前应将方法名改为 `from_files()`，并保证序列化端也使用相同的数据格式。现有测试适配器通过 `get_tokenizer()` 直接构造 `Tokenizer(vocab, merges, special_tokens)`，所以这些测试不会覆盖文件加载方法名。

本次提供的 `tests/test_tokenizer.py` 输出中，German、TinyStories、特殊 token 换行以及 iterable 编码相关用例为 `PASSED`。两个内存限制用例为 `SKIPPED`：测试仅在 Linux 上运行这些用例，跳过不代表内存行为已经验证。
