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