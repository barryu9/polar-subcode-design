# 复现图 5–10 中的 Proposed 系列码

[English](README.md) · 简体中文

## 运行构造

在仓库根目录激活 CPython 3.12 环境后运行下列命令。

```sh
python -m pip install -r reproduction/table_ii/requirements.txt
python reproduction/proposed.py check
python reproduction/proposed.py cold --cases n256_k64 --output runs/short
python reproduction/proposed.py cold --output runs/table-ii
```

第一条命令安装复现使用的数值计算库和求解器版本。从头构造需要支持 C++17 的 GCC 或 Clang，可将编译器加入 PATH，或用环境变量 `CXX` 指定路径。Windows 上编译 CPython 的 SCL 扩展还需要 MSVC 构建工具。脚本先从仓库源码编译原生计算内核，再开始构造计时。

`check` 核对随仓库提供的九套配置，检查码的身份和冻结表达式的因果性。`cold` 从头搜索，每次均需指定新的输出目录。初次使用可选 `n256_k64`，跟随例 9 的构造过程。省略 `--cases` 时，脚本依次运行表 II 的八条构造，并从图 10 的主码候选库提取消融码 Proposed-NP。

## 按图号查找构造脚本

每个脚本分别列出该码的主要步骤、信道选择、候选搜索预算和随机种子。[python/](python/) 中的函数负责具体搜索、重量谱核查和译码前缀评分。六个主码均从空的冻结表达式与候选库开始。码长相同的设计也可以使用不同的信息集搜索。例 8 与图 8 的 Proposed 码同为 [256,128]，均达到最小距离 24，使用的信息集不同。例 9 构造的是图 7 的 [256,64] Proposed 码。

| 图 | 曲线和码参数 | Python 入口 |
|---|---|---|
| 5 | Proposed [64,32] | [结构反馈、信息位置交换与细化](examples/reproduce_n64_k32.py) |
| 6 | Proposed [128,64] | [信息集排序与核心部分的重量谱细化](examples/reproduce_n128_k64.py) |
| 7 | Proposed [256,64] | [可靠性引导的信息位置交换与系数构造](examples/reproduce_n256_k64.py) |
| 8 | Proposed [256,128] | [信息集搜索与联合细化](examples/reproduce_n256_k128.py) |
| 9 | Proposed [256,171] | [信息集选择、整行细化与局部细化](examples/reproduce_n256_k171.py) |
| 10 | Proposed、Proposed-NP [1024,512] | [信息集搜索、后缀细化与前缀保护](examples/reproduce_n1024_k512.py) |
| 10 | Proposed-LM [1024,512] | [从主码继续降低最小重量码字数](examples/reproduce_n1024_k512_lm.py) |
| 10 | Proposed-D32 [1024,512] | [从主码继续构造最小距离 32 的码](examples/reproduce_n1024_k512_d32.py) |

## 读取构造结果

搜索完成并固定候选库后，核查程序根据完整信息集和全部冻结系数列的 SHA-256 哈希匹配论文曲线，将每套匹配的文件写入 `runs/table-ii/selected/`。已有完整运行结果时，可用 `python reproduction/proposed.py collect --output runs/table-ii` 再次导出配置。

`*.profile.json` 给出固定信息集并开始设计系数时选定的信息位置。配对的 `*.code.json` 还给出每个动态冻结位置所用的较早信息位置，将对应比特异或即得其值。未列出的冻结位置取零，所有下标均从零开始。两种文件都可用 `polar_subcodes.code.load_code` 加载，也可以直接读取[仓库附带的配置](../../configs/proposed)。

## 追踪长度 1024 的变体

要跟着代码读懂图 10 的消融对比，可从主码构造脚本内层循环中的两次调用读起。第一步在每个可编辑的冻结位置，用设计帧上竞争译码前缀的得分选择冻结表达式。第二步将这些表达式与同一信息集上已经细化的后缀组合起来。将组合码同前缀保护前的码比较，就可以在重量不超过 24 的码字集合相同的条件下观察译码表现的变化。[主码构造脚本](examples/reproduce_n1024_k512.py) · [表达式评分](python/design_scl_protected_prefix.py) · [前后缀组合](python/compose_scl_prefix_completion.py)

Proposed-NP 是后缀细化完成、尚未进行前缀保护的中间码。Proposed 在同一后缀上采用保持截断谱的前缀。Proposed-LM 和 Proposed-D32 都从新生成的主码候选 `combined_02_floor16` 继续构造。用 `--cases` 单独选择某个变体时，脚本会自动运行共享阶段。信息位置交换、设计噪声种子和搜索预算可查阅[变体构造脚本](examples/reproduce_n1024_variants.py)。

Proposed-D32 的最后一步保持重量 31 以内的谱。表 II 中的重量 32 码字数来自对匹配码单独进行的完整计数。若要重做该计数，请使用运行日志中[最后一步](python/run_ga_priority_d32_origin_prefix.py)的命令，另选新输出目录，并去掉 `--defer-spectrum-verification`。

## 理解构造时间

`table_ii_times.csv` 给出表 II 的两种累计用时。**w/o refinement** 已计入可靠性引导的信息集搜索和冻结系数构造。此时已获得兼顾可靠性且达到距离目标的码，单独计时的组装及验证也计入本列。**w/ refinement** 还计入后续重量谱细化和前缀保护。两种长度 1024 的变体在各自用时中均计入一次共享主码阶段。

`stage_breakdown.csv` 列出各阶段的秒数及占比，`paper_step_summary.csv` 按论文中的操作归类。[论文计时](published_timings/table_ii_times.csv)来自原实验的串行运行。构造协议将先前确定的设计决策作为输入，计时覆盖自动搜索、设计样本上的译码和运行中的核查。FER 曲线另用独立仿真得到。机器速度和数值计算中出现的并列结果可能影响限时搜索实际遍历的候选。若完整候选库中找不到论文曲线对应的码，脚本会报告错误。
