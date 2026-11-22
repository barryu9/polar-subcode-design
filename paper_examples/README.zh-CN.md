# 论文例 1–9

[English](README.md) · **简体中文**

在仓库根目录运行下列命令。编号与论文一致，截止重量提高到 4 的计算归在例 4 中。例 1–6 共用同一个长度为 16 的码和两条冻结表达式，每个脚本集中展示一个例子的主要计算。

```shell
python -m paper_examples all
python -m paper_examples 4
python -m paper_examples.example07
python -m unittest paper_examples.test_examples
```

| 例 | 脚本 | 可以核对的内容 |
|---|---|---|
| 1 | [example01.py](example01.py) | 建立输入校验矩阵和输入生成矩阵，经极化变换得到实际生成行。 |
| 2 | [example02.py](example02.py) | 在截止重量 3 下改变较后的系数，核对谱的变化，并恢复与该系数有约束关系的另一个系数。 |
| 3 | [example03.py](example03.py) | 枚举公共超码的 64 个输入，将 9 个重量 2 码字分组，再检验所有 16 个设计，得到 4 个可行配置。 |
| 4 | [example04.py](example04.py) | 将两条释放的校验嵌入七维校验空间，计算两个商投影的秩，将满足条件的外部赋值提升为两个重量 4 码字。 |
| 5 | [example05.py](example05.py) | 将秩 11 的联合观测限制到四个坐标，得到秩 2 的映射，再通过核从可行代表元恢复全部可行配置。 |
| 6 | [example06.py](example06.py) | 对比无条件均值与两个来源组的均值，依次确定冻结表达式系数。 |
| 7 | [example07.py](example07.py) | 枚举全部 16,384 个外层矩阵，核验两个计数界和图 3 中两条选择路径。 |
| 8 | [example08.py](example08.py) | 在指定的 $[256,128]$ 信息集上补全系数，比较末列均值、计算固定前缀下的可行数下界，核验重量 24 的码字。 |
| 9 | [example09.py](example09.py) | 求解两个长度 32 的域，按后缀反馈作两次信息位置交换，再解出外层系数，得到距离 48 的码。 |

## 从头构造两个长度 256 的例子

默认运行方式适合对照正文。例 8 读取记录中的关键消息目录与条件均值，重算联合秩、补全数下界，并编码核验达到目标重量的码字。例 9 读取记录中的 GA 次序和内部表达式，重新求解两个长度 32 的域、生成父节点的禁止集，并据此逐项核对父节点的系数。两份码配置 JSON 都列出了各动态冻结位置所依赖的信息位。

```shell
python tools/build_native.py
python -m paper_examples.example08 --construct --verify-spectrum
python -m paper_examples.example08 --recompute-means
python -m paper_examples.example09 --construct --verify-spectrum
```

- **例 8。** `--construct` 按正文从 RM–GA 信息集作七处替换，从零系数求解右码，枚举完整左消息目录，再依次构造 39 个外层列，与记录逐系数比对。`--recompute-means` 对三个前缀重新计算禁止集与可达仿射像的交集。`--verify-spectrum` 核验右码以及全部 920 个非零左消息的陪集，配合重量 24 的见证确定最小距离。补全数下界 354,383,872 适用于固定早先 38 列及末列前六位之后的 33 个自由位。此处的 $[256,128]$ 码用于固定信息集的流程示例，第六部分的 **Proposed** 曲线另行选取了信息集。
- **例 9。** `--construct` 以设计信噪比 2 dB 重算 RM–GA 次序，执行后缀预筛和两次交换，重新构造长度 64、128 的分量码，再解出 48 个外层系数，并与原设计逐个核对。第一次失败的证明范围仅限长度 32 后缀的 20 系数域。`--verify-spectrum` 结合 RM 包含关系及重量 48 的见证，对各后缀与最终码作精确距离核验。

可分别查阅[例 8 的输入与计数](data/example08_trace.json)和[例 9 的输入与计数](data/example09_trace.json)。两个 `--construct` 入口都是先计算系数，再分别与[例 8 的码配置](data/example08_code.json)和[例 9 的码配置](data/example09_code.json)逐项比对。
