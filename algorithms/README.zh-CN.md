# 论文的四个算法

[English](README.md) · **简体中文**

在仓库根目录运行下列命令。每个模块包含一个可调用的函数和一个小例子。实现沿用论文的行向量极化变换，下标从零起算。程序以二进制形式打印结果时，最右边一位的下标为零。

| 论文位置 | 运行命令 | 阅读重点 |
|---|---|---|
| 算法 1，第 III.C 节，式 (30) | `python -m algorithms.algorithm1_directory` | `list_inputs(code, cutoff)` 先处理右码字重复两次的分支，再枚举外部比特并恢复每个被接受赋值的内部解。返回的是输入 $u$，对其作极化变换得到受重量上限约束的码字 $x$。可用 `from_causal_checks(N, rows)` 将较早的冻结表达式代入校验行。 |
| 算法 2，第 IV.B 节，式 (24)、(34)、(41) | `python -m algorithms.algorithm2_outer` | `complete_outer(left, right, cutoff, domains)` 生成禁止集，按来源逆序选基，再按条件均值从前到后搜索。每个列域给出基准列及允许的独立变化方向。找到解时返回外层矩阵，穷尽域后仍无解则返回 `None`。 |
| 算法 3，第 V.B 节，式 (48)、(52)–(53) | `python -m algorithms.algorithm3_fixed_profile` | `construct(N, profile, target)` 先检查初始域，证明失败后扩大该域，返回码前核验距离。选 `direct` 使用公共超码条件，选 `factored` 枚举分量码对并调用算法 2。 |
| 算法 4，第 V.C–D 节，式 (59)、(61)–(62) | `python -m algorithms.algorithm4_profiles` | `design_profiles(...)` 按调用者给出的可靠性次序、成本和候选族排列单次交换，按后缀类别选出至多 $Q$ 个不同信息集，并保留各阶段通过验证的结果。 |

## 从算法 3 跟踪到算法 4

算法 3 接受可选的后缀长度 `suffix_lengths` 和工作量回调 `tick`。回调抛出 `Deferred` 时，构造暂时停止，返回已获得的后缀结果。当前坐标域穷尽后可再释放一个系数。若整个信息集的搜索空间均已穷尽仍无解，则返回 `infeasible`。算法 4 利用最短的未决后缀和已验证的最长较短后缀安排信息位置的交换。若已证明障碍与系数取值无关，还可排除具有相同后缀的其他信息集。

## 表达式细化与设计观测

若提供 SCL 整帧路径损失的测量函数，算法 4 的教学实现就按式 (62) 比较候选表达式。可将传输全零码字的设计帧对数似然比和列表大小传给 `frame_loss_measure(llrs, list_size)` 来生成测量函数。省略测量函数时，程序会完成谱细化并返回该阶段的结果。对第 VI.C 节报告的 **Proposed** 码，实际构造在每个可编辑位置依据局部前缀评分选择表达式，随后再测量整帧路径损失。[仿真码复现命令](../README.zh-CN.md)

直接求解器会建立低重量目录，必要时遍历所有自由参数。建议先用小码观察搜索树。测试覆盖长度 4 的全部信息集、外层列的仿射域、仅有部分后缀已经解决时的未决返回，以及初始提案为空的情形。

```bash
python -m unittest discover -s tests -p test_paper_algorithms.py -v
```
