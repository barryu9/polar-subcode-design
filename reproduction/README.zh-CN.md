# 复现图 5–10 中的码

[English](README.md) · 简体中文

请在仓库根目录的 Python 3.12 环境中运行以下命令。先按论文的图号和图例，从 `list` 输出中找到对应的案例编号及 SCL 列表大小 `L`。同一编号串起该码的构造输入、最终冻结表达式和 FER 测量记录。

```sh
python -m pip install -r requirements.txt
python -m reproduction list
python -m reproduction check
python -m reproduction.table_ii_values
```

`check` 读取仓库中的 30 对配置和已保存的 FER 测量点，核对码的标识、载荷长度、译码列表大小，以及错误帧数与总帧数之比。最后一条命令仅从信息集重算表 II 中的三列。例如，图 7 的 **Proposed** 信息集给出 `d_P=32`，在 2.25 dB 下，**Proposed** 和 **RecursCode** 的 GA 信息集可靠性指标分别为 0.113066 和 0.194171。该行中的最小距离 48 对应预编码后的两个码。[表 II 和 FER 测量数据](../results/README.zh-CN.md)

## 重建对比码

```sh
python -m reproduction construct --cases n64_k32_sc_ml --output runs/fig5-sc-ml
```

以图 5 的 **SC-ML** 为例，[references.py](references.py) 读取原作者给出的预编码矩阵 $Z$。文件中每个十六进制整数表示一行，第 $j$ 个比特对应极化输入位置 $j$。对矩阵作行消元即可识别信息集，并将冻结比特写成更早的信息比特之和，与第二节的定义一致。输出目录的 `reference/` 下，`n64_k32_sc_ml.profile.json` 保存信息集，`n64_k32_sc_ml.code.json` 同时给出动态冻结表达式。[矩阵输入文件](reference_inputs/Spec_n64_k32_d12_Z_2G.json) 保留了原文件名和 SHA-256，可用于核对原始规格。

**RecursCode** 和 **Number-polar** 采用相同的矩阵转换。**PAC**、**CRC** 和 **eBCH polar subcode** 各曲线则按[对比码清单](../configs/reference/manifest.json)中的参数生成。对于 CRC 辅助 SCL 译码，信息集包含承载 CRC 的位置，最终配置还给出 CRC 多项式。

## 重建 Proposed 码

```sh
python -m pip install -r reproduction/table_ii/requirements.txt
python -m reproduction construct --cases n256_k64 --output runs/fig7-proposed
```

这条命令沿例 9 的构造过程得到图 7 的 **Proposed** 码。脚本计算比特信道可靠性，搜索信息集，并为最小距离目标 48 构造动态冻结表达式。构造结束后，[proposed.py](proposed.py) 核对所得信息集与每个冻结表达式是否和图 7 的码一致。最终码保存在输出目录下的 `proposed/selected/n256_k64.code.json`。每次构造请使用新的输出目录。省略 `--cases` 可重建全部 30 个码。各案例的脚本入口及编译环境见[表 II 构造指南](table_ii/README.zh-CN.md)。

生成的 `proposed/table_ii_times.csv` 给出两个累计耗时。**w/o refinement** 涵盖考虑可靠性且已得到最小距离达标码的构造过程，并计入单独计时的组合与核验。**w/ refinement** 进一步计入该案例采用的重量谱细化、译码前缀保护及二者的联合反馈。码长 1,024 的 **Proposed-LM** 和 **Proposed-D32** 各自在本行计入一次共有的 **Proposed** 构造耗时。**Proposed-NP** 取自主码构造中加入译码前缀保护前的结果。第六节用独立噪声样本评估入选的码，曲线上的实测点见 [fer.csv](../results/fer.csv)。
