# Deterministic Polar Subcode Design via Exact Low-Weight Spectrum Characterization

[English](README.md) · **简体中文**

**Junhua You、Shaohua Wu、Aolin Liu、Yajing Deng、Qinyu Zhang** · 稿件版本 r112

本仓库是论文的可执行配套材料。初次阅读建议先运行例 3 和例 4。例 3 直接枚举一个长度为 16 的码的低重量码字，例 4 在同一码上用极化商投影得到相同的排除条件。若要核对仿真曲线，可按图号、图例和译码列表大小找到对应配置，查看动态冻结表达式，也可以从头构造该码。

## 安装与首次运行

建议使用 Python 3.12，在仓库根目录运行下列命令。入门示例使用 Python 和 NumPy。较大规模构造、原生枚举器和加速译码器需要 C++17 编译器。Windows 可用 MinGW 构建枚举器，用 Microsoft C++ Build Tools 构建 SCL 扩展。Linux 可用 GCC，macOS 可用 Clang。

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m paper_examples 3
python -m paper_examples 4
python -m reproduction list
python -m reproduction check
python -m reproduction.table_ii_values
```

## 对照论文阅读

| 论文位置 | 运行入口 | 可以核对的内容 |
|---|---|---|
| 算法 1–4 | [算法导读](algorithms/README.zh-CN.md) | 递归枚举、逐源补全、固定信息集搜索和信息位置交换。可跟踪算法 3 怎样把未决后缀传给算法 4。 |
| 例 1–6 | [例子导读](paper_examples/README.zh-CN.md) · `python -m paper_examples 5` | 复核长度 16 的手算、投影矩阵的秩，以及从代表元恢复四个可行系数配置的过程。 |
| 例 7 | `python -m paper_examples 7` | 对照长度 32 的逐源补全计数下界和精确计数。 |
| 例 8–9 | `python -m paper_examples 8` · `python -m paper_examples 9` | 跟踪长度 256 的固定信息集构造和信息位置交换。例 8 与图 8 使用不同的信息集。例 9 构造图 7 的 Proposed 码。例子导读给出了从零构造及核验两者重量谱的命令。 |
| 定理 1–3、命题 1–3、推论 1–2 | [理论核查](theory/README.zh-CN.md) · `python -m theory.check_results` | 通过八组有限规模的精确计算，对照论文中的证明。 |
| 第六部分，图 5–10、表 II–III | [码配置](configs/README.zh-CN.md) · [实测数据](results/paper_r112/README.zh-CN.md) | 按曲线核对信息集、动态冻结表达式、译码列表大小和实测帧错误率（FER）。 |

```bash
python -m algorithms.algorithm1_directory
python -m algorithms.algorithm2_outer
python -m algorithms.algorithm3_fixed_profile
python -m algorithms.algorithm4_profiles
python -m paper_examples all
python -m theory.check_results theorem2 proposition3
```

## 一键复现仿真码

下面最后一条命令在新的输出目录中构造图 5–10 的全部 **30 个码**。若只需某条曲线，可以用 `python -m reproduction list` 查看其编号，再用 `--cases` 指定。

```bash
python -m pip install -r reproduction/table_ii/requirements.txt
python -m reproduction construct --cases n256_k64 --output runs/fig7-proposed
python -m reproduction construct --cases n64_k32_crc4_ga n64_k32_sc_ml --output runs/fig5-refs
python -m reproduction construct --output runs/all-curves
```

**9 个 Proposed 系列码。** 构造从空候选库开始，执行原实验确定的搜索和细化步骤，再将生成码的信息位置与全部冻结表达式同仿真配置核对。表 II 计时的是八次构造。第九个码是图 10 的 Proposed-NP，取自主码完成前缀保护前的阶段。核对通过的结果写入 `proposed/selected/`。部分搜索设有运行时限，因此实际遍历的分支可能随机器性能变化。如果未找到原实验使用的码，脚本会在最后的配置核对中报告差异。

**21 个对比码。** SC-ML、RecursCode 和 Number-polar 由文献公开的预编码矩阵重建。PAC、CRC 和 eBCH 码由各自的设计参数构造。结果写入 `reference/`。

**配置与计时。** 每份 `*.profile.json` 给出信息集，配对的 `*.code.json` 给出动态冻结表达式以及该码使用的 CRC 或 PAC 参数。所有下标从零起算。运行 `python -m reproduction check` 可核对随仓库提供的配置。[配置格式与位序](configs/README.zh-CN.md)

在原实验机器上串行执行八次构造约需 100 分钟，共享的主码阶段只计一次。表 II 的 **w/o refinement** 表示已经得到一个兼顾信道可靠性、最小距离达到目标的码所需的时间。**w/ refinement** 进一步计入重量谱细化和前缀保护的时间。前缀保护会在设计样本上执行译码，论文绘图使用的独立 FER 测量数据另列于实测表中。[实测数据表](results/paper_r112/README.zh-CN.md)

## 函数库

| 功能 | 实现 |
|---|---|
| GA、RM–GA、BEC 和 5G NR 信息集 | [profiles.py](polar_subcodes/construction/profiles.py) |
| 规范极化子码、二元线性代数和极化变换 | [code.py](polar_subcodes/code.py)、[algebra.py](polar_subcodes/algebra.py) |
| PAC 与 CRC 编码、预编码矩阵转换、eBCH 构造 | [precoding.py](polar_subcodes/precoding.py)、[ebch.py](polar_subcodes/construction/ebch.py) |
| 载荷编码、SC、SCL 和 CRC 辅助的最终路径选择 | [codec.py](polar_subcodes/codec.py)、[NumPy SCL](polar_subcodes/decoding/full.py)、[C++ SCL](cpp/scl_step_decoder_cpp.cpp) |
| 低重量谱精确计数与完整码字列表 | [谱枚举函数](polar_subcodes/spectrum/)、[算法 1](algorithms/algorithm1_directory.py) |

下面的例子编码一个非零载荷，再将无噪声条件下的对数似然比送入 NumPy SCL 译码器，核对恢复出的载荷。

```python
from polar_subcodes.codec import load_codec

code = load_codec("configs/reference/n64_k32_crc4_ga.code.json")
x = code.encode(37)  # 32 位载荷，包含高位的零
llr = [12.0 * (1 - 2*((x >> i) & 1)) for i in range(code.mother.n)]
assert code.decode(llr, list_size=32, backend="numpy") == [37]
```

```bash
python tools/build_native.py --scl
python main.py inspect configs/proposed/n64_k32.code.json
python main.py enumerate --support configs/proposed/n64_k32.code.json --cutoff 12 --method native --output runs/fig5-spectrum
python main.py enumerate --support configs/proposed/n64_k32.code.json --cutoff 12 --method words --output runs/fig5-words
python main.py simulate --support configs/reference/n64_k32_crc4_ga.code.json --snr 3 --list-size 32 --frames 100 --messages random --backend cpp --output runs/crc-smoke
```

图 5 的 Proposed 码在正重量处的首个非零谱系数为 $A_{12}=2048$。上面两条枚举命令分别计算重数、列出对应码字。仿真命令会生成新的噪声样本。两个译码后端都支持非零载荷，并按载荷码率折算 $E_b/N_0$。选择 `--backend numpy` 使用 Python 译码器，选择 `--list-size 1` 即为 SC 译码。[复现核查记录](docs/R112_VALIDATION.md)

## 参考文献

以下文献介绍了本仓库用到的基础方法以及若干对比码的构造。

- 极化变换与信道递归见 E. Arıkan 的 “Channel Polarization”（IEEE Trans. Inf. Theory，2009）。[原文](https://doi.org/10.1109/TIT.2009.2021379)
- 动态冻结符号和 eBCH 子码见 P. Trifonov 与 V. Miloslavskaya 的 “Polar Codes With Dynamic Frozen Symbols and Their Decoding by Directed Search”（ITW，2013）和 “Polar Subcodes”（IEEE JSAC，2016）。[动态冻结符号](https://doi.org/10.1109/ITW.2013.6691213) · [eBCH 子码](https://doi.org/10.1109/JSAC.2015.2504269)
- 极化码的 GA 构造见 P. Trifonov 的 “Efficient Design and Decoding of Polar Codes”（IEEE Trans. Commun.，2012）。实现中使用的 GA 校验节点近似见 S.-Y. Chung、T. Richardson 和 R. Urbanke 的工作（IEEE Trans. Inf. Theory，2001）。[极化码 GA 构造](https://doi.org/10.1109/TCOMM.2012.081512.110872) · [GA 近似](https://doi.org/10.1109/18.910580)
- 列表译码见 I. Tal 与 A. Vardy 的 “List Decoding of Polar Codes”（IEEE Trans. Inf. Theory，2015）。PAC 码见 E. Arıkan 的 “From Sequential Decoding to Channel Polarization and Back Again”（IEEE ITS Newsletter，2019）。CRC6、CRC11 和 NR 母码可靠性排序见 ETSI TS 138 212 V19.4.0 的第 5.1 和 5.3.1.2 节。[列表译码](https://doi.org/10.1109/TIT.2015.2410251) · [5G 标准](https://www.etsi.org/deliver/etsi_ts/138200_138299/138212/19.04.00_60/ts_138212v190400p.pdf)
- SC-ML、RecursCode 和 Number-polar 的预编码矩阵分别见 V. Miloslavskaya 等人发表于 IEEE Trans. Commun. 的三项工作（2020、2021、2024）。[SC-ML](https://doi.org/10.1109/TCOMM.2020.3014946) · [RecursCode](https://doi.org/10.1109/TCOMM.2021.3111625) · [Number-polar](https://doi.org/10.1109/TCOMM.2023.3331532)

[较早的五个构造案例](docs/LEGACY.md) 另含一个 $[128,42]$ 码。第三方代码与数据来源见[署名说明](NOTICE.md)。

## AI 声明

本仓库由 OpenAI 的 GPT-6 根据作者本地实验代码、构造记录和论文整理而成，旨在方便读者或潜在的审稿人理解与复现该论文，并基于此进一步研究。
