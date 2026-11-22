# Reproduction checks for manuscript r112 · 复现核查

The following checks connect the released code to the manuscript and the recorded simulations. They can be run separately from the repository root.

| Check | What was compared | Reader command |
|---|---|---|
| Algorithms 1–4 and codec routines | 76 automated checks, including exhaustive small-code comparisons and CRC-aided decoding with nonzero payloads | `python main.py verify-theory` |
| Examples 1–9 | The numbered examples against the paper calculations, including the two length-256 construction examples | `python -m paper_examples all` |
| Theorems 1–3, Propositions 1–3, Corollaries 1–2 | Eight numerical checks with exact finite-case calculations | `python -m theory.check_results` |
| Figs. 5–10 | All 30 curve configurations, including the payload and CRC used for terminal selection | `python -m reproduction check` |
| Table II | Eight rows of GA reliability scores using the paper's common design SNR | `python -m reproduction.table_ii_values` |
| Fig. 5 spectrum | The native counter and explicit listing both give 2,048 weight-12 words and no positive weight below 12 for the Proposed code | See the two `main.py enumerate` commands in the [README](../README.md) |

The 21 comparison codes were rebuilt from the construction parameters or precoding matrices. Both the NumPy and C++ decoders recovered a nonzero payload for each of the 30 curve configurations on noiseless input. The native build and the full Fig. 7 Proposed construction were independently tested on Windows and Ubuntu 24.04 with CPython 3.12. The regenerated information positions and frozen expressions agreed on both systems.

All eight Proposed construction runs in Table II completed. Their candidate archives reproduced all nine Proposed curves, including the intermediate Proposed-NP code in Fig. 10. Both length-1024 continuations used the newly constructed main-code checkpoint. For every curve, both exported JSONs matched the distributed profile and code configuration.

表 II 的八次构造均已从头运行，得到九套 Proposed 配置。图 10 的 Proposed-NP 取自主码候选库内尚未进行前缀保护的阶段，Proposed-LM 和 Proposed-D32 则从新生成的主码继续构造。九套信息位置及冻结表达式均与原实验一致。21 个对比码也逐一由构造参数或文献预编码矩阵重建，并核对了相应的载荷与校验参数。读者可用 [构造入口](../reproduction/) 重新生成结果，用 [原始测量表](../results/paper_r112/README.md) 核对 FER 曲线。表 II 使用原实验的串行计时。
