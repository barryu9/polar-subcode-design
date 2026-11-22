# 仿真码配置

[English](README.md) · 简体中文

运行 `python -m reproduction list`，可按码长、图例和译码列表大小查找图 5–10 中的码。[Proposed 配置索引](proposed/manifest.json)列出了表 II 的八种构造和消融码 Proposed-NP。[对比码索引](reference/manifest.json)给出了各条参考曲线的构造方法和输入。

每份 `*.profile.json` 记录固定信息集设计开始时选定的信息位置，配对的 `*.code.json` 还包含冻结表达式及该曲线的编码参数。从头运行构造脚本时，程序也会执行选取信息集的搜索。Proposed-NP 已含前缀保护之前得到的动态冻结表达式。

| 字段或约定 | 含义 |
|---|---|
| `N`、`profile`、`K` | 依次为码长、按 SC 顺序递增且从零开始的信息位置、信息位置数。对 CRC 曲线，`payload` 为载荷长度，`K` 包含 CRC 位。 |
| `dynamic_expressions["j"] = [i, k]` | 冻结位满足二元域上的 $u_j=u_i+u_k$。来源位置均早于 $j$，省略的冻结位取零。若文件提供 `columns[j]`，其整数的第 $i$ 位表示位置 $i$ 是否参与该表达式。 |
| 极化与 PAC 编码 | 极化变换按自然顺序取 $x=uF^{\otimes m}$。PAC 编码先将载荷置于 $v$，再用按延迟递增排列的抽头计算 $u=v*g$。规范编码以 $u$ 的信息位作为载荷坐标，生成同一码。 |
| `crc.polynomial` | 包含最高次项。载荷位按 SC 顺序进入寄存器。译码器在母码的全部信息位置分支，再对最终列表做 CRC 校验。全部候选都未通过时记为帧错误。 |

用 `polar_subcodes.codec.load_codec(path)` 读取完整配置后可编码或进行 SCL 译码。精确枚举实际发送码的重量谱时，可用 `codec.as_subcode()` 将 CRC 约束一并纳入。

SC-ML、RecursCode 和 Number-polar 从文献的 SC 预编码矩阵 $Z$ 重建。[矩阵输入](../reproduction/reference_inputs)以十六进制整数逐行储存，最低位对应位置零，并附原始文件名和 SHA-256。程序将 $Z$ 行化简为因果形式。PAC、CRC 和 eBCH 对比码则由各自的设计参数生成。
