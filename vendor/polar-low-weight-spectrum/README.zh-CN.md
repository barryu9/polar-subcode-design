# 极化码低重量谱计算内核

[English](README.md) · 简体中文

本目录提供论文中精确枚举低重量谱所用的 Python 和 C++ 内核。其软件作者和版本见 [CITATION.cff](CITATION.cff)，使用条款见 [MIT 许可证](LICENSE)。

在仓库根目录执行 `python tools/build_native.py` 可编译原生库。[谱枚举接口](../../polar_subcodes/spectrum/backend.py)读取码的校验矩阵，按给定的截止重量调用内核。图 5 的完整计算示例见[仓库指南](../../README.zh-CN.md)。
