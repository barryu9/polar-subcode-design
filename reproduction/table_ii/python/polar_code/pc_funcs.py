"""Initialize the polar transforms and SCL decoder tables used in the timed constructions."""
import numpy as np
from typing import NamedTuple


def get_crc_objective(crc_length):
    """
    根据输入的CRC长度返回CRC生成多项式。

    Args:
        crc_length (int): CRC编码的长度。

    Returns:
        np.ndarray: CRC生成多项式的系数数组。

    Raises:
        ValueError: 如果crc_length不受支持。
    """
    if crc_length == 4:
        return np.array([1, 0, 0, 1, 1], dtype=int)
    elif crc_length == 6:
        return np.array([1, 1, 0, 0, 0, 0, 1], dtype=int)
    elif crc_length == 8:
        return np.array([1, 1, 1, 1, 1, 1, 0, 0, 1], dtype=int)
    elif crc_length == 10:
        return np.array([1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1], dtype=int)
    elif crc_length == 11:
        return np.array([1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1], dtype=int)
    elif crc_length == 12:
        return np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1], dtype=int)
    elif crc_length == 16:
        return np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=int)
    elif crc_length == 24:
        return np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1], dtype=int)
    else:
        raise ValueError(f"Unsupported CRC length: {crc_length}. Program terminates")


def get_pi_seq(N):
    """
    生成用于 Action 2 随机预编码的确定性伪随机序列
    基于 pi 的有理数逼近 (104348/33215) 的二进制展开。

    Args:
        N (int): 码长。

    Returns:
        np.ndarray: 预计算的 pi 伪随机序列。
    """
    MAX_BITS = N * N
    seq = np.zeros(MAX_BITS, dtype=np.int8)
    seq[0] = 1
    seq[1] = 1  # 整数部分 3 的二进制表示 '11'
    rem = 4703  # 余数: 104348 % 33215
    for i in range(2, MAX_BITS):
        rem *= 2
        seq[i] = rem // 33215
        rem = rem % 33215
    return seq

def crc_generator_matrix(g, K):
    """
    根据给定的生成多项式g和信息长度K，计算CRC生成矩阵G和校验矩阵H。

    Args:
        g (np.ndarray): 生成多项式系数。
        K (int): 信息序列的长度。

    Returns:
        tuple[np.ndarray, np.ndarray]: 返回生成矩阵G和校验矩阵H。
    """
    r = len(g) - 1
    N = K + r

    G = np.zeros((K, N), dtype=int)

    # 将g放置在G的最后一行
    G[-1, K - 1:] = g

    # 逐行生成G的其余部分（通过循环移位）
    for i in range(K - 2, -1, -1):
        G[i, :] = np.roll(G[i + 1, :], -1)

    # 通过行变换（类高斯消元）将G转换为系统形式 [I_K | P]
    # 注意：MATLAB代码中的j同时用作列索引和行索引，这里进行了忠实转换。
    for i in range(K - 2, -1, -1):
        # 在MATLAB中, j的范围是 N-r(=K) 到 i+1
        for j in range(K - 1, i, -1):
            if G[i, j] == 1:
                G[i, :] = np.mod(G[i, :] + G[j, :], 2)

    # 构建校验矩阵 H = [P.T | I_r]
    H = np.zeros((r, N), dtype=int)
    H[:, :K] = G[:, K:].T
    H[:, K:] = np.eye(r, dtype=int)

    return G, H


def get_G_N(F_N, N):
    """
    使用Kronecker积生成Polar码的生成矩阵G_N。

    Args:
        F_N (np.ndarray): 极化核 (例如 [[1, 0], [1, 1]])。
        N (int): 码长，必须是2的幂。

    Returns:
        np.ndarray: Polar码生成矩阵G_N。
    """
    # 初始化生成矩阵
    G_N = np.zeros((N, N), dtype=int)
    G_N[0:2, 0:2] = F_N

    # 计算迭代次数
    n = int(np.log2(N))

    # 通过Kronecker积逐步构造生成矩阵
    for i in range(2, n + 1):
        G_prev = G_N[0:2 ** (i - 1), 0:2 ** (i - 1)]
        G_N[0:2 ** i, 0:2 ** i] = np.kron(G_prev, F_N)

    return G_N



def get_llr_layer(N):
    """
    计算SCL译码中用于LLR（对数似然比）存储的层索引。

    Args:
        N (int): 码长。

    Returns:
        np.ndarray: LLR层索引向量。
    """
    # 初始化层索引向量，长度为N，数据类型为整型
    layer_vec = np.zeros(N, dtype=int)

    # 遍历从1到N-1的每个位置，计算其对应的层索引
    for phi in range(1, N):
        psi = phi
        layer = 0

        # 通过不断除以2来计算phi的二进制表示中末尾连续0的个数
        while psi % 2 == 0:
            psi //= 2
            layer += 1

        # 将计算得到的层索引存储到对应位置
        layer_vec[phi] = layer

    return layer_vec



def get_bit_layer(N):
    """
    计算SCL译码中用于比特存储的层索引。

    Args:
        N (int): 码长。

    Returns:
        np.ndarray: 比特层索引向量。
    """
    # 初始化层索引向量
    layer_vec = np.zeros(N, dtype=int)

    # 遍历每个比特位置，计算其对应的存储层索引
    for phi in range(N):
        # 计算当前比特位置对应的父节点位置
        psi = phi // 2
        layer = 0

        # 通过不断除以2来计算二进制表示中末尾连续1的个数
        while psi % 2 == 1:
            psi //= 2
            layer += 1

        # 将计算得到的层索引存入向量
        layer_vec[phi] = layer

    return layer_vec



class PC_Params(NamedTuple):
    """
    存储Polar码相关参数的命名元组。

    Attributes:
        N (int): 码长。
        K (int): 信息序列长度。
        R (float): 码率，即 K / N。
        crc_length (int): CRC编码长度（若不使用CRC则为0）。
        F_N (np.ndarray): 极化核矩阵，例如 [[1, 0], [1, 1]]。
        n (int): 码长N的以2为底的对数，即 log2(N)。
        G_N (np.ndarray): Polar码的整体生成矩阵。
        lambda_offset (np.ndarray): SC译码中使用的偏移量数组。
        llr_layer_vec (np.ndarray): LLR计算层结构向量。
        bit_layer_vec (np.ndarray): 比特重建层结构向量。
        H_crc (np.ndarray): CRC校验矩阵。
        crc_parity_check (np.ndarray): 校验子生成矩阵（用于CRC校验位提取）。
    """
    N: int
    K: int
    R: int
    crc_length : int
    F_N: np.ndarray
    n : int
    G_N: np.ndarray
    lambda_offset: np.ndarray
    llr_layer_vec: np.ndarray
    bit_layer_vec: np.ndarray
    H_crc: np.ndarray
    crc_parity_check: np.ndarray
    pi_seq: np.ndarray


def pc_init(N, K, crc_length, F_N):
    """
    设置并初始化Polar码的相关参数。

    Args:
        N (int): 码长。
        K (int): 信息序列长度。
        crc_length (int): CRC编码长度（如果不需要则设为0）。
        F_N (np.ndarray): 极化核，例如 np.array([[1, 0], [1, 1]])。

    Returns:
        PC_Params: 包含所有Polar码参数的命名元组对象。
    """
    R = K/N
    n = int(np.log2(N))
    G_N = get_G_N(F_N, N)
    lambda_offset = 2 ** np.arange(n + 1)
    llr_layer_vec = get_llr_layer(N)
    bit_layer_vec = get_bit_layer(N)

    # 如果启用CRC编码，则生成相应的CRC校验矩阵
    if crc_length > 0:
        g_crc = get_crc_objective(crc_length)
        G_crc, H_crc = crc_generator_matrix(g_crc, K)
        # 提取校验位部分 P.T
        crc_parity_check = G_crc[:, K:].T
    else:
        crc_parity_check = np.array([])
        H_crc = np.array([])

    # 保留构造计时所用的完整译码器初始化过程。
    pi_seq = get_pi_seq(N)

    pc_params = PC_Params(
        N=N, K=K, R=R, crc_length=crc_length, F_N=F_N, n=n, G_N=G_N,
        lambda_offset=lambda_offset, llr_layer_vec=llr_layer_vec,
        bit_layer_vec=bit_layer_vec, H_crc=H_crc, crc_parity_check=crc_parity_check,
        pi_seq=pi_seq
    )

    return pc_params
