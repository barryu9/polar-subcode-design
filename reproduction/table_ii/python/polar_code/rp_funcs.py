"""Fitted Gaussian-approximation recursion for Table II channel ordering and scores."""
import numpy as np
from scipy.special import erf
import math
from typing import NamedTuple

# ======================================================================================
# 等同于: phi.m
# ======================================================================================
def phi(x):
    """
    GA构造中Phi函数的快速运算

    参数:
        x (float): 输入值

    返回:
        float: 根据不同区间计算得到的函数值
    """
    # 处理 0 <= x <= 10 区间的情况
    if 0 <= x <= 10:
        return math.exp(-0.4527 * x ** 0.86 + 0.0218)
    else:
        # 处理 x > 10 的情况，特别处理 x = 0 的边界情况避免除零错误
        if x == 0:
            return float('inf')
        return math.sqrt(math.pi / x) * math.exp(-x / 4) * (1 - 10 / (7 * x))



# ======================================================================================
# phi_inverse.m 所需的辅助函数 (原MATLAB文件中未提供)
# ======================================================================================
def derivative_phi(x):
    """
    计算phi函数的导数值，用于phi_inverse函数中的牛顿法迭代。

    参数:
        x (float): 输入值，用于计算phi函数在该点的导数

    返回:
        float: phi函数在x点的导数值

    说明:
        当x在[0,10]区间内时，使用近似公式计算导数；
        当x>10时，使用渐近展开式的导数公式计算。
    """
    if 0 <= x <= 10:
        return (-0.4527 * 0.86 * x ** (-0.14)) * math.exp(-0.4527 * x ** 0.86 + 0.0218)
    else:
        # 计算y = sqrt(pi/x) * exp(-x/4) * (1 - 10/(7x))的导数
        # 原函数分解为两部分：exp(-x/4) 和 (sqrt(pi)*x**-0.5 - 10*sqrt(pi)/(7*x**1.5))
        # 使用乘积法则：(uv)' = u'v + uv'
        e_neg_x_4 = math.exp(-x / 4)
        sqrt_pi = math.sqrt(math.pi)

        term1 = -0.25 * e_neg_x_4 * (sqrt_pi * x ** -0.5 - 10 * sqrt_pi / (7 * x ** 1.5))
        term2 = e_neg_x_4 * (-0.5 * sqrt_pi * x ** -1.5 + 15 * sqrt_pi / (7 * x ** 2.5))

        return term1 + term2



def phi_inverse(y):
    """
    计算函数phi的反函数在给定值y处的近似解。

    该函数根据输入y的范围选择不同的计算策略：
    - 当y在闭式近似有效区间[0.0388, 1.0221]内时，使用闭式表达式直接计算；
    - 否则使用牛顿迭代法进行数值求解。

    参数:
        y (float): 要计算反函数的目标值

    返回:
        tuple: 包含两个元素的元组
            - x (float): 反函数phi^{-1}(y)的近似解
            - num_iter (int): 牛顿法迭代次数（当使用闭式解时为0）
    """
    num_iter = 0
    # 闭式近似解有效的范围
    if 0.0388 <= y <= 1.0221:
        # log(y)可以为负数，这是正常的
        val = (0.0218 - math.log(y)) / 0.4527
        # 确保开方的参数不为负数
        x = val ** (1 / 0.86) if val >= 0 else 0
    else:
        # 对其他值使用牛顿法
        x0 = 0.0388  # 初始猜测值
        # 避免除以接近零的导数
        deriv = derivative_phi(x0)
        if abs(deriv) < 1e-9: deriv = -1e-9
        x1 = x0 - (phi(x0) - y) / deriv

        delta = abs(x1 - x0)
        epsilon = 1e-3

        while delta >= epsilon:
            num_iter += 1
            x0 = x1
            deriv = derivative_phi(x0)
            if abs(deriv) < 1e-9: deriv = -1e-9  # 避免除零
            x1 = x1 - (phi(x1) - y) / deriv

            # 对于较大的x值，放宽收敛阈值
            if x1 > 1e2:
                epsilon = 10

            delta = abs(x1 - x0)
            if num_iter > 100:  # 防止无限循环
                break
        x = x1
    return x, num_iter



# ======================================================================================
# 用于复现MATLAB bitrevorder功能的辅助函数
# ======================================================================================
def bit_reversal_permutation(x):
    """
    对输入数组x执行比特翻转置换，功能类似于MATLAB的bitrevorder。

    参数:
        x: 输入数组，长度应为2的幂次

    返回值:
        numpy.ndarray: 比特翻转置换后的数组，与输入数组长度相同

    示例:
        对于长度为8的数组，索引映射关系为:
        0(000) -> 0(000)
        1(001) -> 4(100)
        2(010) -> 2(010)
        3(011) -> 6(110)
        4(100) -> 1(001)
        5(101) -> 5(101)
        6(110) -> 3(011)
        7(111) -> 7(111)
    """
    N = len(x)
    # 计算表示N个元素所需最少比特数
    num_bits = N.bit_length() - 1
    res = np.empty_like(x)
    for i in range(N):
        # 将索引i的二进制位翻转，作为新数组的索引
        rev_i = int(format(i, f'0{num_bits}b')[::-1], 2)
        res[rev_i] = x[i]
    return res




def GA(sigma, N):
    """
    函数 (高斯近似) 的Python实现。

    参数:
        sigma: 高斯分布的标准差参数
        N: 输出数组的长度

    返回值:
        tuple: 包含以下元素：
            - u: 经过高斯近似计算和位反转置换后的结果数组
            - ave_iter: 平均迭代次数
    """
    u = np.zeros(N)
    u[0] = 2 / sigma ** 2
    num_for_runs = 0
    sum_iter = 0

    # 外层循环：按log2(N)次迭代进行处理
    for i in range(1, int(np.log2(N)) + 1):
        j = 2 ** (i - 1)
        # 内层循环：对当前层级的元素进行高斯近似计算
        for k in range(j):
            tmp = u[k]
            u[k], num_iter = phi_inverse(1 - (1 - phi(tmp)) ** 2)
            sum_iter += num_iter
            u[k + j] = 2 * tmp
            num_for_runs += 1

    # 对结果数组进行位反转置换
    u = bit_reversal_permutation(u)
    ave_iter = sum_iter / num_for_runs if num_for_runs > 0 else 0
    return u, ave_iter



def get_RM_score(N):
    """
    计算RM分数 (索引的二进制表示的汉明权重)。

    参数:
        N (int): 索引范围，计算从0到N-1每个索引的RM分数

    返回:
        numpy.ndarray: 包含每个索引对应RM分数的一维数组，长度为N
    """
    # 将每个索引转换为二进制并计算'1'的个数
    rm_score = [bin(i).count('1') for i in range(N)]
    return np.array(rm_score)



class RP_Params(NamedTuple):
    """
    存储极化码可靠性剖面参数的命名元组。

    属性:
        info_bits_indices (np.ndarray): 信息位索引数组。
        frozen_bits_indices (np.ndarray): 冻结位索引数组。
        info_bits_mask (np.ndarray): 信息位掩码数组，1表示信息位，0表示冻结位。
        frozen_bits_mask (np.ndarray): 冻结位掩码数组，1表示冻结位，0表示信息位。
        channel_params (np.ndarray): 每个信道的质量参数。
        pe (np.ndarray): 各信道的错误概率估计。
        rw (np.ndarray): 每个位置对应的RM得分或权重。
    """
    info_bits_indices: np.ndarray
    frozen_bits_indices: np.ndarray
    info_bits_mask: np.ndarray
    frozen_bits_mask: np.ndarray
    channel_params: np.ndarray
    pe: np.ndarray
    rw: np.ndarray


# ======================================================================================
# 等同于: rp_GA.m
# ======================================================================================
def rp_GA(N, k, dsnr_dB):
    """
    rp_GA.m 函数的Python实现。

    该函数实现了极化码的GA（Gaussian Approximation）算法，用于确定信息位和冻结位的分配。

    参数:
        N (int): 码长，必须是2的幂次
        k (int): 信息位长度
        dsnr_dB (float): 设计SNR，以dB为单位

    返回:
        RP_Params: 包含以下字段的对象：
            - info_bits_indices: 信息位索引数组，按升序排列
            - frozen_bits_indices: 冻结位索引数组
            - info_bits_mask: 信息位掩码数组，1表示信息位，0表示冻结位
            - frozen_bits_mask: 冻结位掩码数组，1表示冻结位，0表示信息位
            - channel_params: 信道参数数组
            - pe: 误码率估计数组
            - rw: RM得分数组
    """
    R = k / N

    # 计算噪声标准差
    sigma = 1 / np.sqrt(2 * R) * 10 ** (-dsnr_dB / 20)

    # 使用GA算法获取信道参数
    channel_params, _ = GA(sigma, N)

    # 按可靠性对信道进行排序 (降序)
    channels_ordered = np.argsort(channel_params)[::-1]

    # 选择最可靠的k个信道作为信息位信道，并按索引升序排列
    info_bits_indices = np.sort(channels_ordered[:k])

    # 创建信息位掩码
    info_bits_mask = np.zeros(N, dtype=int)
    info_bits_mask[info_bits_indices] = 1

    # 创建冻结位掩码和索引
    frozen_bits_mask = 1 - info_bits_mask
    frozen_bits_indices = np.where(frozen_bits_mask == 1)[0]

    # 计算误码率估计
    pe = 0.5 - 0.5 * erf(np.sqrt(channel_params) / 2)

    # 获取RM得分
    rw = get_RM_score(N)

    rp = RP_Params(
        info_bits_indices=info_bits_indices,
        frozen_bits_indices=frozen_bits_indices,
        info_bits_mask=info_bits_mask,
        frozen_bits_mask=frozen_bits_mask,
        channel_params=channel_params,
        pe=pe,
        rw=rw
    )
    return rp
