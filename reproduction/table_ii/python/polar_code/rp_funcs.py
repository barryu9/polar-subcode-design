import numpy as np
from scipy.special import erf
import math
from typing import NamedTuple
import copy

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



# ======================================================================================
# 等同于: get_BEC_ZWi.m
# ======================================================================================
def get_BEC_ZWi(N, ZW):
    """
    get_BEC_ZWi.m 函数的Python实现。

    该函数计算BEC（二进制擦除信道）下的权重信息，通过递归方式构建不同层次的权重分布。

    参数:
        N (int): 码长，通常为2的幂次方
        ZW (float): 初始权重值，表示信道擦除概率

    返回:
        numpy.ndarray: 长度为N的数组，包含经过变换后的权重信息
    """
    ZWi = np.zeros(N)
    ZWi[0] = ZW

    # 递归计算不同层次的权重信息
    for level in range(1, int(np.log2(N)) + 1):
        B = 2 ** level
        for j in range(B // 2):
            Z_tmp = ZWi[j]
            ZWi[j] = 2 * Z_tmp - Z_tmp ** 2  # 使用上界
            ZWi[B // 2 + j] = Z_tmp ** 2

    ZWi = bit_reversal_permutation(ZWi)  # 执行比特翻转排序
    return ZWi


# ======================================================================================
# 用于计算里德-穆勒(RM)分数的辅助函数
# ======================================================================================
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



from typing import NamedTuple
import numpy as np
from scipy.special import erf


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


# ======================================================================================
# 等同于: rp_BEC.m
# ======================================================================================
def rp_BEC(N, k, dsnr_dB):
    """
    计算BEC信道下的极化码信息位和冻结位分配。

    该函数实现了MATLAB中rp_BEC.m函数的功能，用于确定在给定信道条件下，
    哪些比特通道应该作为信息位，哪些应该作为冻结位。

    参数:
        N (int): 码长，必须是2的幂次
        k (int): 信息位长度
        dsnr_dB (float): 设计SNR，以dB为单位

    返回:
        RP_Params: 包含以下字段的对象：
            - info_bits_indices: 信息位索引数组，按升序排列
            - frozen_bits_indices: 冻结位索引数组
            - info_bits_mask: 信息位掩码，1表示信息位，0表示冻结位
            - frozen_bits_mask: 冻结位掩码，1表示冻结位，0表示信息位
            - channel_params: 各比特通道的擦除概率参数
            - pe: 各比特通道的错误概率，与channel_params相同
            - rw: RM得分数组
    """
    dsnr = 10 ** (dsnr_dB / 10)
    ZW = np.exp(-dsnr)  # AWGN信道到BEC信道的Bhattacharyya参数

    channel_params = get_BEC_ZWi(N, ZW)  # 信道擦除参数

    # 按擦除概率排序 (升序，最好的信道具有最低的ZW值)
    channels_ordered = np.argsort(channel_params)

    info_bits_indices = np.sort(channels_ordered[:k])

    info_bits_mask = np.zeros(N, dtype=int)
    info_bits_mask[info_bits_indices] = 1

    frozen_bits_mask = 1 - info_bits_mask
    frozen_bits_indices = np.where(frozen_bits_mask == 1)[0]

    pe = channel_params
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


# ======================================================================================
# 等同于: rp_RM_Polar.m
# ======================================================================================
def rp_RM_Polar(N, k, dsnr_dB):
    """
    rp_RM_Polar.m 函数的Python实现。

    该函数用于构造 Polar 码的信息位和冻结位分配方案。结合了 RM（Reed-Muller）分数与高斯近似（GA）方法，
    根据给定的码长、信息位数以及信噪比，确定哪些位置作为信息位传输数据，其余则设为冻结位。

    参数:
        N (int): 码长，必须是2的幂次。
        k (int): 信息位数量，满足 0 < k <= N。
        dsnr_dB (float): 设计信噪比，单位为 dB。

    返回:
        RP_Params: 包含以下字段的对象：
            - info_bits_indices: 信息位索引数组，按升序排列。
            - frozen_bits_indices: 冻结位索引数组。
            - info_bits_mask: 长度为N的掩码数组，信息位位置标记为1。
            - frozen_bits_mask: 长度为N的掩码数组，冻结位位置标记为1。
            - channel_params: 每个信道的参数（来自GA算法）。
            - pe: 各信道的错误概率估计。
            - rw: 每个位置对应的 RM 分数。
    """
    R = k / N

    # 获取每个位置的 Reed-Muller 分数
    RM_score = get_RM_score(N)

    # 计算噪声标准差 sigma
    sigma = 1 / np.sqrt(2 * R) * 10 ** (-dsnr_dB / 20)

    # 使用高斯近似方法获取每个信道的质量参数
    channel_params, _ = GA(sigma, N)

    # 对 RM 分数从大到小排序以选择前 k 个最优位置
    RM_score_ordered = np.sort(RM_score)[::-1]
    # 取第 k 大的 RM 分数作为阈值（注意：Python 是从0开始索引）
    RM_score_k = RM_score_ordered[k - 1]

    # 创建临时副本用于后续处理
    channel_params_temp = channel_params.copy()

    # 将 RM 分数低于阈值的位置对应的信道质量设置为极低（淘汰）
    channel_params_temp[RM_score < RM_score_k] = -1.0

    # 将 RM 分数高于阈值的位置对应的信道质量提升（优先保留）
    channel_params_temp[RM_score > RM_score_k] = np.max(channel_params) + 1.0

    # 综合考虑 RM 分数和信道参数进行排序，解决 RM 相同情况下的冲突
    channels_ordered = np.argsort(channel_params_temp)[::-1]

    # 提取前 k 个最佳信道索引，并保持升序排列
    info_bits_indices = np.sort(channels_ordered[:k])

    # 构建信息位掩码
    info_bits_mask = np.zeros(N, dtype=int)
    info_bits_mask[info_bits_indices] = 1

    # 构建冻结位掩码及索引
    frozen_bits_mask = 1 - info_bits_mask
    frozen_bits_indices = np.where(frozen_bits_mask == 1)[0]

    # 填充返回结果字典
    pe = 0.5 - 0.5 * erf(np.sqrt(channel_params) / 2)
    rw = RM_score

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


# ======================================================================================
# 等同于: rp_DIY.m
# ======================================================================================
def rp_DIY(N, info_bits_indices):
    """
    rp_DIY.m 函数的Python实现。
    基于用户提供的一组信息位索引来创建可靠性剖面。

    参数:
        N (int): 码长，表示整个编码序列的长度
        info_bits_indices (list or array): 信息位索引列表，指定哪些位置是信息位

    返回:
        RP_Params: 可靠性剖面参数对象，包含信息位和冻结位的相关信息
    """

    # 确保索引是排好序的numpy数组
    info_bits_indices = np.sort(np.array(info_bits_indices))

    # 创建信息位掩码，标记信息位位置为1，其余为0
    info_bits_mask = np.zeros(N, dtype=int)
    info_bits_mask[info_bits_indices] = 1

    # 根据信息位掩码计算冻结位掩码和索引
    frozen_bits_mask = 1 - info_bits_mask
    frozen_bits_indices = np.where(frozen_bits_mask == 1)[0]

    # 获取Reed-Muller得分作为权重
    rw = get_RM_score(N)

    # 构建并返回可靠性剖面参数对象
    rp = RP_Params(
        info_bits_indices=info_bits_indices,
        frozen_bits_indices=frozen_bits_indices,
        info_bits_mask=info_bits_mask,
        frozen_bits_mask=frozen_bits_mask,
        channel_params=[],
        pe=[],
        rw=rw
    )

    return rp


# ======================================================================================
# 等同于: rp_modify.m
# ======================================================================================
def rp_modify(rp: RP_Params, new_info_bits_indices, indices_to_be_replaced):
    """
    rp_modify.m 函数的Python实现。
    修改一个已有的可靠性剖面(rp)。
    此函数不验证更改；信息位的数量可能会发生变化。

    参数:
        rp (RP_Params): 原始可靠性剖面对象，应包含以下属性：
            - info_bits_mask: 信息位掩码数组
            - info_bits_indices: 信息位索引数组
            - frozen_bits_mask: 冻结位掩码数组
            - frozen_bits_indices: 冻结位索引数组
            - channel_params: 信道参数
            - pe: 错误概率相关参数
            - rw: 可靠性权重参数
        new_info_bits_indices (array-like): 需要设置为信息位的索引数组
        indices_to_be_replaced (array-like): 需要设置为冻结位的索引数组

    返回:
        RP_Params: 修改后的可靠性剖面对象，包含更新后的所有索引和掩码信息
    """

    # 最好使用副本以避免副作用
    info_bits_mask = rp.info_bits_mask.copy()

    # 确保输入索引是可迭代的 (列表或数组)
    new_info_bits_indices = np.atleast_1d(new_info_bits_indices)
    indices_to_be_replaced = np.atleast_1d(indices_to_be_replaced)

    # 根据输入参数更新信息位掩码
    if new_info_bits_indices.size > 0:
        info_bits_mask[new_info_bits_indices] = 1
    if indices_to_be_replaced.size > 0:
        info_bits_mask[indices_to_be_replaced] = 0

    # 重新计算信息位和冻结位的索引及掩码
    info_bits_indices = np.where(info_bits_mask == 1)[0]
    frozen_bits_mask = 1 - info_bits_mask
    frozen_bits_indices = np.where(frozen_bits_mask == 1)[0]

    # 更新修改后可靠性剖面的所有字段
    modified_rp = RP_Params(
        info_bits_indices=info_bits_indices,
        frozen_bits_indices=frozen_bits_indices,
        info_bits_mask=info_bits_mask,
        frozen_bits_mask=frozen_bits_mask,
        channel_params=rp.channel_params,
        pe=rp.pe,
        rw=rp.rw
    )

    return modified_rp




# ======================================================================================
# 示例用法
# ======================================================================================
if __name__ == '__main__':
    N = 64  # 码长
    k = 32  # 信息位数量
    dsnr_dB = 2.0  # 设计信噪比 (dB)

    print(f"正在为 N={N}, k={k}, dsnr_dB={dsnr_dB} 生成可靠性剖面")

    # 1. 使用高斯近似法 (GA)
    rp1 = rp_GA(N, k, dsnr_dB)
    print("\n--- rp_GA (高斯近似法) ---")
    print("前10个信息位索引:", rp1.info_bits_indices)
    print("信息位总数:", len(rp1.info_bits_indices))

    # 2. 使用二进制擦除信道法 (BEC)
    rp2 = rp_BEC(N, k, dsnr_dB)
    print("\n--- rp_BEC (二进制擦除信道法) ---")
    print("前10个信息位索引:", rp2.info_bits_indices)
    print("信息位总数:", len(rp2.info_bits_indices))

    # 3. 使用混合RM-Polar法
    rp3 = rp_RM_Polar(N, k, dsnr_dB)
    print("\n--- rp_RM_Polar (混合RM-Polar法) ---")
    print("前10个信息位索引:", rp3.info_bits_indices)
    print("信息位总数:", len(rp3.info_bits_indices))

    # 4. 使用自定义的信息位索引集合
    custom_indices = list(range(k))  # 一个简单的例子：前k个信道
    rp4 = rp_DIY(N, custom_indices)
    print("\n--- rp_DIY (自定义索引法) ---")
    print("前10个信息位索引:", rp4.info_bits_indices)

    # 5. 修改一个已有的剖面
    # 假设我们想在rp_GA的结果中，将信息位索引'63'替换为'70'
    rp_original = rp_GA(N, k, dsnr_dB)
    print(f"\n--- rp_modify (修改剖面) ---")
    print("修改前，索引 63 是信息位吗?", 63 in rp_original.info_bits_indices)
    print("修改前，索引 18 是信息位吗?", 18 in rp_original.info_bits_indices)

    rp_modificado = rp_modify(rp_original, new_info_bits_indices=[18], indices_to_be_replaced=[63])

    print("修改后:")
    print("修改后，索引 63 是信息位吗?", 63 in rp_modificado.info_bits_indices)
    print("修改后，索引 18 是信息位吗?", 18 in rp_modificado.info_bits_indices)
    print("信息位总数:", len(rp_modificado.info_bits_indices))