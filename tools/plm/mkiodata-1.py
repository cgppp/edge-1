# -*- coding: utf-8 -*-
# 指定文件编码为UTF-8，支持中文/特殊字符

# 导入系统模块，用于接收命令行参数
import sys

# 从numpy库导入：
# array = 用于把Python列表转成numpy数组
# int32 = 指定数据类型为32位整型（节省显存、模型训练通用）
from numpy import array as np_array, int32 as np_int32

# ==================== 关键模块导入 ====================
# 导入 Roberta 模型专用的 batch_padder（批量数据填充器）
# 作用：把长短不一的句子统一长度，填充<pad>，生成模型可接受的批量数据
from utils.fmt.plm.roberta.dual import batch_padder

# 导入HDF5文件读写工具（高效存储训练数据的二进制格式）
from utils.h5serial import h5File

# ==================== 配置文件导入 ====================
# 导入训练超参数配置（batch大小、最大token数等）
from cnfg.ihyp import *
# 导入Roberta词表配置：pad_id = 填充符的ID（一般是1或0）
from cnfg.vocab.plm.roberta import pad_id

# ==================== 核心函数：数据预处理+打包 ====================
def handle(
    finput,         # 输入文件路径（源句子：问题/文本）
    ftarget,        # 目标文件路径（标签/答案）
    frs,            # 输出HDF5文件路径（最终打包结果）
    minbsize=1,     # 最小batch大小（多GPU时用）
    expand_for_mulgpu=True,  # 是否为多GPU扩展batch（自动翻倍）
    bsize=max_sentences_gpu, # 单GPU最大句子数（来自配置）
    maxpad=max_pad_tokens_sentence,  # 单句最大填充长度
    maxpart=normal_tokens_vs_pad_tokens,  # 真实token vs 填充token比例
    maxtoken=max_tokens_gpu,  # 单GPU最大token总数
    minfreq=False,    # 词频过滤（这里未启用）
    vsize=False,      # 词表大小（这里未启用）
    pad_id=pad_id,    # 填充符ID（来自Roberta配置）
    **kwargs          # 其他扩展参数
):
    """
    功能：读取文本语料 → 批量填充 → 转成numpy数组 → 写入HDF5文件
    输出：src(输入) + tgt(目标) 两组数据，按batch存储
    """

    # ==================== 多GPU自动扩容 ====================
    # 如果开启多GPU模式，batch大小和最大token数 × minbsize（GPU数量）
    if expand_for_mulgpu:
        _bsize = bsize * minbsize      # 实际使用的batch大小
        _maxtoken = maxtoken * minbsize# 实际使用的最大token数
    else:
        _bsize = bsize
        _maxtoken = maxtoken

    # ==================== 创建HDF5输出文件 ====================
    # h5File = 安全写入HDF5文件（训练数据标准格式）
    # h5_fileargs = 压缩/分块等配置（来自ihyp.py）
    with h5File(frs, "w", **h5_fileargs) as rsf:
        
        # 创建两个组：src=输入数据，tgt=目标标签（类似文件夹）
        src_grp = rsf.create_group("src")
        tgt_grp = rsf.create_group("tgt")
        
        curd = 0  # 记录当前处理到第几个batch

        # ==================== 核心：逐批读取+填充+保存 ====================
        # batch_padder：读取输入&目标文件 → 动态生成合适大小的batch
        # 返回：i_d(输入batch), td(目标batch)
        for i_d, td in batch_padder(
            finput, ftarget,
            _bsize, maxpad, maxpart, _maxtoken,
            minbsize, pad_id=pad_id
        ):
            # 把Python列表 → numpy int32数组（模型训练标准格式）
            rid = np_array(i_d, dtype=np_int32)
            rtd = np_array(td, dtype=np_int32)

            # batch编号：0,1,2,3... 转字符串作为HDF里的key
            wid = str(curd)

            # 把当前batch写入HDF5文件的对应组
            src_grp.create_dataset(wid, data=rid, **h5datawargs)
            tgt_grp.create_dataset(wid, data=rtd, **h5datawargs)

            curd += 1  # batch计数+1

        # ==================== 写入总batch数 ====================
        # 训练时加载数据用：知道总共有多少个batch
        rsf["ndata"] = np_array([curd], dtype=np_int32)

    # 打印最终打包了多少个batch
    print("Number of batches: %d" % curd)

# ==================== 命令行执行入口 ====================
if __name__ == "__main__":
    # 从命令行接收4个参数，传入handle函数
    # sys.argv[1] = 输入文件路径
    # sys.argv[2] = 目标文件路径
    # sys.argv[3] = 输出HDF5文件路径
    # sys.argv[4] = minbsize（GPU数量/最小batch）
    handle(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))