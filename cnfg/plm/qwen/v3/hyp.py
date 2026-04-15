#encoding: utf-8
"""
Qwen3 基座模型的「结构级」超参（与具体任务/数据无关）。

**在配置链中的位置**
- 继承：`from cnfg.hyp import *`（见文件内 `cnfg.hyp`，全局默认 Transformer 超参）。
- 本文件覆盖其中与 **Qwen3 架构** 一致的部分（如 SwiGLU、RoPE、Pre-LN v2、无 sliding window 等）。
- 若需「解释后的」派生量（如 encoder/decoder 是否用相对位置、是否开 RoPE），见同目录 **`ihyp.py`**（`ihyp` = interpret hyp）。

**与 `cnfg/ihyp.py` 的关系**
- `cnfg/ihyp.py` 在 `from cnfg.hyp import *` 之后，根据 `computation_order`、`use_rope` 等 **推导** 出一批默认变量（如 `norm_residual_default`）。
- 训练 Qwen3 时，应通过 **`cnfg.plm.qwen.v3.ihyp`** 导入：先执行 `cnfg.ihyp` 再 `from cnfg.plm.qwen.v3.hyp import *`，最后 `ihyp.py` 里会 **再次** 根据 v3 覆盖后的 `computation_order` 等重算部分派生量，避免仍停留在根 `cnfg.hyp` 的旧值。

**本文件不负责**：batch 大小、学习率、HDF5 压缩等（仍在根 `cnfg/hyp.py` 与 `cnfg/ihyp.py`）。
"""

from cnfg.hyp import *

# ----- 训练/推理稳定性与结构偏好（与 Qwen3 官方实现一致） -----
ease_optimization = True
disable_ffn_bias = True
remove_classifier_bias = True
add_attn_qkv_bias = False  # Qwen3 0.6/1.7/4/8/14B 均为 False
add_self_attn_qknorm = True
sliding_window = -1  # -1 表示关闭；Qwen3 上述规格不使用 sliding window attention
sliding_window_khead = 0

# choices: None, "GeLU", "GeLUTanh", "Swish", "Sigmoid", "SReLU", "Mish", "NormSwish"
advance_activation_function = None
# FFN 内 GLU 变体：choices: None, "GLU", "GEGLU", "SwiGLU", "GETanhGLU".
use_glu_ffn = "SwiGLU"  # Swish 用于 0.6/1.7/4/8/14B 的另一种写法见项目注释

# Pre-LN 与残差顺序：choices: "v1", "v2"（v2 为常见 Qwen/LLaMA 类）
computation_order = "v2"

# 位置编码缓存长度上界（与 mask / 缓存相关，非单条样本最大长度唯一来源）
# 与 cnfg/hyp.py 的批次长度上限配合：通过降低 HDF5 的 max seq_len，避免 attention/cached mask 的显存峰值过高。
cache_len_default = 2048

# 相对位置（T5 式 bucket）：0 表示关闭
# 论文参考：Self-Attention with Relative Position Representations (NAACL 2018)
use_k_relative_position = 0
relative_position_max_bucket_distance = 0
# RoPE：见 RoFormer；disable_std_pemb=True 表示不用正弦绝对位置表
disable_std_pemb = True
use_rope = True
# ALiBi：见 Train Short, Test Long
use_alibi = False

# BPE 全词表时通常无真正 <unk>，可关
use_unk = False
