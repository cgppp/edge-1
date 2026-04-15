#encoding: utf-8
"""
Qwen3 的「解释后」超参（interpreted hyper-parameters），供模型构建、数据 IO 等统一读取。

**导入顺序（重要）**
1. `from cnfg.ihyp import *`  
   载入根目录 **`cnfg.hyp`** 经 **`cnfg/ihyp.py`** 解释后的全部符号（如 `h5_fileargs`、`max_sentences_gpu`、各类 `*_default` 等）。
2. `from cnfg.plm.qwen.v3.hyp import *`  
   用 **Qwen3 结构级** 覆盖（`cnfg/plm/qwen/v3/hyp.py`）：如 `computation_order="v2"`、`use_rope=True`、`use_glu_ffn="SwiGLU"` 等。
3. 本文件 **再次** 定义若干与架构相关的派生量（如 `norm_residual_default`、bias 默认、位置编码开关），  
   确保它们基于 **v3.hyp 覆盖之后** 的 `computation_order`、`ease_optimization` 等计算，而不是根 `cnfg.hyp` 的旧值。

**与根 `cnfg/ihyp.py` 的差异**
- 根 `cnfg/ihyp.py` 在 `from cnfg.hyp import *` 之后推导一次；若只导入根 `cnfg.hyp` 而不经过 v3 的 `hyp`/`ihyp`，**RoPE/计算顺序** 等可能与 Qwen3 不一致。
- 本文件 **不重复** HDF5 压缩、`h5_fileargs`、编译选项等大块逻辑；这些仍来自 **`cnfg.ihyp`** 的第一次导入。
- 本文件将 **正弦位置编码频率** `sinusoid_base_frequency` 设为 **1e6**（与根 ihyp 的 1e4 区分），与 Qwen3 0.6B 等文档一致。

**谁应 `import` 本模块**
- 训练入口、加载 Qwen3 权重/构建图的代码：应使用 **`from cnfg.plm.qwen.v3.ihyp import *`**（或显式子集），而不是仅 `cnfg.ihyp`。
- 若某工具脚本（如早期 `tools/plm/mkiodata.py`）仍写 `from cnfg.ihyp import *`，则 batch/HDF5 等仍用全局 `cnfg.hyp`；**架构相关** 若与 Qwen3 训练脚本不一致，需自行评估是否改为导入本文件。

**与 `cnfg/plm/qwen/v3/hyp.py` 的分工**
- `hyp.py`：只放 **结构/算法开关** 的原始赋值（与 Qwen3 官方实现一致）。
- `ihyp.py`：把上述开关 **展开** 为「模型代码里真正用到的」布尔/元组默认值（如 encoder/decoder 是否启用相对位置）。
"""

from math import inf

from utils.fmt.parser import parse_double_value_tuple, parse_none

from cnfg.ihyp import *
from cnfg.plm.qwen.v3.hyp import *

# biases
enable_prev_ln_bias_default = enable_proj_bias_default = not ease_optimization

# computation order
norm_residual_default = not (computation_order.lower() == "v2")

# Layer Norm
enable_ln_parameters = True

# activation fucntion
use_adv_act_default = advance_activation_function is not None
adv_act = advance_activation_function.lower() if use_adv_act_default else None
inplace_after_Custom_Act = use_adv_act_default and (adv_act not in set(["sigmoid"]))

# absolute position encoding
sinusoid_base_frequency = 1e6  # 1e6 for 0.6B

# relative position encoding
use_k_relative_position_encoder, use_k_relative_position_decoder = parse_double_value_tuple(use_k_relative_position)
rel_pos_enabled = (max(use_k_relative_position_encoder, use_k_relative_position_decoder) > 0) or use_rope or use_alibi
relative_position_max_bucket_distance_encoder, relative_position_max_bucket_distance_decoder = parse_double_value_tuple(relative_position_max_bucket_distance)
disable_std_pemb_encoder, disable_std_pemb_decoder = parse_double_value_tuple(disable_std_pemb)
relpos_reduction_with_zeros = True

# hyper-parameters
inf_default = inf

ieps_default = 1e-9
ieps_ln_default = 1e-6
ieps_adam_default = 1e-9
ieps_ln_default = parse_none(ieps_ln_default, ieps_default)
ieps_adam_default = parse_none(ieps_adam_default, ieps_default)
ieps_noise_default = ieps_ln_default
ieps_upper_bound_default = ieps_default
ieps_dropout_multinomial_default = ieps_default

adam_betas_default = (0.9, 0.98,)
