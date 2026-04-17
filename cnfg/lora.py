#encoding: utf-8
"""
LoRA 微调默认配置，供 `adv/train/plm/train_lora_qwen.py` 在构建模型时 `import cnfg.lora as lcnfg` 读取。

- `std2lora(..., name_cfunc=lcnfg.name_cfunc)` 会遍历 Decoder 中的 **nn.Linear / nn.Embedding**，
  仅当 **`name_cfunc(子模块名字符串) == True`** 时才替换为 LoRA；否则该层保持原样（是否训练仍受 freeze 等控制）。
- 运行前可用环境变量 **`LORA_RANK` / `LORA_ALPHA`** 覆盖 `lora_features` / `lora_alpha`（见 `train_lora_qwen.py` 开头）。
- `train_hplstm_qwen.py` 可用 **`HPLSTM_FUSION`**（`A`/`B`/`C`）覆盖 **`hplstm_fusion`**，以选择 `Decoder_1`/`Decoder_2`/`Decoder_3`。

**name_cfunc**：默认 **全部** 参与 LoRA（与 `utils.func.always_true` 一致）。若需「只给 attention / 不给 lm_head」等，改为自定义
`lambda mname: ...`，按 `mname` 子串判断。
"""

from utils.func import always_true as name_cfunc

save_base = True
lora_fine_tune_m = None

lora_features = None  # HPLSTM-only：禁用默认 LoRA 训练分支；可通过环境变量 LORA_RANK/LORA_ALPHA 重新启用
lora_alpha = 16       # 保留字段占位：仅当 lora_features 非空时才会生效，可被环境变量覆盖
scaling = 1.0       # 直接 scaling；若与 lora_alpha 同时配置，以 std2lora 内逻辑为准
update_bias = False # LoRA 版 Linear 是否训练 bias（False=只训 LoRA 低秩分支）

# 上：name_cfunc 已从 always_true 导入，等价于 lambda mname: True，避免误写成非法的 `True *`。
name_cfunc = lambda mname: "self_attn" in mname 

keep_lora_weight_tying = True  # 是否尽量恢复原模型的参数共享（weight tying）

fine_tune_linear_bias = False  # 是否在 LoRA 之外再解冻部分 Linear 的 bias
fine_tune_normer = False        # 是否再解冻 LayerNorm 等
fine_tune_hplstm = True         # 是否再解冻 HPLSTM
fine_tune_reslstm = False        # 是否再解冻 ResHPLSTM

# HPLSTM 与 Attention 融合拓扑（仅 `adv/train/plm/train_hplstm_qwen.py`）："A" | "B" | "C"
# → 分别加载 `transformer/PLM/QWen/v3/Decoder_1|2|3.py`。可用环境变量 **HPLSTM_FUSION** 覆盖（见该脚本开头）。
hplstm_fusion = "A"

name_cfunc_lb = lambda mname: True      # fine_tune_linear_bias=True 时，对哪些参数名解冻 bias
name_cfunc_normer = lambda mname: True  # fine_tune_normer=True 时，对哪些参数名解冻 norm

prefix_ids = None           # 验证时可选固定前缀 token id；None 表示不强行加
find_common_prefix = False # 无 prefix_ids 时是否从 dev HDF5 推断公共前缀
