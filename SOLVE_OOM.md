# 解决 LoRA 训练/预测 OOM（显存不足）

当训练或预测 Qwen3-8B + LoRA 在单卡 32GiB 上出现 `torch.OutOfMemoryError` 时，按下面顺序操作。

---

## 零、先区分是训练 OOM 还是预测 OOM

- **训练 OOM**：常见堆栈在 `train_lora_qwen.py` / forward / backward，通常与 `train.h5/dev.h5` 打包 batch 过大有关（见第一～七节）。
- **预测 OOM**：常见堆栈在 `adv/predict/plm/qwen/predict.py -> beam_decode -> index_tensors`，例如：
  - `torch.OutOfMemoryError: Allocation on device 0 would exceed allowed memory`
  - 已分配接近设备上限（如 `30.35 GiB / 31.36 GiB`），再申请约 `100 MiB` 失败。

若是后者，优先看下面“八、预测阶段 OOM（beam decode）”。

---

## 一、原因简述

- 每个 batch 的**形状在生成 HDF5 时就固定**了，由 `cnfg/hyp.py` 的 `max_tokens_gpu`、`max_sentences_gpu` 控制。
- 默认 `max_tokens_gpu = 6144`、`max_sentences_gpu = 6144//6 = 1024`，对 8B 模型而言单 batch 显存过大，容易在 forward（如 RMSNorm、LoRA Linear）时再申请几十～两百 MiB 就 OOM。
- 训练脚本**只读已有 HDF5**，不会在运行时改 batch 大小，因此要**先改配置、重新生成 HDF5，再训练**。

---

## 二、推荐步骤（按顺序做）

### 1. 减小每 batch 的 token/句数上限

编辑 **`cnfg/hyp.py`**，把与 batch 相关的量改小，例如：

```python
# 原值示例（易 OOM）
# max_tokens_gpu = 6144
# max_sentences_gpu = max_tokens_gpu // 6

# 单卡 32GiB、Qwen3-8B + LoRA 建议先改为：
max_tokens_gpu = 2048
max_sentences_gpu = max_tokens_gpu // 6   # 约 341
```

若仍 OOM，可再改为 `max_tokens_gpu = 1024`（或更小），然后**必须重新生成 train/dev HDF5**。

### 2. 用新配置重新生成 HDF5

在项目根目录执行（与 LORA_DATA_FORMAT_AND_STEPS.md 中 3.2～3.5 一致，仅需把 `$WKD` 换成你的数据目录）：

```bash
export WKD=cache/llm/pubdatasets_metamathqa   # 或你的 DATA_ID 对应目录

python tools/plm/mkiodata_llmdec.py $WKD/src.train.srt.ids $WKD/tgt.train.srt.ids $WKD/train.h5 1
python tools/plm/mkiodata_llmdec.py $WKD/src.dev.srt.ids   $WKD/tgt.dev.srt.ids   $WKD/dev.h5   1
```

未排序的 .ids 用 `src.train.txt.ids` / `tgt.train.txt.ids` 等替代上面的 `*.srt.ids`。  
生成后，每个 batch 的 token 数、句数会受新的 `max_tokens_gpu` / `max_sentences_gpu` 限制。

### 3. 开启 AMP（可选，进一步省显存）

- **方式 A**：在 **`cnfg/base.py`** 里设 `use_amp = True`。
- **方式 B**：不改 cnfg，用环境变量（推荐）：

```bash
USE_AMP=1 bash task.sh
```

或：

```bash
export USE_AMP=1
python adv/train/plm/train_lora_qwen.py
```

训练脚本会读 `USE_AMP=1`（或 `true`/`yes`）并开启混合精度，减少激活与部分权重的显存。

### 4. 再跑训练

```bash
bash task.sh
```

或按你现有方式启动 `train_lora_qwen.py`。若仍 OOM，回到步骤 1，再减小 `max_tokens_gpu` 并重新做步骤 2。

---

## 三、其他说明

| 项目 | 说明 |
|------|------|
| **Batch 大小** | 完全由生成 HDF5 时的 `cnfg/hyp.py` 决定；改小后必须重新跑 `mkiodata_llmdec.py`，否则旧 HDF5 仍是大批次。 |
| **LoRA Embedding 的 view 报错** | 若出现 “view size is not compatible ... Use .reshape(...)”：当前 `modules/lora/base.py` 的 Embedding 已用 `reshape`，若仍报错请确认运行的是该仓库最新版。 |
| **Dev Loss 为 nan** | 可能与数据或学习率有关；先解决 OOM，再检查数据与 lr；必要时在 loss 后加 `torch.isfinite(loss)` 等检查。 |
| **多卡** | 若有多卡，可考虑 DataParallel 或 ZeRO 等把优化器/梯度摊到多卡；单卡 32GiB 下优先按上面减 batch + AMP。 |

---

## 四、关于“减少单步压力（tokens_optm）”的影响、原理与操作

你在 `cnfg/base.py` 里会看到：

```python
tokens_optm = 25000
```

它控制的是“累计多少 token 再执行一次 optimizer step（参数更新）”这一节奏（可理解为有效 batch 的一部分调度开关）。  
当单卡显存紧张时，适当降低它可以减少单步更新时的瞬时压力，降低 OOM 概率。

### 1) 原理（为什么会缓解 OOM）

- 训练不是每个 micro-batch 都立刻 `optimizer.step()`，而是会累计到一定 token 量再更新。
- `tokens_optm` 越大，单次更新前累计的计算/状态越多，更新节点附近的显存峰值更容易偏高。
- 把 `tokens_optm` 调小，相当于“更频繁地做较小步更新”，通常能降低峰值显存压力。

> 注意：它不是替代 `max_tokens_gpu` 的主开关。  
> **OOM 的第一优先项仍是 `cnfg/hyp.py` 的 batch 打包上限**（且要重生 H5）。

### 2) 对训练效果/速度的影响

- **显存稳定性**：更稳，OOM 风险降低（正向收益）。
- **训练速度**：通常会略慢（step 更频繁，更新开销比例变大）。
- **收敛噪声**：梯度平均规模变小，训练曲线可能更抖一些；一般可通过多跑一点 step/epoch 抵消。
- **最终效果**：通常不会“必然变差”，但需要结合学习率与总步数观察 dev 指标。

### 3) 推荐操作方式（从保守到激进）

在 `cnfg/base.py` 做分档尝试：

```python
# 原值
# tokens_optm = 25000

# 档位 1（先试）
tokens_optm = 16000

# 档位 2（仍 OOM）
# tokens_optm = 12000

# 档位 3（仍 OOM）
# tokens_optm = 8000
```

建议每次只改一个档位，连续观察 500~2000 step 的稳定性与 dev 指标变化，再决定是否继续下调。

### 4) 与其他开关的配合顺序（实战）

1. 先降 `cnfg/hyp.py`：`max_tokens_gpu`（并重生 H5）  
2. 开 `USE_AMP=1`  
3. 再降 `tokens_optm`（`25000 -> 16000 -> 12000 -> 8000`）

这样做更稳，且更容易判断“到底是 batch 过大，还是更新节奏过激”导致的问题。

---

## 五、激活检查点（Gradient Checkpointing）

**含义**：前向时不保存整网中间激活，反向时再算一遍对应子图，用**算力换显存**。数学上与常规训练一致（同一计算图），**不缩小 batch、不改数据、不换模型结构**，适合「宁可慢也要和原设定可比」的场景。

**代价**：训练时间通常增加约 **30%～100%+**（依层数与实现而定）；混合精度下偶发**数值非确定性**（与无 checkpoint 的逐位一致不保证，但 loss 曲线一般仍可对照）。

### 在本仓库里应改哪里

训练时 Decoder-only 的整段前向在 **`transformer/PLM/LLMDecoder.py`** 的 `Decoder.forward`：当 `states is None`（常见训练路径）时，大致为：

```python
for net in self.nets:
    out = net(out, tgt_pad_mask=_mask, **kwargs)
```

这里每一层（或「共享层重复跑 `num_layer` 次」）对应一次 `DecoderLayer`（见 `transformer/PLM/QWen/v3/Decoder.py` 的 `DecoderLayer`）。

### 推荐实现步骤（概念）

1. **加开关**（便于对照实验）：在 `cnfg/hyp.py` 或 `cnfg/ihyp.py` 增加布尔项，例如 `use_gradient_checkpointing = False`，需要时再改为 `True`（或读环境变量）。
2. **在 `LLMDecoder.forward` 顶部**增加：  
   `from torch.utils.checkpoint import checkpoint`
3. **仅包裹「整段序列训练」分支**：在 `if states is None:` 下，把上面的循环改为用 `checkpoint` 包一层，例如（需与当前参数一致，**把 `tgt_pad_mask=_mask` 和 `**kwargs` 传进闭包**）：

   ```python
   def _layer_forward(inp, layer):
       return layer(inp, tgt_pad_mask=_mask, **kwargs)

   for net in self.nets:
       out = checkpoint(_layer_forward, out, net, use_reentrant=False)
   ```

   若你更习惯 lambda，需保证 lambda 内仍传入 `_mask` 与 `kwargs`。PyTorch 2.x 建议 **`use_reentrant=False`**。

4. **不要**对 `states is not None` 的增量解码路径随意套 checkpoint（推理/带 KV 的路径逻辑不同）；LoRA 训练通常走 `states is None`，以训练脚本实际调用为准。
5. **Qwen3 在本项目中 `share_layer=True` 时**：`self.nets` 里多个元素可能指向**同一** `DecoderLayer` 实例，循环仍执行 `num_layer` 次前向；对**每一次**调用使用 checkpoint，仍可分段释放中间激活，行为与「多层共享参数」的设定一致。
6. **验证**：先跑少量 step，确认 loss 有限、无 NaN；再与不开 checkpoint 的曲线对比（不必逐点相同，趋势应接近）。

### 若仍 OOM

- 先保证 **`max_tokens_gpu` + 重生 HDF5** 与 **AMP** 已到位；checkpoint 主要缓解 **深度方向** 的激活存留，极长序列时 attention 中间张量仍可能很大。
- 可与 **`tokens_optm`** 下调、`USE_AMP=1` 叠加；在论文/报告中写明「启用 gradient checkpointing」即可保持可比性说明清晰。

---

## 六、其他「不改变可比性」或影响可说明的省显存手段

下列方式**不改变模型结构、不改数据打包形状**（在固定 HDF5 与超参前提下），便于与基线对照；请在实验记录里**写清开关**，便于审稿与复现。

| 手段 | 做法（本仓库） | 可比性说明 |
|------|----------------|------------|
| **混合精度 AMP** | `USE_AMP=1` 或 `cnfg/base.py` 的 `use_amp = True` | 与 FP32 训练相比可能有微小数值差；同一 AMP 设定下各 run 之间可比。 |
| **关闭 `torch.compile`** | `cnfg/hyp.py` 中 `use_torch_compile = False`（默认即为关） | 关闭后为 eager 执行，省编译器额外开销/显存波动；与「未 compile」的数学路径一致。 |
| **独占 GPU** | 训练前 `nvidia-smi` 确认无其它占显存进程 | 不改变算法，仅避免峰值被其它任务抬高。 |
| **下调 `tokens_optm`** | `cnfg/base.py` 减小 `tokens_optm`（见第四节） | 改变的是**何时**做 `optimizer.step()`，若总 token 与学习率调度对齐，可保持「有效训练量」可比；需在文档中写清。 |
| **Gradient checkpointing** | 第五节 | 计算图等价，主要多花时间；建议写明启用。 |

**会明显改变「故事」或需单独当实验组的**（不建议与主 LoRA 基线混比而不加说明）：换更小基座、权重量化、ZeRO/FSDP、改写注意力为不同实现（如 FlashAttention）且未做对齐验证等。

---

## 七、相关文件速查

- **Batch 上限**：`cnfg/hyp.py`（`max_tokens_gpu`、`max_sentences_gpu`）
- **生成 HDF5**：`tools/plm/mkiodata_llmdec.py`（读 `cnfg.ihyp` 的上述值）
- **训练入口**：`adv/train/plm/train_lora_qwen.py`（支持 `USE_AMP`、`DATA_ID`、`PRE_TRAINED_M`、`LORA_RANK`、`LORA_ALPHA`）
- **Decoder 前向（加 checkpoint 的挂点）**：`transformer/PLM/LLMDecoder.py` 的 `Decoder.forward`
- **Qwen3 单层与 Decoder 定义**：`transformer/PLM/QWen/v3/Decoder.py`
- **`torch.compile` 总开关**：`cnfg/hyp.py` 的 `use_torch_compile` → `utils/torch/comp.py` 中映射为 `torch.compile` 或恒等函数
- **任务脚本**：`task.sh`（可设 `USE_AMP=1`）
- **数据与流程**：`LORA_DATA_FORMAT_AND_STEPS.md`

---

## 八、预测阶段 OOM（beam decode）

适用报错特征（与你的 `stderr-p.txt` 一致）：

- 调用链包含 `adv/predict/plm/qwen/predict.py`、`transformer/PLM/QWen/v3/Decoder.py`；
- OOM 点在 `beam_decode` 中重排状态（`index_tensors`）；
- 显存已接近满载，仅剩几十 MiB。

### 8.1 最有效的降显存顺序（建议按序做）

1. **先降 `beam_size`**（最直接）
   - 例如 `cnfg/base.py`：`beam_size = 4 -> 2`；仍 OOM 再到 `1`（即 greedy）。
2. **再降生成长度 `max_len`**
   - `adv/predict/plm/qwen/predict.py` 里默认 `max_len = 512`，可先改到 `256`，仍 OOM 再 `128`。
3. **减小推理 batch**
   - 重新生成 `test.h5` 时降低打包强度（让每个 batch 更小），本质是降低单次 decode 并行样本数。
4. **确保 GPU 独占**
   - 运行前先 `nvidia-smi`，避免其他进程占用显存。

### 8.2 具体操作示例

#### A) 先改 beam（推荐第一步）

编辑 `cnfg/base.py`：

```python
beam_size = 2   # 原来若是 4，先降到 2；仍 OOM 再改 1
```

#### B) 再改最大生成长度

编辑 `adv/predict/plm/qwen/predict.py`：

```python
max_len = 256   # 原默认 512，先减半
```

#### C) 若仍 OOM，重做更小 batch 的 `test.h5`

沿用 `tools/plm/llmdec/mktest.py` 生成 `test.h5`，但降低打包规模后重生成（与当前配置保持一致）。

> 注意：预测用 `test.h5` 必须来自 `mktest.py`，不要用 `dev.h5` 代替。

### 8.3 建议的“保守可跑通”起始配置

- `beam_size = 1`
- `max_len = 256`
- 独占 GPU 后再跑 `task-predict.sh`

跑通后再逐步回调（先把 `beam_size` 调高，再视情况恢复 `max_len`），每次只改一个量，方便定位阈值。

### 8.4 这些参数的原理与对训练影响

下表对应 8.1 里三类主要参数：

| 参数 | 为什么能降显存（原理） | 对预测结果/速度的影响 | 对训练的影响 |
|------|--------------------------|------------------------|--------------|
| **`beam_size`** | beam search 会在每步保留并扩展多个候选，KV/cache 与状态重排（`index_tensors`）近似随 beam 线性放大；beam 越大，解码态显存越高。 | 降低后：显存明显下降、速度更快；但搜索空间变小，文本质量/稳定性可能略降。`beam_size=1` 退化为 greedy。 | **无直接影响**（仅推理参数，不改变训练图与梯度）。 |
| **`max_len`** | 限制解码步数上限，减少每条样本在时间维累计的中间状态与缓存占用。 | 降低后：显存更稳、推理更快；但可能提前截断长答案，召回下降。 | **无直接影响**（不改变训练时序列打包与损失计算）。 |
| **推理 batch 大小**（由 `test.h5` 打包决定） | 同时并行解码的样本数变少，激活与解码状态近似按 batch 线性下降。 | 降低后：单步更稳、OOM 风险低；但总耗时通常上升（吞吐下降）。 | **无直接影响**（仅影响预测吞吐；不改训练权重更新）。 |

补充说明：

- 上述三个参数属于**解码阶段参数**，不会修改模型权重，也不会改变已经完成训练的 checkpoint。
- 你若只改这些参数并重新跑 `predict.py`，属于“同一模型不同解码策略”的对比；建议在实验记录中标注，便于复现。
- 真正影响训练显存/收敛的主开关仍是前文一到七节中的 `max_tokens_gpu`、`max_sentences_gpu`、`tokens_optm`、AMP、checkpointing 等。
