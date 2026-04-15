# Qwen3 LoRA 挂载层选择指南（基于本项目实现）

本文基于以下代码与模型打印进行说明：

- 模型定义：`transformer/PLM/QWen/v3/Decoder.py`
- 注意力实现：`modules/plm/qwen/v3.py`
- LoRA 注入逻辑：`utils/lora/base.py`、`utils/module/patcher.py`
- 训练入口：`adv/train/plm/train_lora_qwen.py`
- LoRA 配置：`cnfg/lora_1.py`
- 结构打印参考：`stdout.txt`（`Decoder(...)`）

---

## 1. 先看这份模型结构到底是什么

从 `stdout.txt` 的 `Decoder(...)` 可读到：

- `wemb`: `Embedding(151936, 4096)`，词嵌入
- `nets`: 36 层 `DecoderLayer`
  - `self_attn.net.adaptor`: `Linear(4096 -> 6144)`，QKV 合并投影
  - `self_attn.net.outer`: `Linear(4096 -> 4096)`，注意力输出投影（Wo）
  - `ff.net.0`: `Linear(4096 -> 24576)`，FFN 上投影（gate/up 合并）
  - `ff.net.2`: `Linear(12288 -> 4096)`，FFN 下投影
- `classifier`: `Linear(4096 -> 151936)`，LM head
- `out_normer`: 输出 RMSNorm

关键点：这版 Qwen3 里 **Wq/Wk/Wv 不是三个独立 Linear**，而是合并在 `self_attn.net.adaptor` 一个线性层里。

---

## 2. `name_cfunc` 实际是怎么生效的

在 `train_lora_qwen.py` 中，LoRA 注入调用：

- `std2lora(mymodel, ..., name_cfunc=lcnfg.name_cfunc, ...)`

`std2lora -> patch_std` 的行为是：

1. 遍历 `named_modules()`；
2. 仅当 `name_cfunc(module_name) == True` 才进一步判断类型；
3. 类型仅处理 `nn.Linear`、`nn.Embedding`，替换成 LoRA 版层。

因此：

- `name_cfunc` 的输入是 **模块名字符串**（例如 `nets.3.self_attn.net.adaptor`）。
- 它只能控制“选哪个模块”，不能控制模块内部某些权重切片。

---

## 3. 在 `cnfg/lora_1.py:24` 可以选择哪些部分

当前你写的是：

```python
name_cfunc = lambda mname: "self_attn" in mname
```

这会覆盖所有名字含 `self_attn` 的线性层，通常包含：

- `nets.*.self_attn.net.adaptor`（QKV 合并）
- `nets.*.self_attn.net.outer`（Wo）
- 以及可能匹配到 `ref_ropem` 分支下同名模块（若在模块树中出现）

下面给出更精细可控的写法：每个小节都包含“单独训练 LoRA 代码 + 解释它会训练到模型里的哪部分/起什么作用”。

### 3.1 只训练注意力 QKV+O（常见首选）

把 LoRA 挂到注意力的两处线性投影：

```python
name_cfunc = lambda n: (
    "self_attn.net.adaptor" in n or
    "self_attn.net.outer" in n
)
```

作用解释：

- `self_attn.net.adaptor`：本项目里它负责生成注意力里的 Q/K/V（QKV 合并在同一个 `Linear` 里）
- `self_attn.net.outer`：把注意力聚合后的结果（相当于 `Wo/O`）投影回 `4096` 维的主干隐藏空间
- 因此它能同时改变“关注哪里”（由 Q/K/V 决定）以及“把关注信息怎么写回隐藏状态”（由 Wo 决定）
- 在 LoRA 里这通常是最稳的 baseline，参数效率高且效果往往优于只训 QKV 或只训 Wo

### 3.2 只训练 QKV（不训练 Wo）

```python
name_cfunc = lambda n: "self_attn.net.adaptor" in n
```

作用解释：

- 只允许 LoRA 改变注意力的 Q/K/V 投影方式（本项目：`adaptor` 对应 Wq/Wk/Wv 的合并形式）
- 注意力权重分布（QK 的相似度）与被聚合的内容（V）会随之改变
- 但注意力输出写回主干的方式（Wo）保持冻结

### 3.3 只训练 Wo

```python
name_cfunc = lambda n: "self_attn.net.outer" in n
```

作用解释：

- Wo 决定了“注意力聚合后的向量如何变换/混合后写回隐藏状态”
- QKV（即注意力权重与聚合内容的来源）保持冻结，因此它通常比 3.1 更受限
- 但它能显著影响最终输出头前的表征质量，所以在显存极紧张时可以作为更轻量的替代

### 3.4 训练注意力 + FFN（高能力、较高显存）

```python
name_cfunc = lambda n: (
    "self_attn.net.adaptor" in n or
    "self_attn.net.outer" in n or
    "ff.net.0" in n or
    "ff.net.2" in n
)
```

作用解释：

- 除了注意力（QKV+Wo），还把前馈网络 FFN 的两段线性投影也纳入 LoRA
- FFN 是逐 token 的高维特征变换（通过 `Linear -> SwiGLU -> Linear` 提供非线性表达能力）
- 因此它同时具备：
  - 更强的上下文交互能力（attention）
  - 更强的 token 级表征改写能力（FFN）
- 代价是 LoRA 覆盖更多层，参数与训练激活压力更大，更容易 OOM

### 3.5 只训练 FFN

```python
name_cfunc = lambda n: ("ff.net.0" in n) or ("ff.net.2" in n)
```

作用解释：

- 注意力模块冻结，仅允许 FFN 改写每个 token 的特征
- 这种配置有时对“风格迁移/领域知识注入”更有效，因为 FFN 容易学习到任务特定的非线性映射
- 但对“需要精细改写注意力模式”的任务（如长依赖、复杂指令对齐）能力通常不如 attention+FFN

### 3.6 训练 embedding + attention（少见）

```python
name_cfunc = lambda n: ("wemb" in n) or ("self_attn.net" in n)
```

作用解释：

- `wemb` 训练意味着词表到向量空间的映射也会被 LoRA 调整
- 同时仍对 attention 做 LoRA
- 这属于更“重”的微调方式：可能带来更快的适配，但更容易过拟合或对泛化造成影响
- 在你的工程里通常不作为第一优先方案

### 3.7 训练 lm_head（`classifier`）

```python
name_cfunc = lambda n: "classifier" in n
```

作用解释：

- `classifier` 是最终把 `4096` 维隐藏状态映射到 vocab logits 的线性层
- 训练它相当于学习“在保持内部表征基本不变时，如何把表征更好地对齐到词表分布”
- 相比 attention/FFN，lm_head-only 往往更像“输出层校准”，上限通常较低

### 3.8 全部可替换层都上 LoRA（通常不建议）

```python
name_cfunc = lambda n: True
```

作用解释：

- 会把所有能替换成 LoRA 的 `nn.Linear / nn.Embedding` 都替换为 LoRA 版本
- 训练自由度最大，但参数数量与训练压力也最大
- 在你遇到 OOM 的上下文下，这通常不是稳妥选择

---

## 4. 能不能只做 Wk/Wv 的 LoRA？

结论：**按当前实现，不能仅靠 `name_cfunc` 精确做到只 Wk/Wv。**

原因：

- `Wq/Wk/Wv` 合并在一个 `Linear`（`self_attn.net.adaptor`）；
- `name_cfunc` 只能选中整个 `adaptor` 模块；
- 不能只对其中 K/V 对应行启用 LoRA，而对 Q 行禁用。

如要 KV-only，有两种改法：

1. 结构改造：把 `adaptor` 拆成 `q_proj/k_proj/v_proj` 三个线性层；
2. LoRA 层改造：对 `adaptor` 的 LoRA 增量施加行掩码，仅保留 K/V 对应区段。

---

## 5. LoRA 常见训练层策略（结合业界与本实现）

### 5.1 最常见配置（推荐起点）

- 目标层：attention 的投影层（本实现即 `adaptor + outer`）
- rank：`r=8/16`
- alpha：常见与 `r` 同或 `2r`（如 `16/32`）
- bias：通常不训（`update_bias=False`）

优点：参数效率好、泛化稳、显存可控。  
适用：大多数 SFT/指令微调场景。

### 5.2 能力优先配置

- 目标层：attention + FFN（`adaptor/outer/ff.net.0/ff.net.2`）

优点：任务拟合能力更强。  
代价：参数和显存开销上升，训练更易 OOM。

### 5.3 极限省参配置

- 只做 `adaptor`（QKV 合并）

优点：参数最省。  
代价：部分任务效果不如 `adaptor+outer`。

---

## 6. 面向你当前工程的推荐落地方案

考虑你此前出现 OOM，建议先用“稳健 baseline”：

```python
name_cfunc = lambda n: (
    "self_attn.net.adaptor" in n or
    "self_attn.net.outer" in n
)
```

并配合：

- `lora_features=8`，`lora_alpha=16`
- `update_bias=False`
- `fine_tune_linear_bias=False`
- `fine_tune_normer=False`
- 训练数据处理时，`sort.py` 的 train 长度上限用 `2048`（不要 `1048576`）
- 建议启用 `USE_AMP=1`

如果效果不够，再升级到 attention+FFN 方案。

---

## 7. 可直接粘贴到 `cnfg/lora_1.py` 的示例

### 示例 A：推荐 baseline（attention only）

```python
name_cfunc = lambda n: (
    "self_attn.net.adaptor" in n or
    "self_attn.net.outer" in n
)
```

### 示例 B：能力增强（attention + FFN）

```python
name_cfunc = lambda n: (
    "self_attn.net.adaptor" in n or
    "self_attn.net.outer" in n or
    "ff.net.0" in n or
    "ff.net.2" in n
)
```

### 示例 C：仅 QKV 合并层

```python
name_cfunc = lambda n: "self_attn.net.adaptor" in n
```

---

## 8. 快速自检（防止“以为挂了其实没挂”）

训练启动后可打印或检查：

- `print(mymodel)` 中目标层是否显示为 LoRA 版本模块；
- `requires_grad=True` 的参数名是否主要集中在 `...lora_wa` / `...lora_wb`；
- 若你只配了 attention，`ff` 和 `classifier` 不应出现 LoRA 参数。

---

## 9. 一句话总结

在本项目 Qwen3 实现中，`name_cfunc` 的最佳实践是按模块名精确选择 `adaptor/outer/ff` 等层；  
`Wk/Wv-only` 不能仅靠当前配置实现，需要改模型或改 LoRA 层逻辑。

---

## 10. 模型主要层结构与作用（按本项目 Qwen3 Decoder 打印）

本项目的 `Decoder` 是 **Decoder-only Transformer**。结合 `stdout.txt` 的结构打印，它的主流程可概括为：

`token ids -> wemb -> N x DecoderLayer(nets) -> out_normer -> classifier(lm_head) -> logits`

其中 `nets` 是 36 层堆叠（你打印里是 `(0)` 和 `(1-35)`）。

### 10.1 `wemb`：词嵌入层

- 结构：`Embedding(151936, 4096, padding_idx=151643)`
- 作用：
  - 将离散 token id 映射为 4096 维向量表示
  - `padding_idx` 对应 pad token，在计算中不参与有效梯度/学习

### 10.2 `nets`：堆叠的 `DecoderLayer`（自注意力 + 前馈网络）

每层 `DecoderLayer` 都有：

- `self_attn`：自注意力子层
- `ff`：前馈网络子层（PositionwiseFF）

并且在子层之间包含残差与归一化（你打印为 `ResSelfAttn(... normer ...)` 以及 `ff.normer`）。

#### 10.2.1 `self_attn`：自注意力子层

从 `modules/plm/qwen/v3.py` 看，本项目注意力实现使用 **QKV 合并投影**，打印为：

- `self_attn.net.adaptor: Linear(4096 -> 6144, bias=False)`
  - 作用：产生 Q/K/V（内部把输出切分为不同 head 的 Q、K、V）
  - 你这里 6144 可以理解为多头维度的合并结果（QKV 拼在同一个 Linear 里）
- `self_attn.net.outer: Linear(4096 -> 4096, bias=False)`
  - 作用：attention 计算后得到的聚合表示再投影回模型维度（常称 `Wo`）
- `self_attn.net.normer: Softmax(dim=-1)`
  - 作用：对注意力分数做归一化，得到注意力权重
- `self_attn.net.q_normer/k_normer: RMSNorm((128,), ...)`
  - 作用：对每个头的 Q/K 做归一化（提升训练稳定性，数值上更鲁棒）
- `self_attn.normer: RMSNorm((4096,), ...)`
  - 作用：注意力子层外侧的归一化（与残差一起稳定深层训练）

注意：你的打印里还出现了 `ref_ropem`（部分层内出现），这是位置编码/缓存共享相关的实现细节，不是额外的主干层。

#### 10.2.2 `ff`：前馈网络子层（PositionwiseFF）

从打印可见：

- `ff.net.0: Linear(4096 -> 24576, bias=False)`
- `ff.net.1: SwiGLU(SiLU)`
  - 作用：门控式非线性变换（SwiGLU）
- `ff.net.2: Linear(12288 -> 4096, bias=False)`
- `ff.normer: RMSNorm((4096,), ...)`

作用：

- FFN 是 **逐 token** 的高维特征变换，不像 attention 那样跨位置混合信息
- 通过扩展维度（4096 -> 24576 再压回 4096）并使用 SwiGLU 提供强表达能力

#### 10.2.3 为什么 LoRA 通常选 attention 投影层

在传统 Transformer/LoRA 场景中，attention 的投影矩阵（Q/K/V/O）对任务适配很敏感，而且：

- 参数规模相对可控
- 更新方向直接影响“信息如何在上下文之间流动”

在本项目由于 QKV 合并到 `self_attn.net.adaptor`，所以 LoRA 通常挂载到：

- `self_attn.net.adaptor`（相当于覆盖 Q/K/V）
- `self_attn.net.outer`（相当于覆盖 Wo/O）

### 10.3 `out_normer`：最终输出归一化

- 结构：`RMSNorm((4096,), ...)`
- 作用：
  - 在进入 `classifier` 前对最后层 hidden 做归一化
  - 提高输出分布与训练稳定性

### 10.4 `classifier`：语言模型输出头（lm_head）

- 结构：`Linear(4096 -> 151936, bias=False)`
- 作用：
  - 将每个 token 位置的 hidden 映射到词表维度，得到 logits
  - 你的模型还带了 `lsm: LogSoftmax(dim=-1)` 用于把 logits 转为对数概率

### 10.5 `cross_attn`：不存在

- 你打印为 `cross_attn: None`
- 说明这是纯 Decoder-only，没有 Encoder-Decoder 的交叉注意力模块

---

## 11. HPLSTM 与 Attention 的三种融合方案（先设计，后实现）

以下三种方案都可以在 `DecoderLayer` 中实现。区别主要在信息融合路径、训练稳定性、计算开销和可解释性。

**与仓库文件的对应关系（实现已对齐 §11）**

| 方案 | 文件 | 切换方式 |
|------|------|----------|
| A 并行融合 | `transformer/PLM/QWen/v3/Decoder_1.py` | `cnfg/lora.py` 中设 **`hplstm_fusion = "A"`**，或运行前 **`export HPLSTM_FUSION=A`**（`train_hplstm_qwen.py` 会动态加载对应 `Decoder`） |
| B 串联 | `transformer/PLM/QWen/v3/Decoder_2.py` | 默认 **`hplstm_fusion = "B"`**，或 **`export HPLSTM_FUSION=B`** |
| C ResHPLSTM 子层 | `transformer/PLM/QWen/v3/Decoder_3.py` | **`hplstm_fusion = "C"`** 或 **`export HPLSTM_FUSION=C`**；训练时 **`fine_tune_reslstm=True`**、**`fine_tune_hplstm=False`**（该实现无顶层 `hplstm`） |

**重要（与本项目 `ResSelfAttn` 一致）**

- `context = self.self_attn(...)` 的首段 **`context`** 已是 **Pre-Norm 自注意力子层输出并含该子层残差**（见 `modules/base.py` 中 `ResSelfAttn`），**不是**「未加残差的纯注意力」。
- 因此并行方案 A 的融合应写为 **`context + hplstm(inputo)`**（在 `context = self_attn(...)` 之后，即 **`context = context + h`**），**不要**再写 **`inputo + context + ...`**，否则会把主干加两次。

### 11.1 方案 A：Attention 与 HPLSTM 并行（同输入）后融合

结构思路：

- 输入 `x`
- `a = self_attn(x, ...)`
- `h = hplstm(x)`
- 融合后再做残差：`y = x + fuse(a, h)`（最简单 `fuse(a, h) = a + h`）
- 再进入 FFN

可行性：高。

优点：

- 两条分支都直接读取同一输入，互补性强（全局依赖与门控时序建模并存）
- 便于后续做支路贡献分析（ablation）

风险/注意：

- 直接 `context + h` 若两路尺度差异大，可通过学习率或后续归一化调节；需要时再自行加门控（本仓库实现为 **无额外标量系数**）。
- 计算与显存开销高于串联方案

具体实现见 **`Decoder_1.py`**。

1) `DecoderLayer.__init__`：`self.hplstm`（无并行支路缩放参数）。

2) `forward`：与 `Decoder.py` 一致先 `context = self.self_attn(...)`；`h = self.hplstm(inputo)`（增量分支用 `query_unit`）；`context = context + h`；再 `ff`。

3) 训练配置：`fine_tune_hplstm=True`；若开 LoRA，可将 LoRA 限在 attention。

参考代码（与 `Decoder_1.py` 一致，注意 **无** `inputo + context`（重复加主干））：

```python
from modules.hplstm.snbase import HPLSTM

class DecoderLayer(DecoderLayerBase):
    def __init__(self, ...):
        ...
        self.self_attn = ResSelfAttn(...)
        self.ff = PositionwiseFF(...)
        self.hplstm = HPLSTM(isize, num_head=num_head, act_drop=attn_drop)

    def forward(self, inputo, tgt_pad_mask=None, query_unit=None, slen=None, sliding_window_khead=None, **kwargs):
        if query_unit is None:
            context = self.self_attn(inputo, mask=tgt_pad_mask, slen=slen, sliding_window_khead=sliding_window_khead)
            h = self.hplstm(inputo)
            context = context + h
            context = self.ff(context)
            return context
        else:
            context, states_return = self.self_attn(
                query_unit, mask=tgt_pad_mask, states=inputo, slen=slen, sliding_window_khead=sliding_window_khead
            )
            h = self.hplstm(query_unit)
            context = context + h
            context = self.ff(context)
            return context, states_return
```

### 11.2 方案 B：HPLSTM 串联在 Attention 后，再做残差

结构思路：

- 输入 `x`
- `a = self_attn(x, ...)`
- `h = hplstm(a)`
- 残差：`y = x + h`
- 再进入 FFN

可行性：高（实现最简，工程风险最低）。

优点：

- 改动小，训练路径清晰
- 语义直观：先用 attention 做上下文聚合，再用 hplstm 做门控重整

风险/注意：

- HPLSTM 完全依赖 attention 输出，attention 偏差会传递到后续
- 若后续想在解码中利用 hplstm 的 states，推理代码需同步设计（当前可先不引入 states）

具体实现展开：

1) `DecoderLayer.__init__`：

- 新增 `self.hplstm = HPLSTM(...)`
- 不必额外加复杂门控，先做最小可复现实现

2) `DecoderLayer.forward`：

- 顺序改为：`self_attn -> hplstm -> residual -> ff`
- 训练分支与自回归分支都保持同一拓扑，避免训练/推理结构不一致

3) 训练配置：

- 只训 HPLSTM：`lora_features=None`、`fine_tune_hplstm=True`
- 若你只调用 `self.hplstm`，则 `fine_tune_reslstm` 建议关掉，避免解冻无效参数

实现见 **`Decoder_2.py`**：`context = self.self_attn(...)`；`h = self.hplstm(context)`；`context = context + h`；再 `ff`。

参考代码（与 `Decoder_2.py` 一致）：

```python
from modules.hplstm.snbase import HPLSTM

class DecoderLayer(DecoderLayerBase):
    def __init__(self, ...):
        ...
        self.self_attn = ResSelfAttn(...)
        self.hplstm = HPLSTM(isize, num_head=num_head, act_drop=attn_drop)
        self.ff = PositionwiseFF(...)

    def forward(self, inputo, tgt_pad_mask=None, query_unit=None, slen=None, sliding_window_khead=None, **kwargs):
        if query_unit is None:
            context = self.self_attn(inputo, mask=tgt_pad_mask, slen=slen, sliding_window_khead=sliding_window_khead)
        else:
            context, states_return = self.self_attn(
                query_unit, mask=tgt_pad_mask, states=inputo, slen=slen, sliding_window_khead=sliding_window_khead
            )
        h = self.hplstm(context)
        context = context + h
        context = self.ff(context)
        return context if query_unit is None else (context, states_return)
```

这是推荐的首个 baseline（实现最小、最易定位问题）。

### 11.3 方案 C：Attention 残差后，再追加一个 ResHPLSTM 子层

结构思路：

- `x1 = x + self_attn(x, ...)`（attention 子层残差已完成）
- `x2 = ResHPLSTM(x1)`（ResHPLSTM 内部再做规范化/残差）
- `y = FFN(x2)`

可行性：高（模块化最清晰）。

优点：

- 非常适合做结构化对比实验（打开/关闭 ResHPLSTM 子层）
- 与 Transformer“子层堆叠”的工程组织方式一致

风险/注意：

- 残差路径增多，需关注激活尺度与训练稳定性
- 若 `ResHPLSTM` 只定义不调用，即使解冻也不会得到有效梯度

具体实现展开：

1) `DecoderLayer.__init__`：

- 新增 `self.reslstm = ResHPLSTM(...)`
- 可选：新增 `use_reslstm` 开关，便于实验切换

2) `DecoderLayer.forward`：

- 先完成 attention 子层输出（含原有残差逻辑）
- 再显式调用 `self.reslstm(...)`
- 最后进入 `ff`

3) 训练配置：

- 若只做 ResHPLSTM 实验，解冻目标应指向 `reslstm`
- 若 `reslstm` 没进入 forward 路径，即使解冻也不会有效训练

实现见 **`Decoder_3.py`**：无 `hplstm` 模块，仅 `reslstm`。与 `Decoder.py` 相同先用 `context = self.self_attn(...)`，再 `context = self.reslstm(context)`，再 `context = self.ff(context)`。

参考代码（与 `Decoder_3.py` 一致，**不要**再写 `context = inputo + self_attn(...)`）：

```python
from modules.hplstm.snbase import HPLSTM, ResHPLSTM

class DecoderLayer(DecoderLayerBase):
    def __init__(self, ...):
        ...
        self.self_attn = ResSelfAttn(...)
        self.reslstm = ResHPLSTM(isize, num_head=num_head, dropout=dropout, act_drop=act_drop, HPLSTM=HPLSTM)
        self.ff = PositionwiseFF(...)

    def forward(self, inputo, tgt_pad_mask=None, query_unit=None, slen=None, sliding_window_khead=None, **kwargs):
        if query_unit is None:
            context = self.self_attn(inputo, mask=tgt_pad_mask, slen=slen, sliding_window_khead=sliding_window_khead)
        else:
            context, states_return = self.self_attn(
                query_unit, mask=tgt_pad_mask, states=inputo, slen=slen, sliding_window_khead=sliding_window_khead
            )
        context = self.reslstm(context)
        context = self.ff(context)
        return context if query_unit is None else (context, states_return)
```

这种方式最适合做“新增子层是否有效”的结构化对比。

### 11.3.1 三种方案对应的最小实验矩阵

- A（`Decoder_1`）：
  - 开：`self.hplstm`
  - 融合：`context = self_attn(...)` 再 `context + hplstm(与 attn 同源的 inputo/query_unit)`
  - 训练：先只解冻 `hplstm`，再尝试同时开 LoRA
- B（`Decoder_2`）：
  - 开：`self.hplstm`
  - 路径：`context = self_attn` → `hplstm(context)` → `context + h` → `ff`
  - 训练：`fine_tune_hplstm=True`，`fine_tune_reslstm=False`
- C（`Decoder_3`）：
  - 开：`self.reslstm`（无 `hplstm`）
  - 路径：`context = self_attn` → `reslstm(context)` → `ff`
  - 训练：只解冻 `reslstm`，**勿** `unfreeze_hplstm`

### 11.3.2 训练侧参考代码（只训练 HPLSTM/ResHPLSTM）

下面代码可放在 `train_hplstm_qwen.py` 的“非 LoRA 分支”中：

```python
from utils.train.base import freeze_module
from utils.train.ft import unfreeze_hplstm, unfreeze_reslstm, rgrad_filter

freeze_module(mymodel)
if lcnfg.fine_tune_hplstm:
    unfreeze_hplstm(mymodel)    # 只解冻 snbase.HPLSTM
if lcnfg.fine_tune_reslstm:
    unfreeze_reslstm(mymodel)   # 只解冻 snbase.ResHPLSTM
save_model_ps_func = rgrad_filter
```

建议起步开关：

```python
# cnfg/lora.py
lora_features = None
fine_tune_hplstm = True
fine_tune_reslstm = False
```

### 11.4 三种方案的选型建议

- 首个可复现实验：优先方案 B（最稳、改动最少）
- 需要结构可解释性：方案 C（最便于论文式对比）
- 追求表达上限：方案 A（并行融合更强，但调参成本最高）

### 11.5 仅训练 HPLSTM 分支时的通用注意项

- 先 `freeze_module(mymodel)` 冻结全模型
- 再按模块类型或模块名解冻 `hplstm/reslstm`
- 确保被解冻模块在 `forward` 实际被调用
- 用 `rgrad_filter` 保存仅 `requires_grad=True` 参数，避免混入冻结参数

### 11.6 自回归增量解码中缓存 HPLSTM recurrent state（尽量少改动）

#### 目的

当前 `Decoder_1/Decoder_2` 在 `query_unit != None`（增量解码）时调用：

- `context, states_return = self.self_attn(..., states=inputo, ...)`
- `h = self.hplstm(query_unit)`（**未显式传入 `states`**）

因为 `HPLSTM.forward(..., states=None)` 时不会返回 `states_return`，所以 HPLSTM 的 recurrent state 在每一步会相当于“重置/不携带”。

如果你希望在增量解码里真正使用 HPLSTM recurrent state，需要把每层的 state 同时缓存：

- `attn`：Self-Attn 的 KV cache，仍是 `(K, V)`
- `hplstm`：HPLSTM 的 recurrent state（HPLSTM 在 `states!=None` 时会返回 `states_return`）

#### state 数据结构（每层）

把原来每层的 `states`（原类型 `(K, V)`）升级为一个 dict：

```python
states_for_layer = {
    "attn": (K, V),      # 原先的 (K, V) KV cache
    "hplstm": hp_state  # HPLSTM recurrent state；初始用 "init"
}
```

其中：

- `hp_state = "init"`：让内部走 init_cx（MHPLSTMCore 里 `use_init_states = (states is None) or (states == "init")`）
- `hp_state` 的真实类型在 HPLSTM 返回里是 tensor 组成的 tuple（由 `LGateFunc / RS1cumsumstatnorm` 推导得到）

#### 需要改哪些地方（尽量少）

1) `transformer/PLM/QWen/v3/Decoder.py`
   - `build_states / greedy_decode / beam_decode` 里处理 `_states.get(_tmp, (None, None))` 的地方：
     - 传给 `net(...)` 的第一个参数由 `(None, None)` 变为 dict
     - 计算 `_slen/_sid`（滑动窗口用）时要从 `states["attn"][0]` 读取长度
   - beam 扩展已经支持 dict（`utils/decode/beam.py: expand_bsize_for_beam` 对 dict 会递归处理），所以只要 dict 结构一致通常不会炸。

2) `transformer/PLM/QWen/v3/Decoder_1.py`（方案 A）和 `Decoder_2.py`（方案 B）
   - 只需要改 `query_unit != None` 分支：把 `inputo` 当作 dict 取出两类 state
   - 调用 `self.hplstm(query_unit, states=hp_state_in)`，并把返回的 `hplstm_states_return` 再塞回 dict

#### Decoder.py 的最小改动要点（示意代码）

把原先类似：

```python
_ = _states.get(0, (None, None,))[0]
_slen = 0 if _ is None else _.size(-1)
```

改为（示意）：

```python
_st0 = _states.get(0, (None, None,))
_attn = _st0.get("attn", (None, None)) if isinstance(_st0, dict) else _st0
_ = _attn[0]
_slen = 0 if _ is None else _.size(-1)
```

并把传入 `net(...)` 的默认值从 `(None, None)` 改成：

```python
{"attn": (None, None), "hplstm": "init"}
```

#### Decoder_1 的最小改动要点（示意代码，变量名尽量沿用）

在 `Decoder_1.DecoderLayer.forward(..., query_unit != None)` 的 else 分支中，按如下方式组织：

```python
if query_unit is None:
    context = self.self_attn(inputo, ...)
    h = self.hplstm(inputo)                # 训练/整段仍可保持不缓存
    context = context + h
    context = self.ff(context)
    return context
else:
    # inputo 在增量模式下应是 dict: {"attn": (K,V), "hplstm": hp_state}
    attn_states_in = inputo.get("attn", (None, None))
    hp_state_in = inputo.get("hplstm", "init")

    context, states_return = self.self_attn(
        query_unit, mask=tgt_pad_mask, states=attn_states_in, slen=slen, sliding_window_khead=sliding_window_khead
    )

    h, hplstm_states_return = self.hplstm(query_unit, states=hp_state_in)

    context = context + h
    context = self.ff(context)

    # 向父模块返回“下一步要用的两类 state”
    states_return = {
        "attn": states_return,                  # (K, V)
        "hplstm": hplstm_states_return         # recurrent state
    }
    return context, states_return
```

`Decoder_2.py`（方案 B）逻辑完全类似，只是融合时 `hplstm` 的输入是 `context` 还是别的中间量按你原方案写即可。

#### 注意点

- 这套改动会改变 `Decoder.py` 中 `_states` 的类型（从 tuple 变成 dict），所以所有读取 KV 长度、以及 beam/index 扩展的逻辑都要保持一致。
- 如果你仍只想先保证能跑，最小可行路径是先只在 greedy_decode 上做，beam 上做完再扩展到 beam_decode（但最终建议两者保持同一结构）。

