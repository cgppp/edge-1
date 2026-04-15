# HPLSTM 改造指南（统一版，含代码对照）

本文聚焦本仓库 Qwen3 路线下，如何把 LoRA 流程切到 HPLSTM，以及如何在 `DecoderLayer` 做 HPLSTM/ResHPLSTM 融合。内容已和当前实现对齐：

- 训练入口：`adv/train/plm/train_hplstm_qwen.py`
- 融合实现：`transformer/PLM/QWen/v3/Decoder_1.py` / `Decoder_2.py` / `Decoder_3.py`
- 基线结构：`transformer/PLM/QWen/v3/Decoder.py`

---

## 1. 术语与约定（先统一）

- `attn_out`：`self.self_attn(...)` 的返回，已是 `ResSelfAttn` 子层输出。
- `h_out`：`self.hplstm(...)` 或 `self.reslstm(...)` 的输出。
- `query_unit`：增量解码单步输入（自回归一步）。
- `layer_state`：每层解码缓存状态。当前代码存在两种形态：
  - tuple 形态：`(K, V)`（仅 attention KV）
  - dict 形态：`{"attn": (K, V), "hplstm": hp_state}`
- `融合方案 A/B/C`：分别对应 `Decoder_1.py` / `Decoder_2.py` / `Decoder_3.py`。

关键注意：不要把 `attn_out` 当“纯 attention 未残差输出”。在本项目中它已带 attention 子层残差路径，因此融合时通常是 `attn_out + h_out`，不要再额外 `+ inputo`。

---

## 2. 训练脚本行为（与配置项一一对应）

以 `adv/train/plm/train_hplstm_qwen.py` 为准：

1. 通过 `hplstm_fusion`（或环境变量 `HPLSTM_FUSION`）动态选择 `Decoder_1/2/3`。
2. 若 `lora_features is not None`，走 LoRA 分支（`std2lora`）。
3. 若 `lora_features is None`，走 HPLSTM/ResHPLSTM 微调分支：
   - `freeze_module(mymodel)` 先冻结全网；
   - `fine_tune_hplstm=True` 时调用 `unfreeze_hplstm(mymodel)`；
   - `fine_tune_reslstm=True` 时调用 `unfreeze_reslstm(mymodel)`；
   - 保存参数过滤函数使用 `rgrad_filter`（仅保存可训练参数）。

推荐最小开关：

```python
# cnfg/lora.py
lora_features = None
hplstm_fusion = "B"
fine_tune_hplstm = True
fine_tune_reslstm = False
```

常用环境变量：

- `HPLSTM_FUSION=A|B|C`
- `DATA_ID=...`
- `PRE_TRAINED_M=...`
- `USE_AMP=1`

---

## 3. 三种融合方案（结构 + 代码落点）

### 3.1 方案 A（`Decoder_1.py`）：并行融合

路径：

- `attn_out = self_attn(x, ...)`
- `h_out = hplstm(x)`（增量分支用 `query_unit`）
- `context = attn_out + h_out`
- `context = ff(context)`

特点：

- 表达上限高，attention 与 HPLSTM 对同源输入并行建模。
- 调参和资源成本最高。

状态管理：

- 当前 `Decoder_1.py` 与 `Decoder_2/3` 一样，主状态仍是 tuple 形态 `(K, V)`。
- 若要让方案 A 在增量解码中缓存 HPLSTM recurrent state，需要后续改造为 dict 形态（见第 6 节）。

### 3.2 方案 B（`Decoder_2.py`）：串联融合（推荐首个 baseline）

路径：

- `attn_out = self_attn(x, ...)`
- `h_out = hplstm(attn_out)`
- `context = attn_out + h_out`
- `context = ff(context)`

特点：

- 结构最简、风险最低、最容易快速复现。
- 常作为第一组实验基线。

状态管理现状：

- 当前 `Decoder_2.py` 仍主要使用 tuple 状态 `(K, V)`，未把 HPLSTM recurrent state 纳入解码缓存。

### 3.3 方案 C（`Decoder_3.py`）：追加 ResHPLSTM 子层

路径：

- `attn_out = self_attn(x, ...)`
- `context = reslstm(attn_out)`
- `context = ff(context)`

特点：

- 模块化最清晰，适合做结构化消融。
- 该方案无顶层 `hplstm` 分支，主训练目标应是 `reslstm`。

状态管理现状：

- 当前 `Decoder_3.py` 也主要是 tuple 状态 `(K, V)` 路线。

---

## 4. 选型建议（实践顺序）

1. 先跑通方案 B（`hplstm_fusion="B"`）作为稳定 baseline。
2. 再做方案 C，用于验证“新增 ResHPLSTM 子层”的结构贡献。
3. 最后尝试方案 A，追求更高上限。

---

## 5. 与 LoRA 的关系（流程相似，机制不同）

流程上可类比：

- 加载基座 -> 冻结主体 -> 只训小模块 -> 仅保存可训练参数。

机制上不等价：

- LoRA：在线性层上做低秩增量。
- HPLSTM：结构级新增模块（参数初始化、状态路径都不同）。

因此不能把 `std2lora` 直接“替换成 HPLSTM 类型”当作同类 patch。

---

## 6. 增量解码状态管理（重要差异与改造方向）

目标：让 HPLSTM 在 `query_unit` 增量解码时真正保留 recurrent state，而不是每步近似重置。

推荐状态结构：

```python
layer_state = {
    "attn": (K, V),
    "hplstm": hp_state,  # 初始可用 "init"
}
```

改造点：

1. `Decoder*.py` 的 `DecoderLayer.forward(query_unit!=None)`：
   - 读 `attn_states_in` 和 `hp_state_in`；
   - 调 `self.hplstm(query_unit, states=hp_state_in)`；
   - 返回 dict 形态状态。
2. `Decoder*.py` 的 `build_states/greedy_decode/beam_decode`：
   - 统一处理 dict 状态；
   - 读取 KV 长度时从 `state["attn"][0]` 取。

当前进度对齐：

- `Decoder_1.py` / `Decoder_2.py` / `Decoder_3.py`：当前都以 tuple KV 为主。
- 如需在方案 A 中真正保留 HPLSTM state，需要把 `Decoder_1` 及其上层状态读写链路一起升级为 dict。

---

## 7. HPLSTM 代码地图（阅读顺序）

1. `modules/hplstm/snbase.py`：本项目实际常用 `HPLSTM` / `ResHPLSTM` 实现入口。
2. `modules/hplstm/base.py`：核心逻辑与接口语义（`states` 行为）。
3. `utils/hplstm/`：底层门控和累积算子（`LGate.py`、`RS1cumsumstatnorm.py` 等）。
4. `utils/cpp/hplstm/`：高性能扩展（一般不需要先改）。

---

## 8. 训练与排错清单

- 确认日志输出的融合方案是否符合预期（A/B/C）。
- 检查 `requires_grad=True` 参数是否集中在 `hplstm` 或 `reslstm`。
- 若使用方案 C，优先 `fine_tune_reslstm=True`，避免只开 `fine_tune_hplstm` 导致“无效解冻”。
- 如果显存紧张，先用方案 B + `USE_AMP=1`。
- 观察保存的 checkpoint：确认 `rgrad_filter` 仅保留可训练参数。

---

## 9. 一句话结论

在本仓库里，HPLSTM 微调的稳定路径是：`train_hplstm_qwen.py` 走冻结/解冻小模块流程，先用方案 B 跑通，再按实验目标切换到 C 或 A；若要让增量解码发挥 HPLSTM 记忆能力，需要把状态管理统一升级到 dict 形态并在所有 `Decoder_*` 路线保持一致。

---

## 10. 是否必须加入 HPLSTM 解码状态（结合当前报错与最小改动）

### 10.1 先回答“有没有必要”

不一定必须，取决于目标：

- **只想先把训练/验证跑通**：不是必须。保持当前 tuple 状态 `(K, V)` 即可。
- **想让方案 A 在增量解码时发挥 HPLSTM 的“时序记忆”能力**：有必要。否则 `hplstm(query_unit)` 每步都不带历史状态，等价于“无 recurrent cache”的近似用法。

简化理解：

- 不加 HPLSTM state：工程更稳、改动更少、先跑通最快；
- 加 HPLSTM state：推理语义更完整，可能提升长序列与持续生成一致性，但改动面更大。

### 10.2 当前 `KeyError: 0` 的根因（`stderr-l1.txt`）

报错堆栈指向：

- `train_hplstm_qwen.py` 的 `eva(...)` 传入了 `states=prepare_states_bsize(prefix_states, ...)`
- `transformer/PLM/LLMDecoder.py` 的 `forward` 里执行：
  - `states.get(0, (None, None,))[0]`

这说明基类 `LLMDecoder.forward` 默认把「每层状态」当成 **tuple `(K,V)`**，并用 `[0]` 取 K。  
若某处把「每层」改成了 dict（如 `{"attn": ..., "hplstm": ...}`）但没有同步改读取链路，`[0]` 会变成对 dict 的键查找 → 典型即 **`KeyError: 0`**（精确机制见 **§11.2**）。

### 10.3 “尽量少改代码”时的建议路径

建议分三档（与 **§11.8** 优劣表一一对应，表中为改法 1 / 4 / 6）：

1. **最小改动（推荐先做）**
   - 保持全链路 tuple 状态（当前代码就是这样）；
   - 不引入 HPLSTM recurrent state；
   - 先验证方案 A/B/C 的训练稳定性与收益。

2. **中等改动（只改方案 A）**
   - 仅把 `Decoder_1.py` 升级到 dict 状态；
   - 同时在 `Decoder_1.Decoder` 内覆盖 `forward/build_states/greedy_decode/beam_decode` 的状态读取，避免依赖 tuple-only 的基类读取逻辑。
   - 这档可以避免改全局基类，但需要保证 `Decoder_1` 内部状态链路自洽。

3. **完整改动（全局统一）**
   - 把 `LLMDecoder.py` 与 `Decoder_1/2/3.py` 统一成支持 dict 状态；
   - 所有融合方案共享同一状态协议；
   - 工程一致性最好，但改动面最大。

### 10.4 结论（面向你当前阶段）

你目前目标是“重新修改代码且尽量少改”，优先建议：

- **先不加 HPLSTM 解码状态**，保持 tuple 路线把训练/验证跑稳；
- 后续若确认要做“增量解码记忆能力”实验，再切到第 2 档（仅 `Decoder_1` 内自洽升级）；
- 避免半改状态结构（只改 `state_return` 不改上层读取），否则很容易再次触发 `KeyError: 0` 这类问题。

---

## 11. HPLSTM 解码状态：原理、怎么改、要动哪些代码

本节把「改 HPLSTM 解码信息（增量解码时的 recurrent cache）」讲清楚：和 `stderr-l1.txt` 里 `LLMDecoder.forward` 的报错如何对应、状态在数学/工程上是什么、建议的改法档位，以及**必须一起改**的文件清单。

### 11.1 两套「状态」不要混：外层 dict vs 每层内部形态

本仓库约定（见 `transformer/PLM/LLMDecoder.py`）：

- **外层**：`states` 是「按层编号索引」的 `dict`，键为 `0 .. num_layer-1`（或与 `enumerate(self.nets)` 一致），值为**该层**在增量解码时要传入 `DecoderLayer.forward(..., query_unit=...)` 的第一个位置参数 `inputo`。
- **当前实现里「每层」的值**：自注意力 KV，**tuple** `(K, V)`（可能含 `None`）。
- **计划中的扩展（第 6 节）**：每层值为 **dict**，例如 `{"attn": (K, V), "hplstm": hp_state}`，其中 `hp_state` 为 HPLSTM 在步进解码时要传入 `HPLSTM.forward(..., states=...)` 的张量结构（见下）。

训练主循环里 `model(oi, ...)` 通常 **不传** `states`；验证函数 `eva()` 在存在固定 `prefix_ids` 时会先 `build_states` 再对回答段 `forward(..., states=prepare_states_bsize(prefix_states, ...))`，因此**验证路径会走 `LLMDecoder.forward` 中带 `states` 的分支**（`train_hplstm_qwen.py` 中 `eva`）。

### 11.2 `KeyError: 0` 的精确根因（与 `stderr-l1.txt` 对齐）

报错位置（`LLMDecoder.forward`）逻辑等价于：

1. 用 `states.get(0, (None, None,))` 取出**第 0 层**的缓存；
2. 立刻对返回值做 **`[0]`**，意图是取 **attention 的 K**（用 `K.size(-1)` 作为已缓存序列长度 `_sid`，再构造因果 mask）。

当「第 0 层」缓存仍是 **tuple** `(K, V)` 时：`[0]` 是 tuple 下标，得到 `K`，行为正确。

当你把某层返回改成了 **dict**（例如 `{"attn": (K, V), "hplstm": ...}`），但 **没有**改 `LLMDecoder`（或其它读 `states` 的地方）时：

- `states.get(0, ...)` 得到的是一个 **Python dict**；
- 对此 dict 写 `[0]` **不是**「取 tuple 第一个元素」，而是 **`dict[0]`——用整数 `0` 当键去查 dict**；
- 若 dict 的键是 `"attn"` / `"hplstm"` 等字符串，则 **`0` 不是合法键 → `KeyError: 0`**。

因此该报错本质是：**「每层状态」从 tuple 升级为 dict 后，仍用「tuple 取 K」的写法去读**，而不是「外层 `states` 缺键 0」（缺键时 `.get(0, (None, None))` 会回落到默认值，不会 KeyError）。

> 推论：要么**全链路**统一新协议并改所有读取点；要么**只在子类 Decoder 内覆盖 `forward/build_states/...`**，不经过基类那段 tuple 假设（对应第 10 节「第 2 档」）。

### 11.3 HPLSTM 在解码步上的状态是什么（原理）

实现见 `modules/hplstm/base.py` 中 `MHPLSTMCore.forward`：

- 单步/多步输入经投影后为 `heads_input`，核心 `MHPLSTMCore` 可在 `states` 为 `None` 或 `"init"` 时用可学习初值展开；
- 当 `states is not None` 且非 `"init"` 时，`forward` 返回 `(out, states_return)`，其中 **`states_return` 为 `(csum_state_return, cell_slice)`**，供下一步作为 `states` 传入，使 **RS 累积与 cell 在时间上连续**，而不是每步重置。

因此「给 HPLSTM 解码状态」的工程含义是：在 **`query_unit is not None`** 的增量路径上，把上一步的 `states_return` 传回本步的 `self.hplstm(query_unit, states=...)`，并把新的 `states_return` 写回「每层缓存」，与 attention 的 `(K, V)` 并行维护。

当前 `Decoder_1.py`（以及 2/3）在 `query_unit` 分支里仍是 `h = self.hplstm(query_unit)` **不传 `states`**，故增量解码时 HPLSTM 每步等价于冷启动（仅 attention KV 在续写）。这是设计上的取舍，不是偶然遗漏。

### 11.4 推荐的状态协议（与第 6 节一致，细化字段）

建议固定每层结构（示例）：

```python
# 外层：states[layer_idx] -> layer_pack
layer_pack = {
    "attn": (K, V),           # 与 ResSelfAttn 增量接口一致
    "hplstm": hp_state,       # None / "init" / (csum_state_return, cell_tensor)
}
```

约定：

- **`build_states`（整段 prefix）**：对每一解码层，在 `query_unit` 循环中更新 `layer_pack["attn"]`；HPLSTM 若也要在 prefix 上累积，可在整段上 `states=None` 跑完最后一步再取 `states_return` 写入 `layer_pack["hplstm"]`，或在块循环内逐步传递（与实现细节一致即可），关键是 **进入第一个回答 token 前，`layer_pack` 已与 prefix 对齐**。
- **自回归步**：`DecoderLayer.forward` 收到 `inputo=layer_pack`，拆出 `attn` 给 `self_attn`，拆出 `hplstm` 给 `hplstm`；返回新的 `layer_pack`。

### 11.5 怎么改：分档（与第 10 节对应，落实为检查表）

**档 1 — 不引入 HPLSTM recurrent state（先跑通）**

- 保持每层 `(K, V)`；
- 不要只改 `DecoderLayer` 返回值而不改读取方。

**档 2 — 仅 Qwen 某融合 Decoder 自洽（例如只动方案 A）**

- 在 `transformer/PLM/QWen/v3/Decoder_1.py` 的 `Decoder` 子类中 **覆盖** `LLMDecoder.forward`（可复制基类后改「取 K 长度」与 `for net in self.nets` 传参/拆包）；
- 同步改同一文件内的 `build_states`、`greedy_decode`、`beam_decode` 里所有 `_states.get(_tmp, (None, None,))[0]` 或把 `inputo` 当作纯 tuple 的地方；
- 这样可不修改共享基类 `LLMDecoder.py`，但 **方案 B/C 若仍继承基类 `forward`，行为与 A 可能不一致**，需心里有数。

**档 3 — 全局统一（长期维护最省心）**

- 修改 `transformer/PLM/LLMDecoder.py`：`forward`、`build_states` 中凡用 `states.get(i, ...)[0]` 推断序列长度、或把 `states[i]` 直接传入 layer 的地方，改为通过小工具函数解析，例如「若为 tuple 则视为 `(K,V)`；若为 dict 则取 `['attn'][0]`」；
- 修改 `transformer/PLM/QWen/v3/Decoder_1.py` / `Decoder_2.py` / `Decoder_3.py` 的 `DecoderLayer.forward`：在 `query_unit is not None` 分支为 `hplstm` 接入 `states` 与返回值组装；
- 各文件 `Decoder.build_states` / `greedy_decode` / `beam_decode`：凡假设 `_state` 为 tuple 的赋值与索引，改为读写 `layer_pack`；
- **验证**：`adv/train/plm/train_hplstm_qwen.py` 的 `eva()`、`prepare_states_bsize`（`utils/plm/inference.py`）——该函数已对 dict/tuple 递归 `expand_bsize`，一般**无需改**，前提是「最外层仍是 `dict[int, layer_pack]`」且 `layer_pack` 内张量 batch 维可扩。

### 11.6 联动代码清单（按依赖顺序）

| 区域 | 文件 | 需要关注的原因 |
|------|------|------------------|
| 取 KV 长度 / mask | `transformer/PLM/LLMDecoder.py` | `forward`、`build_states` 用 `[0]` 从「每层状态」取 K；tuple→dict 后必改或覆盖 `forward` |
| 融合层前向 + 返回状态 | `transformer/PLM/QWen/v3/Decoder_1.py`（及 2/3 若统一协议） | `DecoderLayer.forward`：`self_attn` 的 `states` 与 `hplstm` 的 `states` 拆包/打包 |
| Prefix 与解码 | 同上 `Decoder.build_states`、`greedy_decode`、`beam_decode` | 多处 `_states.get(i, (None, None,))[0]`；beam 用 `utils/decode/beam.py` 的 `expand_bsize_for_beam`（已支持 dict 递归） |
| HPLSTM 语义 | `modules/hplstm/base.py`、`modules/hplstm/snbase.py` | `states` / `states_return` 形状与 `None`、`"init"` 语义 |
| 验证 batch 扩维 | `utils/plm/inference.py` 中 `prepare_states_bsize`、`expand_bsize` | 已递归 dict/tuple；外层键仍为层号即可 |
| 训练验证入口 | `adv/train/plm/train_hplstm_qwen.py` | `eva` 传 `states=`；改协议后跑一遍 prefix + 回答切分 |
| 其它推理脚本 | `adv/predict/plm/qwen/` 等（若直接 `decode`/`build_states`） | 与训练侧同一套 `states` 协议 |

### 11.7 自测建议

1. **无 prefix**：`states=None` 的 `forward` 应与改前数值行为一致（回归）。
2. **有 prefix、仅 attention tuple**：与当前基线对比 loss/误差。
3. **有 prefix、tuple+dict 全链路**：单 batch 与 `prepare_states_bsize` 扩 batch 后各跑一步，确认无 `KeyError` 且 shape 连续。

完成上述内容后，第 6 节的「改造方向」与第 10 节的排错可对照本节作为实施清单使用。

### 11.8 改法有几种？优劣对照（决策用）

下面按「是否给 HPLSTM 传增量 `states`、改哪些文件」划分常见路线。**编号与 §10.3 三档对应关系**：改法 1 ≈ 档 1；改法 4 ≈ 档 2；改法 6 ≈ 档 3。中间几档是实践中常见的变体。

| 编号 | 改法概要 | 优点 | 缺点 / 风险 |
|------|----------|------|----------------|
| **1** | **维持现状**：每层仍是 `(K,V)`，增量路径 `hplstm(query_unit)` 不传 `states`。 | 与当前 `LLMDecoder` / 训练 / `eva` 完全一致；无 `KeyError` 类协议问题；排错成本最低。 | 增量解码下 HPLSTM 无跨步记忆，与「整段 teacher-forcing 训练」的时序行为不一致，长续写可能弱。 |
| **2** | **基类只做「读状态」兼容**：在 `LLMDecoder.py` 增加小工具（如 `layer_attn_kv(pack)`：`tuple` 走 `[0]`，`dict` 走 `["attn"][0]`），`forward` / `build_states` 统一经此取 K、传层。 | 一处收敛「如何避免 dict 上误用 `[0]`」；子类返回 `layer_pack` 后，**未**覆盖 `forward` 的 Decoder 也可能立刻不炸。 | 仍须各 `DecoderLayer` / `build_states` **写出** `layer_pack`；未改写的层若混用 tuple/dict，调试要盯紧；多一层间接调用，可读性略降。 |
| **3** | **Prefix 内全序列 HPLSTM、回答段增量仍不传 `states`**：`build_states` 跑 prefix 时对 HPLSTM 用 `query_unit is None` 的整段路径，不把 `states_return` 带入后续自回归步。 | 实现量小于「全增量带 state」；prefix 段与训练时「见全长」更接近。 | 回答第一个 token 起 HPLSTM 仍步步冷启动，语义仍是**半连贯**；若实验目标是「全对话级记忆」，不够用。 |
| **4** | **单融合文件自洽（典型：仅 `Decoder_1`）**：在 `Decoder_1.Decoder` 中覆盖 `forward`、`build_states`、`greedy_decode`、`beam_decode`，内部使用 `layer_pack`，不依赖基类对 tuple 的假设。 | 不动 `LLMDecoder` 时影响面小；适合只对方案 A 做实验；与方案 B/C 隔离。 | B/C 仍走基类，**三方案验证集 / 推理语义不一致**；复制基类逻辑易产生后续合并冲突；beam 路径必须与 `expand_bsize_for_beam` 一致测过。 |
| **5** | **用 `namedtuple` / 轻量数据类代替 `dict` 表示每层**（字段如 `attn`、`hplstm`），读 K 时用属性而非 `[0]`。 | 避免「`dict` 被误当下标」类 bug；IDE 补全更好。 | 与现有大量 `dict` 递归工具（`expand_bsize` 等）要确认是否仍按 dict 处理；若工具只认 `dict`，可能要注册序列化或转 dict。 |
| **6** | **全局统一协议**：`LLMDecoder` + `Decoder_1/2/3` 的 `DecoderLayer` 与各类 `build_states` / 解码循环全部使用同一套 `layer_pack`（或改法 2 + 全层返回 dict）。 | 任意融合方案、训练 / 验证 / `greedy_decode` / `beam_decode` 行为一致；长期维护成本最低；新脚本不易踩 tuple/dict 混用。 | 首次改动 diff 大、回归测试面最广；需统一决定 B/C 中 `reslstm` 是否也进 `layer_pack`（若也要增量 state，范围再扩）。 |

**选型提示（简短）**

- 先要 **稳定跑通训练与带 prefix 的验证**：优先 **改法 1**。
- 已确定只在 **方案 A** 上试 HPLSTM 记忆、且想少动公共基类：**改法 4**，必要时配合 **改法 2** 减少重复解析代码。
- 多方案横向对比、或团队多人改 PLM：**改法 6**（或 **2 + 6** 分步：先 2 再全层返回 dict）。

