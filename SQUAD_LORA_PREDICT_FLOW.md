# SQuAD Closed 数据集 LoRA 预测流程（Qwen3-8B 基座）

本文面向当前项目 `lora/transformer-edge`，说明如何把**训练得到的 LoRA 权重**用于数据集预测。

你的基座权重路径是：

- `/home/common/plm/Qwen/Qwen3-8B/model.h5`

当前目录现状（已检查）：

- `expm/llm/pubdatasets_squad_closed/std/base` 下已有以下文件：
  - `base.h5`（约 12G）
  - `eva_11_3.198_3.154_34.89.h5`（约 21M）
  - `init.h5`（约 21M）
  - `last.h5`（约 21M）
  - `train_19_2.981_3.325_35.67.h5`（约 21M）
  - `train.log`
- 其中常用预测候选：
  - 稳妥默认：`last.h5`
  - 按验证效果挑选：`eva_11_3.198_3.154_34.89.h5`

---

## 1) 先明确四个关键文件/脚本角色

- `tools/h5/lora.py`
  - 用于把 LoRA 参数并回基座权重，输出一个可直接推理的 `merge.h5`
- `tools/h5/convert/st.py`
  - `safetensors` 和 `h5` 格式互转（可选）
- `adv/predict/plm/qwen/predict.py`
  - 真正做推理（读取 `cnfg.base.test_data`；`DATA_ID` 下优先同目录 **`test.h5`**，见第 5 节）
- `scripts/plm/llm/mktest.sh`
  - 示例总控脚本：构造测试数据、调用 `predict.py`、恢复顺序

---

## 2) 你需要准备的输入

1. **基座权重**（你已具备）
  `/home/common/plm/Qwen/Qwen3-8B/model.h5`
2. **LoRA 权重文件**（训练产物）
  例如某个 `eva_*.h5` 或 `last.h5`（取决于训练脚本保存策略）
3. **用于预测的 H5**
  须使用 **`tools/plm/llmdec/mktest.py` 生成的 `test.h5`**（见下文第 5 节）。  
  **不要**把 `dev.h5` 复制改名为 `test.h5`：二者 `tgt` 语义不同，复制后格式仍不对。
4. **tokenizer 目录**
  例如：`/home/common/plm/Qwen/Qwen3-8B`（按你实际目录）

---

## 3) 第一步：定位 LoRA 权重文件

先看训练目录里有哪些 `.h5`：

```bash
cd /home/gpchen/lora/transformer-edge
ls -lh expm/llm/pubdatasets_squad_closed/std/base/*.h5
```

你当前目录里已有 `.h5`，可直接进入“合并 + 预测”步骤。

---

## 4) 第二步：合并 LoRA + 基座为可推理权重

> 重点：`predict.py` 直接加载的是“完整模型权重”。  
> 对 LoRA-only checkpoint，推荐先合并成单个 `merge.h5` 再预测。

示例（把 LoRA 文件并入基座）：

```bash
cd /home/gpchen/lora/transformer-edge

BASE="/home/common/plm/Qwen/Qwen3-8B/model.h5"
LORA="expm/llm/pubdatasets_squad_closed/std/base/eva_11_3.198_3.154_34.89.h5"
MERGE="expm/llm/pubdatasets_squad_closed/std/base/merge_last.h5"

python tools/h5/lora.py "$BASE" "$LORA" "$MERGE"
```

合并后检查：

```bash
ls -lh "$MERGE"
```

---

## 5) 第三步：生成预测用 `test.h5`（必做：勿用 `dev.h5` 冒充）

### 5.1 为什么不能把 `dev.h5` 当 `test.h5` 用（复制也不行）

- **`train.h5` / `dev.h5`** 由 `tools/plm/mkiodata_llmdec.py` 生成，约定见 `mkiodata_llmdec.py` 文件头：`tgt` 每个 batch 的形状是 **`(batch_size, 2)`**，每行为 **`[lid, lgth]`**（回答在「指令+回答」拼接序列中的区间），供训练时 **mask** 用。
- **`adv/predict/plm/qwen/predict.py`** 把 `tgt` 读入为 `seq_o`，并作为 **`ilen=`** 传给解码器；`beam_decode` 里要求 **`ilen` 与 batch 一一对应的长度标量**（与 `utils/fmt/plm/llmdec/single.py` + `tools/plm/llmdec/mktest.py` 一致）。
- 若把 `dev.h5` **复制**成 `test.h5`，`tgt` 仍是 **`(B, 2)`**，不会在复制时变成「每样本一个长度」，因此在 beam 等路径下会出现类似 **`view(bsize, 1, 1)` 与元素个数不匹配**（例如 `42` vs `84`）的报错。

结论：**需要单独用 `mktest.py` 生成预测用 `test.h5`**，而不是复制 `dev.h5`。

### 5.2 预测输入用哪份 `.ids`

预测只应喂 **指令侧**（与训练时 `instruct_auto` 映射一致），即已有的：

- `cache/llm/pubdatasets_squad_closed/src.dev.srt.ids`（若做过 sort），或  
- `src.dev.txt.ids`（未排序时）

**不要**用 `mkiodata` 里那种「指令 token + 回答 token」拼好的长序列来跑 `mktest.py`；那是 `dev.h5` 里 `src` 的形态，用于训练，不是本仓库 `predict.py` + `mktest.py` 这条推理链的输入。

### 5.2.1 与 `LORA_DATA_FORMAT_AND_STEPS.md` 第三部分（dev.h5）是否一致

| 步骤 | 生成 **dev.h5**（第三部分） | 生成 **test.h5** 之前 |
| --- | --- | --- |
| **3.2 文本** | `src.dev.txt` + `tgt.dev.txt` | 与 dev **同源**：推理只用 **`src.dev.txt`**（同一文件，无需另导出） |
| **3.3 map** | 四条：`src/tgt` × `train/dev`，其中 dev 为 `src.dev`→`instruct_auto`，`tgt.dev`→`assistant` | **指令侧与 dev 完全一致**：`src.dev.txt` → `src.dev.txt.ids`，模板 **`instruct_auto`**。**不要求**把 `tgt` 喂给 `mktest.py`；但若下一步要做 **sort**，见下表 |
| **3.4 sort** | `sort.py` **同时读入两个等行数文件**（`src.dev.txt.ids` + `tgt.dev.txt.ids`），写出成对的 `.srt.ids` | 与 dev **命令相同**（否则无法按现有 `sort.py` 保持行对齐）。**只取输出里的** `src.dev.srt.ids` 给 `mktest.py`，`tgt.dev.srt.ids` 可忽略 |

**结论（简要）**：

- **与第三部分「完全一致」的部分**：**`src.dev` 的文本 → `instruct_auto` map**；若要做排序，**sort 命令行与 dev 相同**（成对 sort），只是 `mktest` **只用** `src.dev.srt.ids`。
- **与第三部分「不一致 / 可省」的部分**：**不必**为推理再 map 一次 `tgt.dev`，**除非**你要走 **sort.py**（双文件成对排序，需要已有 `tgt.dev.txt.ids`）。若你**已经**按第三部分完整跑过 dev，则 `src.dev.srt.ids` 现成可用。
- **最小流程**（不排序）：只需 **`src.dev.txt` → map（仅 src）→ `mktest.py`**，不必生成 `tgt.dev.txt.ids`。对应第三部分里「未排序则用 `.txt.ids`」的写法。

### 5.3 生成 `test.h5` 的命令（SQuAD）

前置：至少已有 **`src.dev.txt.ids`**；若要与第三部分一致并带排序，则还需按第三部分完成 **成对 sort**，得到 **`src.dev.srt.ids`**（见 5.2.1）。

```bash
cd /home/gpchen/lora/transformer-edge
export WKD=cache/llm/pubdatasets_squad_closed

# 第三个参数为 ngpu 分片数，单卡写 1 即可
python tools/plm/llmdec/mktest.py "$WKD/src.dev.srt.ids" "$WKD/test.h5" 1
```

脚本会打印 `Number of batches: ...`，并在 `WKD/test.h5` 中写入与 `predict.py` 兼容的 `src`/`tgt`/`ndata`。

### 5.4 运行预测时如何选中 `test.h5`

设置 `DATA_ID=llm/pubdatasets_squad_closed` 时，`predict.py` 会读取：

`cache/llm/pubdatasets_squad_closed/test.h5`  

（`cnfg/base.py` 中 `test_data` 亦指向各 `data_id` 目录下的 **`test.h5`**。）须先按 5.3 生成；**不要**用 `dev.h5` 代替。

### 5.5（可选）与 `scripts/plm/llm/mktest.sh` 的关系

仓库里的 `mktest.sh` 演示了「文本 → map → sort → `mktest.py` → `predict.py` → `restore.py`」整条流水线。你已具备 `src.dev.srt.ids` 时，只需执行 **5.3** 一步即可得到 SQuAD 的 `test.h5`。

---

## 6) 第四步：运行预测

`predict.py` 用法：

```bash
python adv/predict/plm/qwen/predict.py <输出文件> <tokenizer路径> <模型h5>
```

示例：

```bash
cd /home/gpchen/lora/transformer-edge

OUT="expm/llm/pubdatasets_squad_closed/std/base/pred_last.txt"
TOKENIZER="/home/common/plm/Qwen/Qwen3-8B"
MODEL="expm/llm/pubdatasets_squad_closed/std/base/merge_last.h5"

python adv/predict/plm/qwen/predict.py "$OUT" "$TOKENIZER" "$MODEL"
```

---

## 7) 推荐的一次性命令模板

```bash
cd /home/gpchen/lora/transformer-edge && \
BASE="/home/common/plm/Qwen/Qwen3-8B/model.h5" && \
LORA="expm/llm/pubdatasets_squad_closed/std/base/last.h5" && \
MERGE="expm/llm/pubdatasets_squad_closed/std/base/merge_last.h5" && \
OUT="expm/llm/pubdatasets_squad_closed/std/base/pred_last.txt" && \
TOKENIZER="/home/common/plm/Qwen/Qwen3-8B" && \
python tools/h5/lora.py "$BASE" "$LORA" "$MERGE" && \
python adv/predict/plm/qwen/predict.py "$OUT" "$TOKENIZER" "$MERGE"
```

如果你想对比 “best eva” 权重，可把 `LORA` 改成：

```bash
LORA="expm/llm/pubdatasets_squad_closed/std/base/eva_11_3.198_3.154_34.89.h5"
MERGE="expm/llm/pubdatasets_squad_closed/std/base/merge_eva11.h5"
OUT="expm/llm/pubdatasets_squad_closed/std/base/pred_eva11.txt"
```

---

## 8) 常见问题与排查

### A. 结合 `train_lora_qwen.py`，如何选预测用的 h5

`train_lora_qwen.py` 的保存逻辑（LoRA 训练模式）可归纳为：

- `base.h5`：训练开始时保存的“基础模型快照”（`lcnfg.save_base` 控制），不是最佳 LoRA 结果；
- `init.h5`：初始化后（或加载后）的起点快照；
- `eva_*.h5`：当验证指标改善时保存（`vprec <= minerr` 或 `vloss <= minloss`）；
- `train_*.h5`：训练损失改善时保存；
- `last.h5`：训练收尾时最后一步保存。

并且在 LoRA 模式下保存时使用 `ps_func=rgrad_filter`，意味着这些 `eva_*/train_*/last.h5` 主要是“可训练参数子集（LoRA 分支等）”，通常需要和基座 `model.h5` 一起用 `tools/h5/lora.py` 合并后再预测。

推荐顺序：

1. **优先 `eva_*.h5`**（更接近验证集最优）
2. 其次 `last.h5`（复现实验最方便）
3. `train_*.h5` 一般用于训练过程对比，不作为首选部署点

- `last.h5`：最后一步权重，复现实验最方便（推荐先用）
- `eva_*.h5`：按验证指标保存，常用于“最佳点”推理对比

### B. `predict.py` 结果异常或为空

- 检查 `DATA_ID` 与 `cache/<DATA_ID>/test.h5` 是否已按第 5 节生成；勿用 `dev.h5` 冒充 `test.h5`
- 检查 tokenizer 路径是否与训练使用的模型族一致
- 检查 `merge.h5` 是否成功生成且体积正常

### C. 只想测试基座效果

可直接：

```bash
python adv/predict/plm/qwen/predict.py "$OUT" "$TOKENIZER" "/home/common/plm/Qwen/Qwen3-8B/model.h5"
```

---

## 9) 你当前最关键下一步

你可以直接跑（以 `last.h5` 为例）：

```bash
cd /home/gpchen/lora/transformer-edge && \
python tools/h5/lora.py \
  /home/common/plm/Qwen/Qwen3-8B/model.h5 \
  expm/llm/pubdatasets_squad_closed/std/base/last.h5 \
  expm/llm/pubdatasets_squad_closed/std/base/merge_last.h5 && \
python adv/predict/plm/qwen/predict.py \
  expm/llm/pubdatasets_squad_closed/std/base/pred_last.txt \
  /home/common/plm/Qwen/Qwen3-8B \
  expm/llm/pubdatasets_squad_closed/std/base/merge_last.h5
```

