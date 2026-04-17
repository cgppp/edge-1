# 四数据集 LoRA 预测流程（Qwen3 基座）

本文将原 `SQUAD_LORA_PREDICT_FLOW.md` 扩展为统一流程，覆盖以下 4 个数据集：

- `pubdatasets_metamathqa`
- `pubdatasets_squad_closed`
- `pubdatasets_nq_closed`
- `pubdatasets_toolbench`

适用项目：`/home/gpchen/lora/transformer-edge`

---

## 1. 流程总览

LoRA 预测的标准链路：

1. 选定数据集与 LoRA checkpoint（如 `last.h5` 或 `eva_*.h5`）
2. 用 `tools/h5/lora.py` 将 LoRA 参数并入基座，得到 `merge.h5`
3. 用 `tools/plm/llmdec/mktest.py` 生成预测专用 `test.h5`
4. 运行 `adv/predict/plm/qwen/predict.py`

---

## 2. 关键脚本说明

- `tools/h5/lora.py`  
将 LoRA 参数并入基座权重，输出可直接推理的完整模型。
- `tools/plm/llmdec/mktest.py`  
从 `src.*.ids` 生成推理用 `test.h5`。
- `adv/predict/plm/qwen/predict.py`  
执行解码预测，输出文本结果。
- `tools/h5/convert/st.py`（可选）  
`safetensors` 与 `h5` 互转。

---

## 3. 输入与目录约定

### 3.1 必备输入

1. **基座权重**（示例）
  `/home/common/plm/Qwen/Qwen3-8B/model.h5`
2. **LoRA checkpoint**（训练产物）
  如：`expm/llm/<DATASET>/std/base/last.h5` 或 `eva_*.h5`
3. **tokenizer 目录**
  如：`/home/common/plm/Qwen/Qwen3-8B`
4. **预测用 test.h5**（由 `mktest.py` 生成）

### 3.2 四数据集目录映射


| 数据集          | DATA_ID                        | 工作目录 WKD                             | checkpoint 目录                                |
| ------------ | ------------------------------ | ------------------------------------ | -------------------------------------------- |
| MetaMathQA   | `llm/pubdatasets_metamathqa`   | `cache/llm/pubdatasets_metamathqa`   | `expm/llm/pubdatasets_metamathqa/std/base`   |
| SQuAD closed | `llm/pubdatasets_squad_closed` | `cache/llm/pubdatasets_squad_closed` | `expm/llm/pubdatasets_squad_closed/std/base` |
| NQ closed    | `llm/pubdatasets_nq_closed`    | `cache/llm/pubdatasets_nq_closed`    | `expm/llm/pubdatasets_nq_closed/std/base`    |
| ToolBench    | `llm/pubdatasets_toolbench`    | `cache/llm/pubdatasets_toolbench`    | `expm/llm/pubdatasets_toolbench/std/base`    |


---

## 4. 先确认可用 LoRA 权重

```bash
cd /home/gpchen/lora/transformer-edge
ls -lh expm/llm/pubdatasets_*/std/base/*.h5
```

推荐优先级：

1. `eva_*.h5`（验证指标最优）
2. `last.h5`（最容易复现实验）

---

## 5. 合并基座 + LoRA

```bash
cd /home/gpchen/lora/transformer-edge

BASE="/home/common/plm/Qwen/Qwen3-8B/model.h5"
LORA="expm/llm/pubdatasets_toolbench/std/base/eva_55_2.266_2.304_21.63.h5"
MERGE="expm/llm/pubdatasets_toolbench/std/base/merge_last.h5"

python tools/h5/lora.py "$BASE" "$LORA" "$MERGE"
ls -lh "$MERGE"
```

只需替换 `LORA` 和 `MERGE` 路径即可切到其它数据集。

---

## 6. 生成预测用 `test.h5`（不要用 `dev.h5` 冒充）

`predict.py` 预期读取的是推理格式的 `test.h5`。  
训练/验证使用的 `train.h5`、`dev.h5` 与推理侧 `tgt` 语义不同，直接复制会导致解码阶段形状/长度不匹配。

### 6.1 生成命令

```bash
cd /home/gpchen/lora/transformer-edge
WKD="cache/llm/pubdatasets_squad_closed"

# 单卡通常第三个参数写 1
# 评测对齐建议：使用未排序 src.dev.txt.ids 生成 test.h5，保证与 tgt.dev.txt 行顺序一致
python tools/plm/llmdec/mktest.py "$WKD/src.dev.txt.ids" "$WKD/test.h5" 1
```

说明：

- 推荐直接用 `src.dev.txt.ids`（未排序），便于与 `tgt.dev.txt` 逐行对齐评测
- 四个数据集的命令完全一致，只改 `WKD`

---

## 7. 执行预测

```bash
cd /home/gpchen/lora/transformer-edge

OUT="expm/llm/pubdatasets_squad_closed/std/base/pred_last.txt"
TOKENIZER="/home/common/plm/Qwen/Qwen3-8B"
MODEL="expm/llm/pubdatasets_squad_closed/std/base/merge_last.h5"

python adv/predict/plm/qwen/predict.py "$OUT" "$TOKENIZER" "$MODEL"
```

---

## 8. 四数据集一键模板

```bash
cd /home/gpchen/lora/transformer-edge

DATASET="pubdatasets_squad_closed"   # 可改为 pubdatasets_metamathqa / pubdatasets_nq_closed / pubdatasets_toolbench
BASE="/home/common/plm/Qwen/Qwen3-8B/model.h5"
TOKENIZER="/home/common/plm/Qwen/Qwen3-8B"

WKD="cache/llm/$DATASET"
CKPT_DIR="expm/llm/$DATASET/std/base"
LORA="$CKPT_DIR/last.h5"
MERGE="$CKPT_DIR/merge_last.h5"
OUT="$CKPT_DIR/pred_last.txt"

python tools/h5/lora.py "$BASE" "$LORA" "$MERGE" && \
python tools/plm/llmdec/mktest.py "$WKD/src.dev.txt.ids" "$WKD/test.h5" 1 && \
python adv/predict/plm/qwen/predict.py "$OUT" "$TOKENIZER" "$MERGE"
```

---

## 9. 常见问题

### 9.1 预测报 shape/view 错误

优先检查是否错误使用了 `dev.h5` 作为 `test.h5`。  
正确做法是重新运行 `mktest.py` 生成 `test.h5`。

### 9.2 输出为空或异常短

- 检查 `DATA_ID` 对应目录下是否存在正确的 `test.h5`
- 检查 `MERGE` 是否成功生成、文件体积是否正常
- 检查 tokenizer 是否与基座模型族一致

### 9.3 想对比基座与 LoRA

直接把 `MODEL` 分别指向：

- 基座：`/home/common/plm/Qwen/Qwen3-8B/model.h5`
- LoRA 合并后：`merge_*.h5`

即可做 A/B 对比。