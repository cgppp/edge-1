# 四个数据集训练后预测与评测说明（Qwen3 + LoRA）

本文面向你当前这套脚本与代码，说明四个数据集在**训练完成后**如何做预测与离线评测，以及每一步实际调用了哪些代码文件。

适用数据集（与 `task-1.sh` ~ `task-4.sh` 对应）：

- `llm/pubdatasets_squad_closed`
- `llm/pubdatasets_toolbench`
- `llm/pubdatasets_nq_closed`
- `llm/pubdatasets_metamathqa`

---

## 1. 总体流程（训练后）

训练完成后，推荐流程是：

1. 准备好预测输入（`cache/<DATA_ID>/test.h5`）与可选评测标注（`cache/<DATA_ID>/tgt.dev.txt`）。
2. 运行对应预测脚本：`task-p1.sh` / `task-p2.sh` / `task-p3.sh` / `task-p4.sh`。
3. 脚本先调用 `adv/predict/plm/qwen/predict.py` 生成 `pred_last.txt`。
4. 若 `SKIP_EVAL=0`，再调用 `tools/eval/run_eval.py` 做离线评测，并输出 `eval_detail.jsonl`。

核心关系如下：

- Shell 编排：`task-p*.sh`
- 模型推理：`adv/predict/plm/qwen/predict.py`
- 解码实现：`transformer/PLM/QWen/v3/Decoder.py`
- 评测入口：`tools/eval/run_eval.py`
- 评分规则：`tools/eval/scorers.py`

---

## 2. 四个预测脚本对应关系

### `task-p1.sh`（SQuAD Closed）

- 默认 `DATA_ID=llm/pubdatasets_squad_closed`
- 默认 `TASK=squad`
- 默认评测开启（`SKIP_EVAL=0`）

### `task-p2.sh`（ToolBench）

- 默认 `DATA_ID=llm/pubdatasets_toolbench`
- 默认 `TASK=tool`
- 默认评测开启（`SKIP_EVAL=0`）

### `task-p3.sh`（NQ Closed）

- 默认 `DATA_ID=llm/pubdatasets_nq_closed`
- 默认 `TASK=nq_v1`
- 默认评测开启（`SKIP_EVAL=0`）

### `task-p4.sh`（MetaMathQA）

- 默认 `DATA_ID=llm/pubdatasets_metamathqa`
- 默认 `TASK=math`（仅在你显式启用评测时生效）
- 默认跳过评测（`SKIP_EVAL=1`）
- 原因：该数据集常见 `tgt.dev.txt` 与 `run_eval.py` 的 `math/gsm8k` 期望 gold 格式不完全对齐

---

## 3. 运行命令（推荐）

在项目根目录：

```bash
source /home/gpchen/lora/transformer-edge/init_conda.sh
conda activate lora-edge
cd /home/gpchen/lora/transformer-edge
```

### SQuAD

```bash
bash task-p1.sh
```

### ToolBench

```bash
bash task-p2.sh
```

### NQ

```bash
bash task-p3.sh
```

### MetaMath（仅预测）

```bash
bash task-p4.sh
```

### MetaMath（你有可对齐 gold 时启用评测）

```bash
SKIP_EVAL=0 TASK=math GOLD_FILE=cache/llm/pubdatasets_metamathqa/tgt.dev.txt bash task-p4.sh
```

---

## 4. 输入与输出文件约定

以 `_DATA_TAG=${DATA_ID##*/}` 为例（例如 `pubdatasets_squad_closed`）：

- 预测输入：
  - `cache/<DATA_ID>/test.h5`（由前处理流程生成）
- 预测输出：
  - `expm/llm/<_DATA_TAG>/std/base/pred_last.txt`
- 模型权重（默认）：
  - `expm/llm/<_DATA_TAG>/std/base/merge_last.h5`
- 评测 gold（默认）：
  - `cache/llm/<_DATA_TAG>/tgt.dev.txt`
- 评测明细（可选输出）：
  - `expm/llm/<_DATA_TAG>/std/base/eval_detail.jsonl`

说明：

- `task-p*.sh` 默认用 `merge_last.h5`。如果你只想验证训练末权重，也可覆盖：
  - `MODEL=expm/llm/<_DATA_TAG>/std/base/last.h5 bash task-pX.sh`
- 预测脚本已经支持读取环境变量 `DATA_ID` 来定位 `test.h5`，与 `task-p*.sh` 中的 `DATA_ID` 保持一致。

---

## 5. 代码级流程说明（按调用顺序）

## 5.1 Shell 层：`task-p*.sh`

统一做这些事：

1. 激活环境、设置 `PYTHONPATH`
2. 设置默认参数：`DATA_ID`、`OUT`、`TOKENIZER`、`MODEL`、`CUDA_VISIBLE_DEVICES`
3. 调用：
   - `python adv/predict/plm/qwen/predict.py "$OUT" "$TOKENIZER" "$MODEL"`
4. 根据 `SKIP_EVAL` 决定是否调用：
   - `python tools/eval/run_eval.py --task ... --pred ... --gold ... --detail ...`

### 5.1.1 `task-p*.sh` 中的实际命令行（按脚本写法）

下面命令与脚本内核心调用保持同一形式，便于直接对照：

#### `task-p1.sh`（SQuAD）

```bash
export DATA_ID="${DATA_ID:-llm/pubdatasets_squad_closed}"
_DATA_TAG="${DATA_ID##*/}"
export OUT="${OUT:-expm/llm/${_DATA_TAG}/std/base/pred_last.txt}"
export TOKENIZER="${TOKENIZER:-/home/common/plm/Qwen/Qwen3-8B}"
export MODEL="${MODEL:-expm/llm/${_DATA_TAG}/std/base/merge_last.h5}"

python adv/predict/plm/qwen/predict.py "$OUT" "$TOKENIZER" "$MODEL"

TASK="${TASK:-squad}"
PRED_FILE="${PRED_FILE:-$OUT}"
GOLD_FILE="${GOLD_FILE:-cache/llm/${_DATA_TAG}/tgt.dev.txt}"
DETAIL_FILE="${DETAIL_FILE:-expm/llm/${_DATA_TAG}/std/base/eval_detail.jsonl}"
python tools/eval/run_eval.py --task "$TASK" --pred "$PRED_FILE" --gold "$GOLD_FILE" --detail "$DETAIL_FILE"
```

#### `task-p2.sh`（ToolBench）

```bash
export DATA_ID="${DATA_ID:-llm/pubdatasets_toolbench}"
_DATA_TAG="${DATA_ID##*/}"
export OUT="${OUT:-expm/llm/${_DATA_TAG}/std/base/pred_last.txt}"
export TOKENIZER="${TOKENIZER:-/home/common/plm/Qwen/Qwen3-8B}"
export MODEL="${MODEL:-expm/llm/${_DATA_TAG}/std/base/merge_last.h5}"

python adv/predict/plm/qwen/predict.py "$OUT" "$TOKENIZER" "$MODEL"

TASK="${TASK:-tool}"
PRED_FILE="${PRED_FILE:-$OUT}"
GOLD_FILE="${GOLD_FILE:-cache/llm/${_DATA_TAG}/tgt.dev.txt}"
DETAIL_FILE="${DETAIL_FILE:-expm/llm/${_DATA_TAG}/std/base/eval_detail.jsonl}"
python tools/eval/run_eval.py --task "$TASK" --pred "$PRED_FILE" --gold "$GOLD_FILE" --detail "$DETAIL_FILE"
```

#### `task-p3.sh`（NQ）

```bash
export DATA_ID="${DATA_ID:-llm/pubdatasets_nq_closed}"
_DATA_TAG="${DATA_ID##*/}"
export OUT="${OUT:-expm/llm/${_DATA_TAG}/std/base/pred_last.txt}"
export TOKENIZER="${TOKENIZER:-/home/common/plm/Qwen/Qwen3-8B}"
export MODEL="${MODEL:-expm/llm/${_DATA_TAG}/std/base/merge_last.h5}"

python adv/predict/plm/qwen/predict.py "$OUT" "$TOKENIZER" "$MODEL"

TASK="${TASK:-nq_v1}"
PRED_FILE="${PRED_FILE:-$OUT}"
GOLD_FILE="${GOLD_FILE:-cache/llm/${_DATA_TAG}/tgt.dev.txt}"
DETAIL_FILE="${DETAIL_FILE:-expm/llm/${_DATA_TAG}/std/base/eval_detail.jsonl}"
python tools/eval/run_eval.py --task "$TASK" --pred "$PRED_FILE" --gold "$GOLD_FILE" --detail "$DETAIL_FILE"
```

#### `task-p4.sh`（MetaMathQA）

```bash
export DATA_ID="${DATA_ID:-llm/pubdatasets_metamathqa}"
_DATA_TAG="${DATA_ID##*/}"
export OUT="${OUT:-expm/llm/${_DATA_TAG}/std/base/pred_last.txt}"
export TOKENIZER="${TOKENIZER:-/home/common/plm/Qwen/Qwen3-8B}"
export MODEL="${MODEL:-expm/llm/${_DATA_TAG}/std/base/merge_last.h5}"

python adv/predict/plm/qwen/predict.py "$OUT" "$TOKENIZER" "$MODEL"

# task-p4.sh 默认 SKIP_EVAL=1，若开启评测时的调用如下：
TASK="${TASK:-math}"
PRED_FILE="${PRED_FILE:-$OUT}"
GOLD_FILE="${GOLD_FILE:-cache/llm/${_DATA_TAG}/tgt.dev.txt}"
DETAIL_FILE="${DETAIL_FILE:-expm/llm/${_DATA_TAG}/std/base/eval_detail.jsonl}"
python tools/eval/run_eval.py --task "$TASK" --pred "$PRED_FILE" --gold "$GOLD_FILE" --detail "$DETAIL_FILE"
```

## 5.2 推理层：`adv/predict/plm/qwen/predict.py`

主要逻辑：

1. 读取 `DATA_ID`（若设置）并拼出 `cache/<DATA_ID>/test.h5`；否则回退到配置 `cnfg.test_data`
2. 打开 HDF5，读取 `src/tgt` 组与样本数
3. 加载 tokenizer 与模型权重（`$MODEL`）
4. 循环 decode，写入 `pred_last.txt`

相关实现：

- 模型类：`transformer.PLM.QWen.v3.Decoder.Decoder`
- 解码调用：`mymodel.decode(...)`

## 5.3 解码层：`transformer/PLM/QWen/v3/Decoder.py`

关键函数：

- `greedy_decode`：`beam_size=1`
- `beam_decode`：`beam_size>1`

当前你已修复过一个边界问题：在按 `ilen` 回填输入 token 时，增加了索引有效性判断，避免 `narrow` 越界导致：

- `RuntimeError: start (...) + length (1) exceeds dimension size (...)`

## 5.4 评测入口：`tools/eval/run_eval.py`

职责：

- 读取 `pred` 与 `gold` 文件（按行对齐）
- 根据 `--task` 在 `TASK_REGISTRY` 里选择对应评测函数
- 聚合输出 `sum_score` / `mean_score` / `accuracy`
- 可写 `--detail` 为 JSONL

支持任务名包括：

- `squad` / `trivia`
- `nq_v1` / `nq_std` / `all_nq`
- `medmcqa` / `pubmedqa` / `mmlu`
- `gsm8k` / `math`
- `tool`

## 5.5 评分规则：`tools/eval/scorers.py`

部分任务规则简述：

- `squad`：忽略大小写子串命中（0/1）
- `nq_v1`：大小写敏感子串命中（0/1）
- `tool`：API 描述集合 Jaccard（连续分数 0~1）
- `gsm8k/math`：抽取数字列表并逐项比对（0/1）

---

## 5.6 四个数据集支持的评测任务与详细规则

下面按你的四个数据集给出“推荐任务 + 可选任务 + 具体判分规则”。

### 5.6.1 `pubdatasets_squad_closed`

- 推荐 `TASK=squad`
- 兼容可选：`TASK=trivia`（同一判分逻辑）
- gold 行格式：多答案列表（如 `["ans1", "ans2"]`，也兼容引号提取）

判分规则（`score_squad`）：

1. 将预测文本转小写；
2. 遍历 gold 候选答案（空串跳过）；
3. 只要任一候选答案的小写字符串是预测文本的子串，就记 1 分，否则 0 分。

特点：

- 是**忽略大小写**的子串命中，不要求严格全等；
- 最终 `accuracy=mean_score`，范围 `[0,1]`。

### 5.6.2 `pubdatasets_toolbench`

- 推荐 `TASK=tool`
- gold 行格式：每行一个 JSON 数组（元素是 API 描述字符串）

判分规则（`score_tool`）：

1. 对 gold 和 pred 的每条 API 描述都做规范化：
   - 按逗号切分只保留前 3 段；
   - 全部转小写；
2. 预测侧先去掉前缀 `API MAIN INFO: `，再按空行分段为多条 API；
3. 计算集合 Jaccard：`|交集| / |并集|`；
4. 若并集为空，分数为 `0.0`。

特点：

- 这是**连续分数**（非 0/1），`mean_score` 即平均 Jaccard；
- `run_eval.py` 会额外提示 `tool uses Jaccard mean`。

### 5.6.3 `pubdatasets_nq_closed`

- 推荐 `TASK=nq_v1`（与现有 `task-p3.sh` 默认一致）
- 可选：`TASK=nq_std` 或 `TASK=all_nq`（两者在当前代码里同一逻辑）
- gold 行格式：答案列表（多候选）

判分规则（`score_nq_v1` / `score_nq_std`）：

1. 遍历 gold 候选答案；
2. 只要任一候选答案是预测文本的子串即判 1，否则 0。

特点（与 squad 不同）：

- 这里是**大小写敏感**子串匹配；
- 若你的预测做了大小写归一化，可能影响分数。

### 5.6.4 `pubdatasets_metamathqa`

- 可用任务：`TASK=math`（在 `run_eval.py` 中映射到 `score_gsm8k`）
- 等价可选：`TASK=gsm8k`
- gold 行格式：数字列表（JSON 数组 / literal 列表 / 逗号分隔数字）

判分规则（`score_gsm8k`，`math` 复用）：

1. 从预测文本中用正则提取：
   - 形如 `...数字...\nThe answer is: 1,2,3` 的答案段；
2. 将答案段解析成浮点列表；
3. 与 gold 浮点列表逐项比较，绝对误差 `<=1e-6` 才算该项相等；
4. 列表长度必须一致且全部相等才记 1，否则 0。

特点：

- 格式要求严格，尤其依赖 `The answer is:` 这类模式；
- 这也是 `task-p4.sh` 默认 `SKIP_EVAL=1` 的核心原因：MetaMath 常见 gold 文本与此规则不对齐。

### 5.6.5 四个数据集的任务映射建议（实操）

- SQuAD：`TASK=squad`
- ToolBench：`TASK=tool`
- NQ：`TASK=nq_v1`（需要时可试 `nq_std`）
- MetaMath：默认先只预测；有对齐数字 gold 再用 `TASK=math`

### 5.6.6 结果字段怎么看

`run_eval.py` 统一输出：

- `total`：参与评测样本数（pred/gold 取较短长度）
- `sum_score`：总分
- `mean_score`：平均分
- `accuracy`：除 `tool` 外通常等于 `mean_score`

若传 `--detail`，每条样本会记录到 JSONL：

- 基础字段：`idx`、`task`、`pred`、`gold_line`、`score`
- 任务附加字段（例如 `hit`、`std_ans`、`model_ans` 或错误信息）

---

## 6. 与训练脚本的关系

四个训练脚本为：

- `task-1.sh`（squad）
- `task-2.sh`（toolbench）
- `task-3.sh`（nq）
- `task-4.sh`（metamath）

它们都调用 `adv/train/plm/train_lora_qwen.py`，并通过环境变量覆盖：

- `DATA_ID`
- `PRE_TRAINED_M`
- `LORA_RANK`
- `LORA_ALPHA`
- `USE_AMP`（主要在 `task-4.sh` 默认提供）

训练输出目录按：

- `expm/<DATA_ID>/std/base/`

预测脚本默认走：

- `expm/llm/<_DATA_TAG>/std/base/merge_last.h5`

所以请确保你用于推理的权重文件路径与实际训练产物一致（必要时用 `MODEL=...` 显式覆盖）。

---

## 7. 常用覆盖参数（不改代码）

### 只跑预测不评测

```bash
SKIP_EVAL=1 bash task-p1.sh
```

### 切换数据集

```bash
DATA_ID=llm/pubdatasets_nq_closed bash task-p3.sh
```

### 指定模型文件

```bash
MODEL=expm/llm/pubdatasets_nq_closed/std/base/last.h5 bash task-p3.sh
```

### 指定 tokenizer

```bash
TOKENIZER=/home/common/plm/Qwen/Qwen3-8B bash task-p2.sh
```

### 切换评测任务

```bash
TASK=nq_std bash task-p3.sh
```

---

## 8. 常见问题与排查建议

### 8.1 `FileNotFoundError: cache/.../test.h5`

- 检查 `DATA_ID` 是否正确
- 检查 `cache/<DATA_ID>/test.h5` 是否存在
- 现在 `predict.py` 已支持优先使用 `DATA_ID` 环境变量

### 8.2 `permission denied` / `stale file handle` / `Errno 512`

- 多为文件系统/挂载层瞬时异常（非模型代码逻辑）
- 先重开终端 + 重新激活环境 + 重新运行
- 必要时重装相关包（如 `numpy/jinja2/transformers`）

### 8.3 OOM

- 减小解码开销（例如改 `beam_size` 或先用 `SKIP_EVAL=1` 做冒烟）
- 检查是否有残留进程占 GPU 显存

### 8.4 MetaMath 评测结果异常

- 优先确认 `GOLD_FILE` 是否满足 `math/gsm8k` 期望格式（数字列表）
- 不对齐时建议仅做预测输出，评测用单独对齐脚本

---

## 9. ToolBench 与 MetaMathQA：基于已生成文本构造评测 gold

结合 `LORA_DATA_FORMAT_AND_STEPS.md` 的数据流，你现在已经有这些文本：

- `cache/llm/pubdatasets_toolbench/tgt.dev.txt`
- `cache/llm/pubdatasets_metamathqa/tgt.dev.txt`

这两个 `tgt.dev.txt` 都是由 `pubdatasets_to_srctgt.py` 生成的“标签侧文本”，但它们**不能直接同方式评测**：

- `metamathqa`：可以从 `tgt.dev.txt` 提取数字答案后评测 `TASK=math`
- `toolbench`：`TASK=tool` 要求每行是 API 列表（JSON 数组），通常需从原始结构化字段（如 `relevant_apis`）构造

统一原则：

- `run_eval.py` 按行对齐，`pred` 第 `i` 行必须对应 `gold` 第 `i` 行同一样本

### 9.1 MetaMathQA（`TASK=math`）：由已生成 `tgt.dev.txt` 生成 gold（建议先筛纯数值样本）

`math/gsm8k` 评测侧 (`tools/eval/scorers.py` 的 `score_gsm8k`) 最稳妥的 gold 是“每行一个纯数字列表”。  
而 `MetaMathQA` 的 `tgt.dev.txt` 中会混有分数、区间、表达式、有序对等格式（例如 `\frac{1}{64}`、`[-9,9]`、`x^3-...`）。

因此建议：

1. **优先抽取 `The answer is:` 后的内容**
2. **仅保留可安全解析为纯数字列表的样本**
3. 生成两份文件：
   - `gold.math.strict.txt`：仅纯数字样本（推荐用于稳定对比）
   - `gold.math.loose.txt`：宽松回退（工程排查用，噪声更大）

示例脚本如下：

```bash
cd /home/gpchen/lora/transformer-edge
python - <<'PY'
import json, os, re

inp = "cache/llm/pubdatasets_metamathqa/tgt.dev.txt"
out_strict = "cache/llm/pubdatasets_metamathqa/gold.math.strict.txt"
out_loose = "cache/llm/pubdatasets_metamathqa/gold.math.loose.txt"
os.makedirs(os.path.dirname(out_strict), exist_ok=True)

pat_answer = re.compile(r"The answer is:\s*(.+)$")
pat_num = re.compile(r"-?\d+(?:\.\d+)?")
pat_pure_nums = re.compile(r"^\s*-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?)*\s*$")

def parse_strict(ans_text):
    # 仅接受纯数字列表，如 "12" / "-3.5" / "1,2,3"
    if pat_pure_nums.match(ans_text):
        return [float(x) for x in pat_num.findall(ans_text)]
    return None

def parse_loose(full_line, ans_text):
    # 宽松回退：先从 answer 子串抓数字；若没有再抓整行最后一个数字
    nums = pat_num.findall(ans_text)
    if nums:
        return [float(x) for x in nums]
    nums_all = pat_num.findall(full_line)
    if nums_all:
        return [float(nums_all[-1])]
    return []

total = 0
strict_kept = 0
fallback_used = 0

with open(inp, "r", encoding="utf-8") as fi, \
     open(out_strict, "w", encoding="utf-8") as fs, \
     open(out_loose, "w", encoding="utf-8") as fl:
    for line in fi:
        s = line.strip()
        total += 1
        m = pat_answer.search(s)
        ans = m.group(1).strip() if m else ""

        strict_vals = parse_strict(ans)
        if strict_vals is not None:
            strict_kept += 1
            fs.write(json.dumps(strict_vals, ensure_ascii=False) + "\n")
        else:
            fallback_used += 1

        loose_vals = parse_loose(s, ans)
        fl.write(json.dumps(loose_vals, ensure_ascii=False) + "\n")

print("total:", total)
print("strict_kept:", strict_kept)
print("strict_drop:", total - strict_kept)
print("loose_fallback_used:", fallback_used)
print("wrote:", out_strict)
print("wrote:", out_loose)
PY
```

评测：

```bash
SKIP_EVAL=0 TASK=math GOLD_FILE=cache/llm/pubdatasets_metamathqa/gold.math.strict.txt bash task-p4.sh
```

说明：

- `strict` 更适合模型版本间稳定对比（噪声更低）
- `loose` 适合快速跑通或排查，但会把分数/区间等样本近似成数字，可能引入误差
- 若你坚持“全量样本”评测，建议改造 `score_gsm8k` 的解析逻辑，使其支持分数和区间文本

### 9.2 ToolBench（`TASK=tool`）：优先由结构化字段生成 gold

`tool` 任务在 `run_eval.py` 中要求 gold 每行是 JSON 数组（API 描述列表）。  
仅用 `pubdatasets_toolbench/tgt.dev.txt`（assistant 长文本）通常不够稳定，推荐从 `benchmark/*.parquet` 的 `relevant_apis` 生成 gold。

```bash
cd /home/gpchen/lora/transformer-edge
python - <<'PY'
import glob, json, os
import pandas as pd

src_root = "/home/gpchen/pubdatasets/toolbench-v1/benchmark"
out_file = "cache/llm/pubdatasets_toolbench/gold.toolbench.benchmark.txt"
files = sorted(glob.glob(os.path.join(src_root, "*.parquet")))
os.makedirs(os.path.dirname(out_file), exist_ok=True)

def to_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v) for v in x]
    if isinstance(x, dict):
        return [str(v) for v in x.values()]
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return [str(v) for v in obj]
            if isinstance(obj, dict):
                return [str(v) for v in obj.values()]
        except Exception:
            pass
        return [s]
    return [str(x)]

with open(out_file, "w", encoding="utf-8") as f:
    for fp in files:
        df = pd.read_parquet(fp)
        if "relevant_apis" not in df.columns:
            continue
        for v in df["relevant_apis"]:
            f.write(json.dumps(to_list(v), ensure_ascii=False) + "\n")

print("wrote:", out_file)
PY
```

评测：

```bash
TASK=tool GOLD_FILE=cache/llm/pubdatasets_toolbench/gold.toolbench.benchmark.txt bash task-p2.sh
```

说明：

- 该 gold 对齐的是 `benchmark` 样本顺序
- 如果你的预测输入不是同一批 benchmark 样本（例如来自 `data/validation`），不能直接混用

### 9.3 快速自检（两类都建议做）

在跑评测前，先看行数是否一致：

```bash
wc -l expm/llm/pubdatasets_metamathqa/std/base/pred_last.txt cache/llm/pubdatasets_metamathqa/gold.math.txt
wc -l expm/llm/pubdatasets_toolbench/std/base/pred_last.txt cache/llm/pubdatasets_toolbench/gold.toolbench.benchmark.txt
```

若行数不一致，`run_eval.py` 会截断到较短长度，分数可参考但不建议作为最终结果。

---

## 10. 最小可复现命令（建议保存）

```bash
cd /home/gpchen/lora/transformer-edge
source /home/gpchen/lora/transformer-edge/init_conda.sh
conda activate lora-edge

# squad
bash task-p1.sh

# toolbench
bash task-p2.sh

# nq
bash task-p3.sh

# metamath（默认仅预测）
bash task-p4.sh
```

