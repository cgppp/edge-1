# LoRA 复现：项目结构与数据集格式说明

本文档说明如何在 **Qwen3 基座 + LoRA 微调** 下，用项目内脚本完成数据处理与训练，并统一使用 **`task.sh`** 与 **环境变量** 控制配置，无需手改 cnfg 文件。

---

## 零、从头学习本项目的建议路径（想看懂代码时怎么学）

若你希望**从零培养“能看懂这个项目”的能力**，可以按下面顺序来：先补一点前置概念，再按“执行流”读代码，最后跟一次训练加深印象。

### 0.1 前置知识（不必全会，够用即可）

| 主题 | 需要到什么程度 | 建议 |
|------|----------------|------|
| **Python** | 能读类、函数、`import`、字典/列表 | 随便一本入门书或 [Python 官方教程](https://docs.python.org/3/tutorial/) 前几章即可。 |
| **PyTorch** | `nn.Module`、`forward`、`state_dict`、`Parameter`、`tensor` 形状 | 官方 [60 分钟入门](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)，重点“神经网络”和“自动求导”两节。 |
| **Transformer** | 知道 Decoder-only：embedding → 多层（self-attn + FFN）→ norm → 输出头 | 不必手写，能看懂“输入 token id → 每层做什么 → 输出 logits”即可。可看 [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) 或《Attention Is All You Need》Decoder 部分。 |
| **LoRA** | 公式：在 Linear 旁加低秩矩阵 A、B，只训练 A、B | 读 [LoRA 论文](https://arxiv.org/abs/2106.09685) 前两页或任意一篇中文 LoRA 科普，理解“冻结原权重 + 只训小矩阵”即可。 |

不必等“全部学完”再碰项目，**边看代码边查**上述概念效果更好。

---

### 0.2 本项目的推荐阅读顺序（按“执行流”走一遍）

目标：**跟着一次训练的启动过程**，从脚本入口一直跟到“模型前向 + 算 loss”。建议顺序如下。

1. **入口与配置**
   - 打开 **`task.sh`**：看它设置了哪些环境变量、最后执行哪条命令。
   - 打开 **`adv/train/plm/train_lora_qwen.py`** 的**最前面约 60 行**：看它如何用 `os.environ.get("DATA_ID", ...)` 等覆盖 `cnfg`，以及 `import cnfg.plm.qwen.v3.base as cnfg` 等从哪读配置。

2. **训练主流程（不抠细节，先看骨架）**
   - 在 **`train_lora_qwen.py`** 里用编辑器的“跳转到定义”或搜索，找到 **`def train(...)`** 和 **`if __name__ == "__main__":`** 下面的调用关系。
   - 看：**主程序**先做什么（建模型、load_plm、std2lora、优化器、数据加载），再进入 **`train(...)`** 循环；循环里如何取一个 batch、调用 `model(...)`、算 loss、反向传播。不必先弄懂每一行，先留下“数据从哪来、loss 从哪来”的印象。

3. **模型是怎么来的**
   - 搜索 **`NMT(`** 或 **`mymodel =`**，找到**模型构建**的那一行；看它用的是哪个类（来自 `transformer.PLM.QWen.v3.Decoder` 的 `Decoder`）。
   - 打开 **`transformer/PLM/QWen/v3/Decoder.py`**：看 **`class Decoder`** 的 **`__init__`**（有哪些子模块：`wemb`、`nets`、`out_normer`、`classifier`），再看 **`forward`** 或 **`build_states`** 的大致数据流（输入 id → embed → 多层 layer → norm → logits）。
   - 看 **`load_plm`**：理解“从字典（.h5/.bin 读出来的 key-value）里按名字把 tensor 拷贝进当前模块”。

4. **LoRA 是怎么挂上去的**
   - 在 **`train_lora_qwen.py`** 里搜索 **`std2lora`**，看调用处传入的 `lora_features`、`lora_alpha` 从哪来（`cnfg.lora`）。
   - 打开 **`utils/lora/base.py`**：看 **`std2lora`** 在做什么（把 `nn.Linear` / `nn.Embedding` 替换成带 lora_wa、lora_wb 的版本）。
   - 打开 **`modules/lora/base.py`**：看 **`Linear`** 的 **`forward`**（`out = 原线性层(x) + scaling * (x @ lora_wa @ lora_wb)`），建立“只多了一小坨可训练参数”的直觉。

5. **数据与 loss**
   - 在 **`train_lora_qwen.py`** 的 **`train()`** 里看：一个 batch 的 `seq_batch`、`seq_o` 从哪来（HDF5 的 `src`、`tgt` 组）。
   - 打开 **`utils/train/llm.py`**：看 **`PMaskDataConverter`** 的 **`forward`**：输入整段 token 和 tgt 的 `[start, end]`，输出只在“回答区间”上算 loss 的 mask。这样就知道“为什么 HDF5 里 tgt 是两列”。

6. **数据是怎么生成出来的（可选）**
   - 按文档第三节的流程，从 **`scripts/plm/llm/pubdatasets_to_srctgt.py`**（原始数据 → src/tgt 文本）→ **`tools/plm/map/qwen/v3.py`**（文本 → .ids）→ **`tools/plm/mkiodata_llmdec.py`**（.ids → train.h5）走一遍，不必每行都懂，知道“每一步产出什么文件、格式是什么”即可。

按 1→2→3→4→5 走完，你就能在脑子里串起一条线：**任务脚本 → 配置 → 模型构建 → 加载基座 → 挂 LoRA → 读 HDF5 → 前向 → 只对回答区间算 loss → 反向**。

---

### 0.3 跟一次“真实训练”来巩固

- 用**小数据**（例如只保留 GSM8K 的一小部分）跑 1～2 个 epoch，观察日志里的 loss、`Load pre-trained model from: ...`。
- 在 **`train_lora_qwen.py`** 里对 `loss` 或 `seq_batch.shape` 加一两行 **`print`**，再跑，确认和你预期一致（例如 batch 形状、loss 是否在算）。
- 改 **`cnfg/lora.py`** 里的 **`lora_features`**（如 8→4），再跑，看参数量或日志是否有变化，加深“LoRA 只改这一处就生效”的印象。

---

### 0.4 遇到看不懂的代码时怎么查

- **某个变量/类从哪来**：在 IDE 里“跳转到定义”（F12 或 Ctrl+Click），或在整个项目里搜索该符号。
- **某个 cnfg 项在哪用**：在项目里搜索该变量名（如 `pre_trained_m`、`lora_features`），看谁在读它。
- **数据形状/格式**：看 **`utils/train/llm.py`** 和 **`utils/fmt/plm/llmdec/dual.py`**、**`tools/plm/mkiodata_llmdec.py`** 的注释或 docstring；本文档第二节也写了 HDF5 的 src/tgt 约定。
- **PyTorch / Transformer / LoRA 概念**：用上面 0.1 的链接或任意你习惯的教程查漏补缺。

把 **0.1 当作字典**，**0.2 当作主线**，**0.3 当作实验**，多走两遍就会越来越熟。

---

## 一、项目结构与关键文件

### 1.1 训练入口与配置

| 路径 | 说明 |
|------|------|
| `adv/train/plm/train_lora_qwen.py` | **LoRA + Qwen 训练入口**。读 HDF5，用 `PMaskDataConverter` 只对回答部分算 loss；**脚本开头支持环境变量覆盖**：`DATA_ID`、`PRE_TRAINED_M`、`LORA_RANK`、`LORA_ALPHA`。 |
| `cnfg/base.py` | 全局：`data_id`、`train_data`、`dev_data`、`cache_dir`、`exp_dir`、`gpuid` 等。 |
| `cnfg/plm/qwen/v3/base.py` | Qwen 结构及 `pre_trained_m`（基座 model.h5 路径）。 |
| `cnfg/lora.py` | LoRA：`lora_features`（默认 8）、`lora_alpha`（默认 16）、`prefix_ids` 等。 |
| `cnfg/hyp.py` | 超参：`max_tokens_gpu`、`max_sentences_gpu` 等（5090 可适当调大）。 |

### 1.2 LoRA 与数据格式

| 路径 | 说明 |
|------|------|
| `modules/lora/base.py` | LoRA 的 `Linear` / `Embedding`（lora_wa、lora_wb）。 |
| `utils/lora/base.py` | `std2lora`、`lora2std`。 |
| `utils/train/llm.py` | **`PMaskDataConverter`**：约定 HDF5 的 `src`/`tgt` 语义与形状。 |
| `utils/fmt/plm/llmdec/dual.py` | 双文件（指令.ids + 回答.ids）→ 拼接序列 + 每条 `[lid, lgth]`。 |
| `tools/plm/mkiodata_llmdec.py` | 生成 LoRA 用 HDF5（src=拼接 token 矩阵，tgt=`(batch,2)` 回答区间 [start+1,end+1]）。 |
| `tools/plm/map/qwen/v3.py` | 文本 → 空格分隔的 token ID 行（Qwen2TokenizerFast，与 Qwen3 兼容）。 |
| `cnfg/vocab/plm/qwen/v3.py` | 模板与特殊 token（`instruct_auto`、`assistant` 等）。 |

### 1.3 脚本与任务

| 路径 | 说明 |
|------|------|
| `task.sh` | 激活 conda、设置 `PRE_TRAINED_M` / `DATA_ID` / `LORA_RANK` / `LORA_ALPHA`，执行 `python adv/train/plm/train_lora_qwen.py`。 |
| `scripts/plm/llm/pubdatasets_to_srctgt.py` | 从 pubdatasets（NQ/SQuAD/GSM8K/ToolBench）生成 src/tgt 文本；NQ 支持 `--workers` 并行。 |

---

### 1.4 Qwen3 + LoRA 复现相关文件树与详细说明（只关心“以 Qwen3 为基座的 LoRA”时看这里）

下面**只列出**与「Qwen3 基座 + LoRA 训练」直接相关的文件，按目录树组织，并逐项说明作用。仓库里其他文件（如 Qwen2.5、LLaMA、T5、RoBERTa 等）与本次复现无关，可忽略。

```
lora/transformer-edge/
├── task.sh                                    # 见下
├── LORA_DATA_FORMAT_AND_STEPS.md               # 本说明文档
│
├── cnfg/                                      # 配置（训练时被读取）
│   ├── base.py                                # 全局：data_id、train_data、dev_data、cache_dir、exp_dir、gpuid 等
│   ├── lora.py                                # LoRA：lora_features、lora_alpha、scaling、name_cfunc 等
│   ├── hyp.py                                 # 训练超参：max_tokens_gpu、batch 相关
│   └── plm/qwen/v3/
│       ├── base.py                            # Qwen3 结构：pre_trained_m、isize、nlayer、ff_hsize、nhead、bindDecoderEmb 等（0.6B/8B 等规模在此区分）
│       ├── hyp.py                             # Qwen3 结构选项：use_glu_ffn、use_rope、cache_len_default 等
│       ├── ihyp.py                            # 从 hyp 派生的中间量（norm、relpos 等）
│       └── __init__.py
├── cnfg/vocab/plm/qwen/v3.py                  # Qwen3 词表：vocab_size、pad_id、sos_id、eos_id、instruct 模板等
│
├── adv/train/plm/
│   └── train_lora_qwen.py                     # 【训练入口】Qwen3 Decoder + LoRA；读 DATA_ID 对应 train/dev.h5，PRE_TRAINED_M 基座，LORA_RANK/LORA_ALPHA；内部用 std2lora、PMaskDataConverter
│
├── transformer/PLM/
│   ├── NMT.py                                 # 通用 PLM 的 load_plm 包装（本项目用 Decoder 单边，不直接跑 NMT）
│   ├── LLMDecoder.py                          # 基类：embed_tokens、layers、norm、classifier、load_plm 调度各层
│   └── QWen/v3/
│       ├── Decoder.py                         # 【Qwen3 解码器】结构 + load_plm（从 .h5/.bin 按 key 灌权重的实现）
│       └── __init__.py
│
├── modules/
│   ├── lora/base.py                           # 【LoRA 核心】LoRA 版 Linear / Embedding（lora_wa、lora_wb），forward 里 base + scaling * (x @ lora_wa @ lora_wb)
│   └── plm/qwen/v3.py                         # Qwen3 的 SelfAttn、PositionwiseFF 等子模块（被 Decoder 引用）
│
├── utils/
│   ├── lora/base.py                           # std2lora：把模型里的 nn.Linear / nn.Embedding 替换成 modules.lora 的 LoRA 版；lora2std 逆操作
│   ├── train/llm.py                           # PMaskDataConverter：根据 HDF5 的 src/tgt 形状，只对“回答区间”算 loss 的 mask
│   ├── fmt/plm/llmdec/dual.py                 # 双文件 batch_loader：指令.ids + 回答.ids → 拼接序列 + [lid, lgth]，与 mkiodata_llmdec 约定一致
│   ├── plm/base.py                            # load_plm_wrapper、copy_plm_parameter、fix_parameter_name（.h5/.bin 读取与 key 映射）
│   ├── io.py                                  # save_model / h5save：保存 checkpoint 时用 ps_func 可只存 LoRA 参数（rgrad_filter）
│   └── h5serial.py                            # HDF5 读写（h5File、h5save、h5load）
│
├── tools/plm/
│   ├── convert_qwen3_hf_to_bin.py             # [新增] HuggingFace Qwen3 目录 → 单文件 .bin 或 .h5，供 PRE_TRAINED_M 使用（见第六节）
│   ├── mkiodata_llmdec.py                     # 双文件 .ids → 写入 train.h5 / dev.h5（src 组 + tgt 组，格式见第二节）
│   ├── map/qwen/v3.py                         # 文本 → 空格分隔的 token ID 行（Qwen2TokenizerFast，与 Qwen3 兼容）
│   └── token/qwen/v3.py                       # 调用 tokenizer 做 tokenize（若脚本需要）
│
└── scripts/plm/llm/
    └── pubdatasets_to_srctgt.py               # 从 pubdatasets 目录生成各数据集的 src/tgt 文本对（NQ/SQuAD/GSM8K/ToolBench）
```

**各层作用简述**

| 层级 | 作用 |
|------|------|
| **入口与数据** | `task.sh` 设好环境变量后执行 `train_lora_qwen.py`；数据来自 `cache/llm/pubdatasets_*/` 下的 train.h5、dev.h5，由 `pubdatasets_to_srctgt` → map → sort → `mkiodata_llmdec` 生成。 |
| **配置** | `cnfg/base.py` 决定读哪个 data_id（进而 train_data/dev_data 路径）；`cnfg/plm/qwen/v3/base.py` 决定 Qwen3 规模与基座路径 `pre_trained_m`；`cnfg/lora.py` 决定 LoRA 的 rank/alpha。 |
| **模型** | `transformer/PLM/QWen/v3/Decoder.py` 是 Qwen3 的解码器实现；`modules/plm/qwen/v3.py` 是其中的 SelfAttn、FFN 等；`modules/lora/base.py` 是 LoRA 的 Linear/Embedding。 |
| **训练逻辑** | `train_lora_qwen.py` 里：构建 Decoder → 若 `pre_trained_m` 非空则 `load_plm` 加载基座 → `std2lora` 把部分 Linear/Embedding 换成 LoRA → 用 `PMaskDataConverter` 按 tgt 区间算 loss，只更新 LoRA 参数。 |
| **权重与格式** | 基座须为单文件 .h5 或 .bin（key 与 load_plm 约定一致）；若只有 HuggingFace 目录，用 `tools/plm/convert_qwen3_hf_to_bin.py` 转换。HDF5 的 src/tgt 格式由 `utils/train/llm.py` 与 `utils/fmt/plm/llmdec/dual.py`、`mkiodata_llmdec.py` 共同约定。 |

**和“复现”无关、可先忽略的目录/文件**

- `cnfg/plm/qwen/v2d5/`、`transformer/PLM/QWen/v2d5/`、`tools/plm/map/qwen/v2d5.py` 等：Qwen2.5，非 Qwen3。
- `transformer/PLM/` 下 T5、BART、LLaMa、Gemma、RoBERTa 等：其他 PLM，本复现不用。
- `adv/train/` 下非 `plm/train_lora_qwen.py` 的脚本：其他任务（GEC、prompt 等）。
- `tools/cnfg/`：若存在，多为工具用配置副本，与主流程 `cnfg/` 二选一即可。

---

## 二、LoRA 所需的数据集格式（定义在哪些文件）

- **训练脚本** 读取的 HDF5 由 **`utils/train/llm.py`** 中的 **`PMaskDataConverter`** 消费。
- **格式约定**：
  - **`src` 组**：key 为 `"0"`,`"1"`,…；value 为 `(batch_size, seq_len)` 的 token 矩阵（指令+回答拼接）。
  - **`tgt` 组**：同 key；value 为 `(batch_size, 2)`，每行 `[start+1, end+1]`，表示只在该区间内算 loss。
- **格式的“使用方”**：`utils/train/llm.py`（PMaskDataConverter）。
- **格式的“生成方”**：`utils/fmt/plm/llmdec/dual.py`；**写入 HDF5 的脚本**：`tools/plm/mkiodata_llmdec.py`。

### 2.1 四个数据集做这套数据处理的原因

NQ、SQuAD、GSM8K、ToolBench 形态不一（问答、推理、多轮对话等），但训练目标统一为：**在「指令 + 回答」的序列上，只对「回答」这一段算 loss**，这样模型学会的是“见指令后生成答案”，而不是去拟合指令里的字。因此需要：

1. **统一成「指令 → 回答」两条文本**  
   四个数据集的原始字段不同（GSM8K 的 question/answer、SQuAD 的 context+question/answers、NQ 的 question/short_answers、ToolBench 的 conversations），第一步要把它们都变成「一条指令（src）、一条回答（tgt）」的文本对，便于后面用同一套 tokenizer 和 HDF5 格式。

2. **只对回答区间算 loss**  
   训练时整段序列是「指令 token + 回答 token」拼在一起；若整段都算 loss，模型会连指令里的字也去拟合。通过 **tgt 组存 [start+1, end+1]**，训练脚本里的 **PMaskDataConverter** 只在回答区间上算 loss，这样梯度只驱动“生成答案”的能力。

3. **从原始表到 HDF5 的流水线**  
   原始数据（parquet 等）→ 抽成 src/tgt 文本 → 用 Qwen tokenizer 打成 token ID → 按长度排序（可选，利于 batching）→ 写成项目约定的 HDF5（src 组 + tgt 组）。这样 **无论哪个数据集**，训练脚本都读同一格式的 train.h5/dev.h5，只需换 **DATA_ID** 指向不同 `pubdatasets_*` 目录即可。

**小结**：数据处理的目的 = **把四类数据统一成「指令+回答」序列，并标明回答区间，使 LoRA 训练时只对回答算 loss**；流程 = 抽文本对 → tokenize → 排序 → 写 HDF5。

### 2.2 数据处理用到的项目文件与建议路径总结

下面按 **“做一次完整数据处理”** 时你会用到的文件，按使用顺序列出，并给出**建议路径**（在项目根下执行命令时，这些路径即为你实际用到的文件）。

| 阶段 | 目的 | 用到的项目文件 | 建议用法 / 路径 |
|------|------|----------------|-----------------|
| **1. 原始数据 → src/tgt 文本** | 四个数据集统一成「指令一行、回答一行」 | **`scripts/plm/llm/pubdatasets_to_srctgt.py`** | `python scripts/plm/llm/pubdatasets_to_srctgt.py --pubdatasets <pubdatasets根目录> --out cache/llm`；单数据集加 `--dataset gsm8k` / `nq` / `squad` / `toolbench`。输出在 **`cache/llm/pubdatasets_<名>/`**，得到 `src.train.txt`、`tgt.train.txt`、`src.dev.txt`、`tgt.dev.txt`。 |
| **2. 文本 → token ID** | 用 Qwen tokenizer 打成 ID，与 Qwen3 基座一致 | **`tools/plm/map/qwen/v3.py`**、**`cnfg/vocab/plm/qwen/v3.py`**（模板名） | `export TOKENIZER=/path/to/Qwen3-*`；`python tools/plm/map/qwen/v3.py $WKD/src.train.txt $TOKENIZER $WKD/src.train.txt.ids instruct_auto`（src 用 `instruct_auto`，tgt 用 `assistant`）。输出 **`*.txt.ids`**。 |
| **3. 按长度排序（可选）** | 同 batch 内长度接近，减少 padding | **`tools/sort.py`** | `python tools/sort.py $WKD/src.train.txt.ids $WKD/tgt.train.txt.ids $WKD/src.train.srt.ids $WKD/tgt.train.srt.ids 2048`。输出 **`*.srt.ids`**。 |
| **4. .ids → HDF5** | 写成训练脚本能读的 src/tgt 组 | **`tools/plm/mkiodata_llmdec.py`**；格式约定见 **`utils/fmt/plm/llmdec/dual.py`** | `python tools/plm/mkiodata_llmdec.py $WKD/src.train.srt.ids $WKD/tgt.train.srt.ids $WKD/train.h5 1`（dev 同理）。输出 **`train.h5`、`dev.h5`**，位于 `cache/llm/pubdatasets_<名>/`。 |
| **5. 训练时读数据** | 按 data_id 找 train/dev.h5 | **`cnfg/base.py`**（data_id、cache_dir）、**`adv/train/plm/train_lora_qwen.py`**（环境变量 DATA_ID 覆盖）、**`utils/train/llm.py`**（PMaskDataConverter） | 在 **`task.sh`** 中设 `DATA_ID=llm/pubdatasets_gsm8k`（或 nq/squad/toolbench）；训练脚本会读 **`cache_dir + DATA_ID + "/train.h5"`** 与 **`dev.h5`**。 |

**建议路径小结**（以 GSM8K 为例，其他数据集只改 `pubdatasets_gsm8k` 为对应名）：

- **工作目录（每个数据集一个）**：`cache/llm/pubdatasets_gsm8k/`（或 `pubdatasets_nq`、`pubdatasets_squad`、`pubdatasets_toolbench`）。
- **你主要要改/要设的**：环境变量 **`TOKENIZER`**（Qwen3 权重目录）、**`WKD`**（当前数据集目录）、**`DATA_ID`**（训练时用，如 `llm/pubdatasets_gsm8k`）；**`task.sh`** 里的 **`PRE_TRAINED_M`**、**`DATA_ID`**（或运行时用环境变量覆盖）。
- **不必改的文件**：`utils/fmt/plm/llmdec/dual.py`、`utils/train/llm.py`、`tools/plm/mkiodata_llmdec.py` 内部逻辑，只要按第三节命令调用即可。

---

## 三、完整流程：从 pubdatasets 到训练（统一步骤）

路径与目录以 **`/home/gpchen/pubdatasets`**、**`/home/gpchen/lora/transformer-edge`** 为准。

### 3.0 环境（建议最先做）

详见 **`docs/CONDA_SETUP.md`**。简要：

```bash
conda create -n lora-edge python=3.12 -y
conda activate lora-edge
cd /home/gpchen/lora/transformer-edge
pip install -r requirements.txt
pip install -r requirements.lora.txt
```

之后所有命令均在 **`conda activate lora-edge`** 下执行。

### 3.1 数据集在 pubdatasets 中的布局

| 数据集 | 训练 | 验证/测试 | 字段说明 |
|--------|------|-----------|----------|
| GSM8K | `gsm8k/main/train-*.parquet` | `gsm8k/main/test-*.parquet` | `question` → 指令，`answer` → 回答 |
| SQuAD | `squad/plain_text/train-*.parquet` | `squad/plain_text/validation-*.parquet` | `context`+`question` → 指令，`answers` → 回答 |
| NQ | `natural_questions/default/train-*-of-00287.parquet` | `natural_questions/dev/validation-*.parquet` | `question.text` → 指令，`annotations[0].short_answers[0].text` → 回答 |
| ToolBench | `toolbench-v1/data/train-*.parquet` | `toolbench-v1/data/validation-*.parquet` | `conversations` 为 dict：`from`/`value` 数组，取最后 user/assistant |

### 3.2 步骤 1：生成 src/tgt 文本

依赖：`pandas`、`pyarrow`（见 `requirements.lora.txt`）。

```bash
cd /home/gpchen/lora/transformer-edge
python scripts/plm/llm/pubdatasets_to_srctgt.py \
  --pubdatasets /home/gpchen/pubdatasets \
  --out cache/llm
```

- 单数据集：`--dataset gsm8k|nq|squad|toolbench`
- 限制样本：`--max-train 5000 --max-dev 500`
- NQ 加速：`--workers 8`

输出：`cache/llm/pubdatasets_gsm8k/` 等，每目录含 `src.train.txt`、`tgt.train.txt`、`src.dev.txt`、`tgt.dev.txt`。

### 3.3 步骤 2：文本 → token ID（map）

`TOKENIZER` 指向 **Qwen3 权重目录**（与 Qwen2 同 tokenizer）。

```bash
export TOKENIZER=/path/to/Qwen3-0.6B
export WKD=cache/llm/pubdatasets_gsm8k

python tools/plm/map/qwen/v3.py $WKD/src.train.txt $TOKENIZER $WKD/src.train.txt.ids instruct_auto
python tools/plm/map/qwen/v3.py $WKD/tgt.train.txt $TOKENIZER $WKD/tgt.train.txt.ids assistant
python tools/plm/map/qwen/v3.py $WKD/src.dev.txt   $TOKENIZER $WKD/src.dev.txt.ids   instruct_auto
python tools/plm/map/qwen/v3.py $WKD/tgt.dev.txt   $TOKENIZER $WKD/tgt.dev.txt.ids   assistant
```

得到 `*.txt.ids`（每行空格分隔的 token ID）。对 NQ/SQuAD/ToolBench 将 `WKD` 换为对应 `pubdatasets_*` 即可。

### 3.4 步骤 3：按长度排序（可选）

```bash
export WKD=cache/llm/pubdatasets_gsm8k
python tools/sort.py $WKD/src.train.txt.ids $WKD/tgt.train.txt.ids $WKD/src.train.srt.ids $WKD/tgt.train.srt.ids 2048
python tools/sort.py $WKD/src.dev.txt.ids $WKD/tgt.dev.txt.ids $WKD/src.dev.srt.ids $WKD/tgt.dev.srt.ids 1048576
```

未排序时，下面步骤用 `.txt.ids` 替代 `.srt.ids`。

### 3.5 步骤 4：生成 LoRA 用 HDF5

```bash
export WKD=cache/llm/pubdatasets_gsm8k
python tools/plm/mkiodata_llmdec.py $WKD/src.train.srt.ids $WKD/tgt.train.srt.ids $WKD/train.h5 1
python tools/plm/mkiodata_llmdec.py $WKD/src.dev.srt.ids   $WKD/tgt.dev.srt.ids   $WKD/dev.h5   1
```

若出现 “Number of batches: 0” 或报错，脚本会提示检查：.ids 文件是否存在、是否为空、每行是否为空格分隔的整数。

### 3.6 步骤 5：用 task.sh 启动训练（推荐）

四个数据集都做完 3.2～3.5 后，**统一用 `task.sh` 启动训练**，无需改 cnfg 文件。训练脚本 **`adv/train/plm/train_lora_qwen.py`** 会在开头根据环境变量覆盖：`DATA_ID`、`PRE_TRAINED_M`、`LORA_RANK`、`LORA_ALPHA`。

**前置**：基座须为项目可读的 **model.h5**（或 .bin），路径设为 `PRE_TRAINED_M`。

1. **修改 `task.sh` 中的默认值（或运行时用环境变量覆盖）**
   - `PRE_TRAINED_M`：Qwen3 的 model.h5 路径。
   - `DATA_ID`：本次使用的数据目录，与 3.2～3.5 生成的一致：
     - `llm/pubdatasets_gsm8k`
     - `llm/pubdatasets_squad`
     - `llm/pubdatasets_nq`
     - `llm/pubdatasets_toolbench`

2. **（可选）5090 显存**：在 `cnfg/hyp.py` 中适当调大 `max_tokens_gpu`（如 12288、16384）。

3. **执行**
   ```bash
   cd /home/gpchen/lora/transformer-edge
   bash task.sh
   ```
   或显式传参：
   ```bash
   DATA_ID=llm/pubdatasets_gsm8k PRE_TRAINED_M=/path/to/Qwen3-0.6B/model.h5 bash task.sh
   ```

4. **换数据集**：改 `DATA_ID` 再执行一次即可。

**环境变量小结**：`DATA_ID`（数据目录）、`PRE_TRAINED_M`（基座路径）、`LORA_RANK`（默认 8）、`LORA_ALPHA`（默认 16）、`CUDA_VISIBLE_DEVICES`（如 0）。未设置时使用 cnfg 中的默认值。

### 3.7 单条龙示例（以 GSM8K 为例）

```bash
cd /home/gpchen/lora/transformer-edge
conda activate lora-edge
export TOKENIZER=/path/to/Qwen3-0.6B
export WKD=cache/llm/pubdatasets_gsm8k

python scripts/plm/llm/pubdatasets_to_srctgt.py --pubdatasets /home/gpchen/pubdatasets --out cache/llm --dataset gsm8k
python tools/plm/map/qwen/v3.py $WKD/src.train.txt $TOKENIZER $WKD/src.train.txt.ids instruct_auto
python tools/plm/map/qwen/v3.py $WKD/tgt.train.txt $TOKENIZER $WKD/tgt.train.txt.ids assistant
python tools/plm/map/qwen/v3.py $WKD/src.dev.txt   $TOKENIZER $WKD/src.dev.txt.ids   instruct_auto
python tools/plm/map/qwen/v3.py $WKD/tgt.dev.txt   $TOKENIZER $WKD/tgt.dev.txt.ids   assistant
python tools/sort.py $WKD/src.train.txt.ids $WKD/tgt.train.txt.ids $WKD/src.train.srt.ids $WKD/tgt.train.srt.ids 2048
python tools/sort.py $WKD/src.dev.txt.ids $WKD/tgt.dev.txt.ids $WKD/src.dev.srt.ids $WKD/tgt.dev.srt.ids 1048576
python tools/plm/mkiodata_llmdec.py $WKD/src.train.srt.ids $WKD/tgt.train.srt.ids $WKD/train.h5 1
python tools/plm/mkiodata_llmdec.py $WKD/src.dev.srt.ids   $WKD/tgt.dev.srt.ids   $WKD/dev.h5   1

# 启动训练（任选其一）
bash task.sh
# 或：DATA_ID=llm/pubdatasets_gsm8k PRE_TRAINED_M=/path/to/Qwen3-0.6B/model.h5 python adv/train/plm/train_lora_qwen.py
```

---

## 四、以 Qwen3 为基座复现 LoRA

**第四部分的作用**：说明如何把 **Qwen3 当作基座模型（预训练权重）**，在本项目里做 LoRA 微调。包括：数据/tokenizer 是否兼容、基座权重要什么格式、**`pre_trained_m` 是什么、如何设置预训练权重**。

---

### 4.1 `pre_trained_m` 是什么？

**`pre_trained_m` 就是预训练基座权重的文件路径**（本项目里也叫「基座模型」）。

- 训练时，脚本会先按 Qwen 结构实例化一个 Decoder；若配置里 **`pre_trained_m` 不为空**，就调用 **`mymodel.load_plm(cnfg.pre_trained_m)`**，从该路径**加载已有参数**到模型里，再在此基础上加 LoRA 并只训练 LoRA 参数。
- 若 **`pre_trained_m` 为 None**，则不加载任何预训练权重，模型会**随机初始化**（一般不用于 LoRA 复现）。

因此：要做「Qwen3 + LoRA」复现，**必须**把 `pre_trained_m` 设成你的 **Qwen3 预训练权重文件** 的路径。

---

### 4.2 基座权重文件格式（.h5 或 .bin）

本项目 **只支持从单个文件加载基座**，且该文件必须是以下两种之一：

| 格式 | 说明 |
|------|------|
| **.h5** | 项目使用的 HDF5 格式，内部是若干 key（参数名）对应 tensor 数据。 |
| **.bin** | PyTorch 的 `torch.save(state_dict)` 保存的权重字典。 |

训练脚本 **不支持** 直接读 HuggingFace 的 **safetensors** 或 **模型目录**（如 `Qwen/Qwen3-0.6B`）。若你只有 HuggingFace 下载的 Qwen3（safetensors），需要**先**用自写或第三方脚本，把权重转成上述 **.h5 或 .bin** 再使用。

---

### 4.2.1 预训练权重从哪里获取（项目内没有）

**本仓库中不包含任何预训练基座权重**（没有 .h5 / .bin 文件）。`cnfg/plm/qwen/v3/base.py` 里 `pre_trained_m = None`，`task.sh` 中的 `/path/to/Qwen3-0.6B/model.h5` 仅为占位，需自行准备并替换为实际路径。

可从以下渠道获取 Qwen3 权重，再按 4.2 的格式要求使用或转换：

| 来源 | 说明与链接 |
|------|------------|
| **Hugging Face** | 官方基座：**[Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base)**（safetensors）。下载后需转成 .h5 或 .bin 才能被本项目加载。命令行示例：`huggingface-cli download Qwen/Qwen3-0.6B-Base --local-dir /你的目录/Qwen3-0.6B`。 |
| **ModelScope** | 国内可用。在 [ModelScope 模型库](https://modelscope.cn/models?name=Qwen3) 搜索 Qwen3，选择对应规模（如 0.6B）下载到本地目录。同样为 safetensors，需再转换为本项目 .h5/.bin。 |
| **实验室/组内** | 脚本中曾出现路径 `/home/common/plm/Qwen/Qwen3-0.6B`，若你所在环境有共享的 Qwen 目录或**已转好的 model.h5**，可直接向导师/同门索取路径，并在 `task.sh` 或 cnfg 中设为 `PRE_TRAINED_M` / `pre_trained_m`。 |

拿到的是 **safetensors 或 HF 目录** 时，必须先用自写脚本或组内工具转成 **.h5 或 .bin**，再按 4.3 设置路径；若拿到的是**已转好的 model.h5**，可直接进行 4.3 步骤 2。

---

### 4.2.2 当你有 /home/common/plm/Qwen/ 下多个 HF 模型时（选哪个、目录里有什么、如何转成本项目权重）

若你所在环境已有 **`/home/common/plm/Qwen/`**，下面通常会有多个 Qwen3 基座目录，例如：

| 目录名 | 说明 |
|--------|------|
| Qwen3-0.6B-Base | 0.6B 预训练基座 |
| Qwen3-1.7B-Base | 1.7B 预训练基座 |
| Qwen3-4B-Base | 4B 预训练基座 |
| Qwen3-8B-Base | 8B 预训练基座 |
| Qwen3-14B-Base | 14B 预训练基座 |
| Qwen3-30B-A3B-Base | 30B-A3B MoE 预训练基座 |
| Qwen3-32B | 32B 基座（无 -Base 后缀） |

**单个模型目录里有什么（以 Qwen3-4B-Base 为例）**  
每个这样的目录都是**标准 HuggingFace 格式**，一般包含：

- **config.json**：模型结构（层数、hidden_size、attention 等）
- **model\*.safetensors**：权重（单文件或分片，如 model-00001-of-00003.safetensors）
- **model.safetensors.index.json**：多分片时的权重索引
- **generation_config.json**：生成配置
- **tokenizer**：tokenizer.json、tokenizer_config.json、vocab.json、merges.txt 等

本项目**不能直接读该目录**，需要把该目录转成**单个 .bin 或 .h5 文件**后再用（见下文「如何转换」）。

**应该使用哪个模型、为什么**

- **推荐先用 Qwen3-0.6B-Base**：当前 **`cnfg/plm/qwen/v3/base.py`** 的默认超参（isize=1024、nlayer=28、ff_hsize 等）是按 **0.6B** 写的，不改配置即可训练，显存占用小、迭代快，适合先跑通 LoRA。
- 若要用 **1.7B / 4B / 8B / 14B**：必须**同时**修改 **`cnfg/plm/qwen/v3/base.py`** 中与规模对应的项（如 isize、ff_hsize、nlayer、bindDecoderEmb 等，文件内注释有各规模的取值），并保证与所选 HF 模型的 config.json 一致；否则结构对不上，加载会报错或精度异常。
- **30B-A3B、32B**：为 MoE 或更大规模，本项目默认 Decoder 为稠密结构，若要用需确认是否有对应 cnfg/实现，一般建议先用 0.6B/1.7B/4B。

**如何把 HF 目录转成本项目可用的 .bin（或 .h5）**

项目内已提供转换脚本 **`tools/plm/convert_qwen3_hf_to_bin.py`**，可将任意 HF 格式的 Qwen3 目录（如 `/home/common/plm/Qwen/Qwen3-0.6B-Base`）转为单个 **.bin** 或 **.h5** 文件，供 `PRE_TRAINED_M` 使用。

1. 在**项目根目录**下执行（示例：0.6B 转成 .bin）：
   ```bash
   python tools/plm/convert_qwen3_hf_to_bin.py /home/common/plm/Qwen/Qwen3-0.6B-Base --output /home/common/plm/Qwen/Qwen3-0.6B-Base/model.bin
   ```
2. 若想输出 **.h5**：
   ```bash
   python tools/plm/convert_qwen3_hf_to_bin.py /home/common/plm/Qwen/Qwen3-0.6B-Base --output /path/to/model.h5 --format h5
   ```
3. 转换完成后，将 **`PRE_TRAINED_M`**（或 cnfg 中的 **`pre_trained_m`**）设为上述 **model.bin** 或 **model.h5** 的路径，按 4.3 步骤 2、3 启动训练并确认日志中出现 `Load pre-trained model from: ...`。

依赖：已安装 **torch**、**transformers**；输出 .h5 时需能 import 本项目 **cnfg**（在项目根下执行即可）。

---

### 4.3 设置预训练权重的步骤（重点）

#### 步骤 1：准备好基座权重文件

- 若你**已有**别人或脚本导出的 **Qwen3 的 model.h5**（或 model.bin），记下该文件的**绝对路径**，例如：  
  `/home/gpchen/plm/Qwen3-0.6B/model.h5`
- 若你**只有** HuggingFace 格式的 Qwen3 目录（如 `/home/common/plm/Qwen/Qwen3-0.6B-Base`，内含 config.json、model\*.safetensors、tokenizer 等）：
  1. 使用项目自带脚本转成 **.bin**（推荐）或 **.h5**：
     ```bash
     python tools/plm/convert_qwen3_hf_to_bin.py /home/common/plm/Qwen/Qwen3-0.6B-Base --output /path/to/model.bin
     ```
  2. 将 **PRE_TRAINED_M**（或 **pre_trained_m**）设为得到的 **model.bin**（或 model.h5）的路径。  
  （若选用的不是 0.6B，需同时把 **`cnfg/plm/qwen/v3/base.py`** 改为对应规模的 isize、nlayer、ff_hsize 等，见 4.2.2。）

#### 步骤 2：在训练时把路径传给程序

任选一种方式即可。

**方式 A（推荐）：环境变量，配合 task.sh**

- 在 **`task.sh`** 里设置：
  ```bash
  export PRE_TRAINED_M="/home/gpchen/plm/Qwen3-0.6B/model.h5"
  ```
  或运行前在终端：
  ```bash
  export PRE_TRAINED_M=/你的路径/model.h5
  bash task.sh
  ```
- **`adv/train/plm/train_lora_qwen.py`** 开头会读 **`PRE_TRAINED_M`**，并把它赋给 **`cnfg.plm.qwen.v3.base.pre_trained_m`**，后续 `load_plm` 就会用这个路径加载预训练权重。

**方式 B：直接改配置文件**

- 打开 **`cnfg/plm/qwen/v3/base.py`**，找到：
  ```python
  pre_trained_m = None
  ```
  改为你的基座文件路径，例如：
  ```python
  pre_trained_m = "/home/gpchen/plm/Qwen3-0.6B/model.h5"
  ```
- 保存后运行 `python adv/train/plm/train_lora_qwen.py` 或 `bash task.sh`（此时可不再设 `PRE_TRAINED_M`）。

#### 步骤 3：确认训练时确实加载了基座

- 训练日志里应出现类似：**`Load pre-trained model from: /你的路径/model.h5`**。  
  若没有这行且也没有报错，多半是 **`pre_trained_m` 仍为 None**（环境变量未生效或 cnfg 未改对），需要回到步骤 2 检查路径是否传入。

**小结**：预训练权重 = 基座模型文件；**设置预训练权重** = 把该文件的路径赋给 **`pre_trained_m`**（通过环境变量 `PRE_TRAINED_M` 或 cnfg 中的 `pre_trained_m`），并保证文件格式为 .h5 或 .bin。

---

### 4.4 数据与 tokenizer（与 Qwen3 的兼容性）

当前流程生成的数据（src/tgt 文本 → Qwen2TokenizerFast 打 id → train/dev.h5）**可直接用于 Qwen3 基座**。Qwen3 与 Qwen2 使用同一套 tokenizer，做 map 时把 **`TOKENIZER` 指到 Qwen3 权重目录**（与 4.3 中基座同版本即可）即可。

---

### 4.5 配置方式小结（二选一）

- **推荐**：不改 cnfg，用 **`task.sh`** 设置 **`PRE_TRAINED_M`**（及 `DATA_ID`、`LORA_RANK`、`LORA_ALPHA` 等），运行 `adv/train/plm/train_lora_qwen.py`。
- **可选**：在 `cnfg/base.py` 中设 `data_id`，在 **`cnfg/plm/qwen/v3/base.py`** 中设 **`pre_trained_m`**，在 `cnfg/lora.py` 中设 `lora_features`/`lora_alpha`，然后直接执行 `python adv/train/plm/train_lora_qwen.py`。

---

## 五、小结

- **LoRA 数据格式** 由 **`utils/train/llm.py`**（PMaskDataConverter）与 **`utils/fmt/plm/llmdec/dual.py`** 约定；**写入 HDF5** 由 **`tools/plm/mkiodata_llmdec.py`** 完成。
- **训练入口**：**`adv/train/plm/train_lora_qwen.py`**，支持环境变量 `DATA_ID`、`PRE_TRAINED_M`、`LORA_RANK`、`LORA_ALPHA`。
- **推荐使用方式**：完成 3.2～3.5 后，在 **`task.sh`** 中设好 `PRE_TRAINED_M` 与 `DATA_ID`，执行 **`bash task.sh`** 即可复现 Qwen3 + LoRA 训练。

---

## 六、本说明文档相关的新增脚本与补充注释

在 **`/home/gpchen/lora/transformer-edge`** 目录下，为配合本复现流程**新生成的**脚本与**在原有文件上补充的注释**如下。

- **新生成的 .py 脚本**（本项目内新建，仅 1 个）：**`tools/plm/convert_qwen3_hf_to_bin.py`**，添加原因见下表。

### 6.1 新生成的脚本

| 文件 | 添加原因 |
|------|----------|
| **`tools/plm/convert_qwen3_hf_to_bin.py`** | 项目训练脚本只支持从**单个 .h5 或 .bin** 加载基座，不支持直接读 HuggingFace 的 safetensors 或模型目录。本脚本将 HuggingFace 格式的 Qwen3 目录（如 `/home/common/plm/Qwen/Qwen3-0.6B-Base`）转为项目可用的 .bin 或 .h5，便于设置 `PRE_TRAINED_M`，无需手写转换逻辑。 |

### 6.2 补充了详细注释的现有文件

| 文件 | 添加原因 |
|------|----------|
| **`utils/train/llm.py`** | 定义 **PMaskDataConverter**（只对回答区间算 loss 的核心逻辑）。补充了模块说明、类/方法 docstring、参数与返回值含义，以及 forward 内各变量与分支的注释，便于理解 HDF5 tgt 组与 loss mask 的对应关系。 |
| **`utils/fmt/plm/llmdec/dual.py`** | 定义双文件（指令 .ids + 回答 .ids）的 batch_loader / batch_padder，与 **mkiodata_llmdec.py** 写入的 HDF5 格式一致。补充了模块说明、两函数的参数/返回值/变量含义及各代码块功能注释，便于理解「指令+回答」拼接与 `[lid, lgth]` 的由来。 |

以上脚本与注释均围绕本文档描述的复现流程与数据格式，若你修改了 HDF5 约定或基座加载方式，可相应调整或参考这些文件。
