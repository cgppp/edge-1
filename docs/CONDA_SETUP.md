# Conda 环境配置步骤（数据处理 + LoRA 训练）

在项目根目录 `/home/gpchen/lora/transformer-edge` 下操作。以下按顺序执行即可。

---

## 1. 创建并激活 conda 环境

项目使用 **Python 3.12**（见 README）。

```bash
# 创建环境，指定 Python 3.12
conda create -n lora-edge python=3.12 -y

# 激活环境（之后所有命令都在此环境下执行）
conda activate lora-edge
```

---

## 2. 安装核心依赖（训练 + 推理）

```bash
cd /home/gpchen/lora/transformer-edge
pip install -r requirements.txt
```

当前 `requirements.txt` 包含：
- `tqdm`
- `torch>=2.10.0`
- `h5py>=3.15.1`

若有 GPU，建议先装好 CUDA，再装 PyTorch（可到 [pytorch.org](https://pytorch.org) 按环境选命令），例如：

```bash
# 示例：CUDA 12.x
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install tqdm h5py
```

---

## 3. 安装数据处理与 Qwen 所需依赖

**数据处理**（pubdatasets → src/tgt 文本）：需要 `pandas`、`pyarrow`（读 parquet）。  
**Token 化与 LoRA 脚本**：需要 `transformers`（Qwen2 分词器）、`safetensors`。

**推荐**：用项目提供的 **`requirements.lora.txt`**，只包含上述依赖，避免与 `requirements.opt.txt` 的 `tokenizers`/`huggingface-hub` 版本冲突：

```bash
pip install -r requirements.lora.txt
```

若你坚持安装完整可选依赖（BPE、COMET、中文分词等），`requirements.opt.txt` 可能与 `transformers>=5.1.0` 冲突（如 tokenizers 与 huggingface-hub 不兼容）。可先装好 `requirements.lora.txt` 再按需单独安装其他包，或跳过 `requirements.opt.txt`。

---

## 4. 验证环境

在项目根下执行：

```bash
cd /home/gpchen/lora/transformer-edge
conda activate lora-edge

# 1）Python 版本
python -c "import sys; print(sys.version)"

# 2）核心包
python -c "import torch; import h5py; import tqdm; print('torch', torch.__version__)"

# 3）数据处理
python -c "import pandas; import pyarrow; print('pandas & pyarrow OK')"

# 4）Qwen 分词器（需能访问到 Qwen 权重目录时再测）
python -c "from transformers import Qwen2TokenizerFast; print('Qwen2TokenizerFast OK')"

# 5）项目内数据转换脚本（不读真实数据，仅测导入）
python -c "
from utils.fmt.plm.llmdec.dual import batch_padder
from cnfg.vocab.plm.qwen.v3 import pad_id
print('project imports OK')
"
```

无报错即表示环境可用。

---

## 5. 建议的完整安装顺序（复制执行）

```bash
conda create -n lora-edge python=3.12 -y
conda activate lora-edge
cd /home/gpchen/lora/transformer-edge

pip install -r requirements.txt
pip install -r requirements.lora.txt
# 不要装 requirements.opt.txt，会和 transformers 的 huggingface-hub 冲突
```

之后即可按 `LORA_DATA_FORMAT_AND_STEPS.md` 做数据处理和 LoRA 训练；所有 `python` 命令均在激活的 `lora-edge` 环境下运行即可。
