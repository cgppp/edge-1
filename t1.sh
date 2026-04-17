cd /home/gpchen/lora/transformer-edge
source /home/gpchen/lora/transformer-edge/init_conda.sh
conda activate lora-edge

python - <<'PY'
import glob, json, os
import pandas as pd

bench_root = "/home/gpchen/pubdatasets/toolbench-v1/benchmark"
wkd = "cache/llm/pubdatasets_toolbench_benchmark"
os.makedirs(wkd, exist_ok=True)

files = sorted(glob.glob(os.path.join(bench_root, "*.parquet")))
if not files:
    raise FileNotFoundError(f"No parquet found under: {bench_root}")

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

queries, gold = [], []
for fp in files:
    df = pd.read_parquet(fp)
    if not {"query", "relevant_apis"}.issubset(df.columns):
        continue
    for _, row in df.iterrows():
        q = str(row["query"]).replace("\n", " ").strip()
        g = to_list(row["relevant_apis"])
        queries.append(q if q else " ")
        gold.append(g)

src_txt = f"{wkd}/src.dev.txt"
tgt_txt = f"{wkd}/tgt.dev.txt"
gold_txt = f"{wkd}/gold.toolbench.benchmark.txt"

with open(src_txt, "w", encoding="utf-8") as fs, \
     open(tgt_txt, "w", encoding="utf-8") as ft, \
     open(gold_txt, "w", encoding="utf-8") as fg:
    for q, g in zip(queries, gold):
        fs.write(q + "\n")
        line = json.dumps(g, ensure_ascii=False)
        ft.write(line + "\n")
        fg.write(line + "\n")

print("wrote:", src_txt)
print("wrote:", gold_txt)
print("samples:", len(queries))
PY

export WKD=cache/llm/pubdatasets_toolbench_benchmark
export TOKENIZER=/home/common/plm/Qwen/Qwen3-8B

# 1) 文本 -> ids（仅 src）
python tools/plm/map/qwen/v3.py "$WKD/src.dev.txt" "$TOKENIZER" "$WKD/src.dev.txt.ids" instruct_auto

# 2) ids -> test.h5（未排序，保持与 gold 行顺序一致）
python tools/plm/llmdec/mktest.py "$WKD/src.dev.txt.ids" "$WKD/test.h5" 1

# 3) 快速检查
wc -l "$WKD/src.dev.txt" "$WKD/gold.toolbench.benchmark.txt"

