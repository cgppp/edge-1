source init_conda.sh && conda activate lora-edge
cd /home/gpchen/lora/transformer-edge
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

python tools/check/debug/plm/qwen/v3_8b_align.py --staged-gpu --max-new-tokens 32