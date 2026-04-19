source init_conda.sh 
conda activate lora-edge
cd /home/gpchen/lora/transformer-edge

python tools/check/debug/plm/qwen/h5_hf_keydiff.py --max-print 50