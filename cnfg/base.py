#encoding: utf-8

group_id = "std"

run_id = "base"

data_id = "w14ed32"

exp_dir = "expm/"
cache_dir = "cache/"

train_data = cache_dir + data_id + "/train.h5"
dev_data = cache_dir + data_id + "/dev.h5"
test_data = cache_dir + data_id + "/test.h5"

fine_tune_m = None

# non-exist indexes in the classifier.
# "<pad>":0, "<sos>":1, "<eos>":2, "<unk>":3
# add 3 to forbidden_indexes if there are <unk> tokens in data
# must be None if use_fast_loss is set in cnfg/hyp.py

#from fbind import fbl

forbidden_indexes = None#[0, 1] + fbl

save_auto_clean = True
overwrite_eva = False
save_every = 1500
num_checkpoint = 4
epoch_start_checkpoint_save = 3

# 单步优化前累积 token 上限；略降可减轻 optimizer step 附近显存峰值（仍 OOM 时再试 12000/8000）
tokens_optm = 16000

earlystop = 8
maxrun = 128
training_steps = 100000

batch_report = 5000
report_eva = False

use_cuda = True
# Data Parallel multi-GPU support can be enabled with values like: "cuda:0, 1, 3". Set to None to use all GPUs.
gpuid = "cuda:0"
# 混合精度，显著降低激活与部分计算显存；若需全精度请将此处改为 False（环境变量仅支持 USE_AMP=1 打开，见 train_lora_qwen.py）
use_amp = True
multi_gpu_optimizer = True
use_cuda_bfmp = True

bindDecoderEmb = True
share_emb = True

isize = 512
ff_hsize = isize * 4
nhead = max(1, isize // 64)
attn_hsize = isize

nlayer = 6

drop = 0.1
attn_drop = drop
act_drop = drop

# False for Hier/Incept Models
norm_output = True

warm_step = 8000
lr_scale = 1.0

label_smoothing = 0.1

weight_decay = 0

beam_size = 1
length_penalty = 0.0
# use multi-gpu for translating or not. `predict.py` will take the last gpu rather than the first in case multi_gpu_decoding is set to False to avoid potential break due to out of memory, since the first gpu is the main device by default which takes more jobs.
multi_gpu_decoding = False

seed = 666666

epoch_save = False

# to accelerate training through sampling, 0.8 and 0.1 in: Dynamic Sentence Sampling for Efficient Training of Neural Machine Translation
dss_ws = None
dss_rm = None

use_ams = False

src_emb = None
freeze_srcemb = False
tgt_emb = None
freeze_tgtemb = False
scale_down_emb = True

train_statesf = None
save_train_state = False
