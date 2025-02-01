#encoding: utf-8

from cnfg.base import *

model_name = "model"
pre_trained_m = None

bindDecoderEmb = True# True for 0.5/1.5/3B, False for 7B
share_emb = True

isize = 896# 896 for 0.5B, 1536 for 1.5B, 2048 for 3B, 3584 for 7B
ff_hsize = 2 * 4864# 2 * (4864 for 0.5B, 8960 for 1.5B, 11008 for 3B, 18944 for 7B)
nhead = max(1, isize // 64)# 14 for 0.5B, 12 for 1.5B, 16 for 3B, 28 for 7B
attn_hsize = isize
kv_nhead = 2# 2 for 0.5/1.5/3B, 4 for 7B

nlayer = 24# 24 for 0.5B, 28 for 1.5B, 36 for 3B

drop = 0.0
attn_drop = drop
act_drop = drop

norm_output = True

greedy_sample_temperature = 0.7
greedy_sample_top_p = 0.8
greedy_sample_top_k = 20
repetition_penalty = 1.1# not supported, 1.1 for 0.5/1.5B, 1.05 for 3B/7B
