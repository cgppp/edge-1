#encoding: utf-8

from cnfg.base import *

model_name = "model"
pre_trained_m = None

bindDecoderEmb = True# True for 0.6/1.7/4B, False for 8/14B

isize = 1024# 1024 for 0.6B, 2048 for 1.7B, 2560 for 4B, 4096 for 8B, 5120 for 14B
ff_hsize = 2 * 3072# 2 * (3072 for 0.6B, 6144 for 1.7B, 9728 for 4B, 12288 for 8B, 17408 for 14B)
nhead = max(1, isize // 64)# 16 for 0.6/1.7B, 32 for 4/8B, 40 for 14B
attn_hsize = nhead * 128
kv_nhead = 8# 8 for 0.6/1.7/4/8/14B

nlayer = 28# 28 for 0.6/1.7B, 36 for 4/8B, 40 for 14B

drop = 0.0
attn_drop = drop
act_drop = drop

norm_output = True

greedy_sample_temperature = 0.6
greedy_sample_top_p = 0.95
greedy_sample_top_k = 20
repetition_penalty = 1.0# 1.0 (disabled) for 0.6/1.7/4/8/14B
