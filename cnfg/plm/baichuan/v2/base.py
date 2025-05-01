#encoding: utf-8

from cnfg.base import *

model_name = "model"
pre_trained_m = None

bindDecoderEmb = False

isize = 4096# 4096 for 7B, 5120 for 14B
ff_hsize = 2 * 11008# 2 * (11008 for 7B, 13696 for 14B)
nhead = max(1, isize // 128)# 32 for 7B, 40 for 14B
attn_hsize = isize

nlayer = 32# 32 for 7B, 40 for 14B

drop = 0.0
attn_drop = drop
act_drop = drop

norm_output = True

greedy_sample_temperature = 0.3
greedy_sample_top_p = 0.85
greedy_sample_top_k = 5
repetition_penalty = 1.05# 1.05 for 7B, 1.1 for 14B
