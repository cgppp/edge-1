#encoding: utf-8

from cnfg.base import *

model_name = "model"
pre_trained_m = None

bindDecoderEmb = False# False for 9B

isize = 4096# 4096 for 9B
ff_hsize = 2 * 13696# 2 * (13696 for 9B)
nhead = 32# 32 for 9B
attn_hsize = isize
kv_nhead = 2# 2 for 9B

nlayer = 40# 40 for 9B

drop = 0.0
attn_drop = drop
act_drop = drop

norm_output = True
