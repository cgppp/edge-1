#encoding: utf-8

from cnfg.base import *

model_name = "model"
pre_trained_m = None

bindDecoderEmb = True# True for 1B

isize = 1152# 1152 for 1B
ff_hsize = 2 * 6912# 2 * (6912 for 1B)
nhead = 4# 4 for 1B
attn_hsize = isize
kv_nhead = 1# 1 for 1B

nlayer = 26# 26 for 1B

drop = 0.0
attn_drop = drop
act_drop = drop

norm_output = True
