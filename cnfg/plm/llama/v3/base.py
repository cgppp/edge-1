#encoding: utf-8

from cnfg.base import *

model_name = "model"
pre_trained_m = None

bindDecoderEmb = True
share_emb = True

isize = 2048# 2048 for 1B
ff_hsize = isize * 8
nhead = max(1, isize // 64)
attn_hsize = isize
kv_nhead = 8

nlayer = 16# 16 for 1B

drop = 0.0
attn_drop = drop
act_drop = drop

norm_output = True
