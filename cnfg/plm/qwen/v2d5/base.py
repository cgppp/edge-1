#encoding: utf-8

from cnfg.base import *

model_name = "model"
pre_trained_m = None

bindDecoderEmb = True
share_emb = True

isize = 896# 896 for 0.5B
ff_hsize = 2 * 4864# 2 * (4864 for 0.5B)
nhead = max(1, isize // 64)
attn_hsize = isize
kv_nhead = 2

nlayer = 24# 24 for 0.5B

drop = 0.0
attn_drop = drop
act_drop = drop

norm_output = True
