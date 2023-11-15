#encoding: utf-8

from cnfg.base import *

kd_layers = tuple(tuple(range(1, _ + 1)) for _ in nlayer) if isinstance(nlayer, (list, tuple,)) else tuple(range(1, nlayer + 1))

gradapt_min_sim = 0.0
min_gold_p = 0.1
num_topk = 64
T = 1.0
min_T = T / 64.0
kd_weight = 0.7
kd_step = 0#warm_step + warm_step + 1

mix_kd = True
iter_kd = True
remove_gold = False
enable_proj = True
