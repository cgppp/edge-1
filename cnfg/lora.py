#encoding: utf-8

save_base = True
lora_fine_tune_m = None

lora_features = None
lora_alpha = None
scaling = 1.0
update_bias = False
name_cfunc = lambda mname: True
keep_lora_weight_tying = True

fine_tune_linear_bias = False
fine_tune_normer = False
name_cfunc_lb = lambda mname: True
name_cfunc_normer = lambda mname: True

prefix_ids = None
find_common_prefix = True
