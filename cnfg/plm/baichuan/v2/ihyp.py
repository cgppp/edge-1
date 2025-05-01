#encoding: utf-8

from math import inf

from utils.fmt.parser import parse_double_value_tuple, parse_none

from cnfg.ihyp import *
from cnfg.plm.baichuan.v2.hyp import *

# biases
enable_prev_ln_bias_default = enable_proj_bias_default = not ease_optimization

# computation order
norm_residual_default = not (computation_order.lower() == "v2")

# Layer Norm
enable_ln_parameters = True

# activation fucntion
use_adv_act_default = advance_activation_function is not None
adv_act = advance_activation_function.lower() if use_adv_act_default else None
inplace_after_Custom_Act = use_adv_act_default and (adv_act not in set(["sigmoid"]))

# absolute position encoding
sinusoid_base_frequency = 1e4

# relative position encoding
use_k_relative_position_encoder, use_k_relative_position_decoder = parse_double_value_tuple(use_k_relative_position)
rel_pos_enabled = (max(use_k_relative_position_encoder, use_k_relative_position_decoder) > 0) or use_rope or use_alibi
relative_position_max_bucket_distance_encoder, relative_position_max_bucket_distance_decoder = parse_double_value_tuple(relative_position_max_bucket_distance)
disable_std_pemb_encoder, disable_std_pemb_decoder = parse_double_value_tuple(disable_std_pemb)
relpos_reduction_with_zeros = True

# hyper-parameters
inf_default = inf

ieps_default = 1e-9
ieps_ln_default = 1e-6
ieps_adam_default = 1e-9
ieps_ln_default = parse_none(ieps_ln_default, ieps_default)
ieps_adam_default = parse_none(ieps_adam_default, ieps_default)
ieps_noise_default = ieps_ln_default
ieps_upper_bound_default = ieps_default
ieps_dropout_multinomial_default = ieps_default

adam_betas_default = (0.9, 0.98,)
