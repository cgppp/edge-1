#encoding: utf-8

from math import inf

from utils.fmt.parser import parse_double_value_tuple, parse_none

from cnfg.ihyp import *
from cnfg.plm.hplstm.v1.hyp import *

# biases
enable_prev_ln_bias_default = enable_proj_bias_default = not ease_optimization

# computation order
norm_residual_default = not (computation_order.lower() == "v2")
