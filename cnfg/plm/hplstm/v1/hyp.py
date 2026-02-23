#encoding: utf-8

from cnfg.hyp import *

ease_optimization = True
remove_classifier_bias = False
use_rmsnorm = True

# choices: None, "GeLU", "GeLUTanh", "Swish", "Sigmoid", "SReLU", "Mish", "NormSwish"
advance_activation_function = None
# using GLU activation function for FFN, choices: None, "GLU", "GEGLU", "SwiGLU", "GETanhGLU".
use_glu_ffn = None#"Swish" for 0.6/1.7/4/8/14B

# choices: "v1", "v2"
computation_order = "v2"

# default cached sequence length (for positional embedding, etc.)
cache_len_default = 16834# 40960 for 0.6B, reduced for subsequent mask cache
