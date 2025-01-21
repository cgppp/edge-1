#encoding: utf-8

from cnfg.hyp import *

ease_optimization = True
disable_ffn_bias = True
remove_classifier_bias = True

# choices: None, "GeLU", "Swish", "Sigmoid", "SReLU", "Mish", "NormSwish"
advance_activation_function = None
# using GLU activation function for FFN, choices: None, "GLU", "GEGLU", "SwiGLU".
use_glu_ffn = "SwiGLU"#"Swish" for 1B

# choices: "v1", "v2"
computation_order = "v2"

# default cached sequence length (for positional embedding, etc.)
cache_len_default = 4096# 131072 for 1B, reduced for subsequent mask cache

# window size (one side) of relative positional embeddings, 0 to disable. 8 and 16 are used in [Self-Attention with Relative Position Representations](https://aclanthology.org/N18-2074/) for Transformer Base and Big respectively. relative_position_max_bucket_distance for the bucket relative positional encoding used by T5, [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://www.jmlr.org/papers/v21/20-074.html), which slightly hampers the performance on WMT 14 En-De. disable_std_pemb to disable the standard positional embedding when use the relative position, or to disable only the decoder side with a tuple (False, True,), useful for AAN.
use_k_relative_position = 0
relative_position_max_bucket_distance = 0
# use rotary position embedding proposed in [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864).
disable_std_pemb = True
use_rope = True
rope_factor = 32.0
rope_high_freq_factor = 4.0
rope_low_freq_factor = 1.0
rope_original_max_position_embeddings = 8192
# use ALiBi position encoding proposed in [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://openreview.net/forum?id=R8sQPpGCv0).
use_alibi = False

# For BPE (using full vocabulary), the special <unk> token will never appear and thus can be removed from the vocabulary. Otherwise, it should be set to True.
use_unk = False
