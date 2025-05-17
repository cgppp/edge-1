#encoding: utf-8

use_quant = False

quant_linear = True
quant_embedding = True
quant_normer = False

# quantization dimension, None (estimate at all dimensions), an integer x (dimensions except for x), "min" (least dimension larger than 1), "max" (largest dimension)
quant_dim = None

# quantize the weight vectors of normers and the bias vectors, may not be necessary normally
quant_normer_weight=False
quant_bias=False

quant_io=False

name_cfunc=lambda mname: True
