#encoding: utf-8

use_quant = False

quant_linear = True
quant_embedding = True
quant_normer = False

# enable if quant_log_shift >= 1.0, with additional computational costs. May disable when quantization with FP8 and enable with (u)int8 to take care of outlier values.
quant_log_shift = 0.0
# quantization dimension, None (estimate at all dimensions), an integer x (dimensions except for x), "min" (least dimension larger than 1), "max" (largest dimension)
quant_dim = None

# quantize the weight vectors of normers and the bias vectors, may not be necessary normally
quant_normer_weight=False
quant_bias=False

quant_io=False

# consider to disable large classifier quantization for efficiency
name_cfunc=lambda mname: True
