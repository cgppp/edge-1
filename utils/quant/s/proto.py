#encoding: utf-8

import torch
from math import floor

from utils.torch.comp import torch_no_grad

# qrs = (x - lb) / (rb - lb) * (qsize - 1) + qmin
# scale = (rb - lb) / (qsize - 1)
# qrs = (x - lb) / scale + qmin
# qrs = (x - lb + scale * qmin) / scale
# bias = lb - scale * qmin
# qrs = (x - bias) / scale
# x = scale * qrs + bias
def estimate_scale_bias_core_fast(x, dim=None, qsize=256, qmin=-128):

	with torch_no_grad():
		_x, _dim = (x.view(-1).msort(), -1) if dim is None else (x.sort(dim)[0], dim)
		n = floor(float(_x.size(_dim)) / 2.0 / float(qsize))
		lb, rb = _x.narrow(_dim, n, 1), _x.narrow(_dim, -(n + 1), 1)
		scale = (rb - lb) / float(qsize - 1)
		bias = lb - scale * float(qmin)

	return scale, bias

estimate_scale_bias_core = estimate_scale_bias_core_fast

def estimate_scale_bias(x, dim=None, qsize=256, qmin=-128, estimate_scale_bias_core=estimate_scale_bias_core):

	_xd = x.dim()
	if (_xd > 2) and (dim is not None):
		_n = x.size(dim)
		if (dim == -1) or (dim == (_xd - 1)):
			_x, _edim = x.view(-1, _n), 0
		elif dim == 0:
			_x, _edim = x.view(_n, -1), -1
		else:
			_x, _edim = x.transpose(dim, 0).contiguous().view(_n, -1), -1
		_s, _b = estimate_scale_bias_core(_x, dim=_edim, qsize=qsize, qmin=qmin)
		_vs = tuple(_n if _ == dim else 1 for _ in range(_xd))

		return _s.view(_vs), _b.view(_vs)
	else:

		return estimate_scale_bias_core(x, dim=None if dim is None else (-1 if dim == 0 else 0), qsize=qsize, qmin=qmin)

def quant(x, scale, bias, dtype=torch.int8, qmin=-128, qmax=127, **kwargs):

	with torch_no_grad():
		return x.sub(bias).div_(scale).clamp_(qmin, qmax).to(dtype, non_blocking=True)

def dequant(x, scale, bias, **kwargs):

	return bias.addcmul(x.to(scale.dtype, non_blocking=True), scale)
