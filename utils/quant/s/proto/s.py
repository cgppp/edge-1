#encoding: utf-8

import torch
from math import floor

from utils.torch.comp import torch_amax, torch_amin, torch_no_grad, torch_quant_dtype

# qrs = x / max(x) * qmax
# scale = max(x) / qmax
# qrs = x / scale
# x = scale * qrs
def estimate_quant_hyp_core(x, dim=None, qmin=None, qmax=None, **kwargs):

	with torch_no_grad():
		_x, _dim = (x.view(-1), -1) if dim is None else (x, dim)

		return torch_amax(_x.abs(), dim=dim, keepdim=True).div_(qmax) if qmax == abs(qmin) else torch_amax(_x, dim=dim, keepdim=True).div(qmax).fmax(torch_amin(_x, dim=dim, keepdim=True).div(qmin))

def estimate_quant_hyp(x, dim=None, qmin=None, qmax=None, estimate_quant_hyp_core=estimate_quant_hyp_core, **kwargs):

	_xd = x.dim()
	if (_xd > 2) and (dim is not None):
		_n = x.size(dim)
		if (dim == -1) or (dim == (_xd - 1)):
			_x, _edim = x.view(-1, _n), 0
		elif dim == 0:
			_x, _edim = x.view(_n, -1), -1
		else:
			_x, _edim = x.transpose(dim, 0).contiguous().view(_n, -1), -1

		return estimate_quant_hyp_core(_x, dim=_edim, qmin=qmin, qmax=qmax, **kwargs).view(tuple(_n if _ == dim else 1 for _ in range(_xd)))
	else:

		return estimate_quant_hyp_core(x, dim=None if dim is None else (-1 if dim == 0 else 0), qmin=qmin, qmax=qmax, **kwargs)

def quant(x, qhyp, dtype=torch_quant_dtype, qmin=None, qmax=None, **kwargs):

	with torch_no_grad():
		return x.div(qhyp).clamp_(qmin, qmax).to(dtype, non_blocking=True)

def dequant(x, qhyp, **kwargs):

	return x.to(qhyp.dtype, non_blocking=True).mul(qhyp)
