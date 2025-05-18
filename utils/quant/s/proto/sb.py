#encoding: utf-8

import torch
from math import floor

from utils.torch.comp import torch_aminmax, torch_no_grad, torch_quant_dtype

# qrs = (x - lb) / (rb - lb) * (qmax - qmin) + qmin
# scale = (rb - lb) / (qmax - qmin)
# qrs = (x - lb) / scale + qmin
# qrs = (x - lb + scale * qmin) / scale
# bias = lb - scale * qmin
# qrs = (x - bias) / scale
# x = scale * qrs + bias
def estimate_quant_hyp_core(x, dim=None, qmin=None, qmax=None, **kwargs):

	with torch_no_grad():
		_x, _dim = (x.view(-1), -1) if dim is None else (x, dim)
		#n = floor(float(_x.size(_dim)) / 2.0 / float(qmax - qmin))
		#lb, rb = _x.narrow(_dim, n, 1), _x.narrow(_dim, -(n + 1), 1)
		lb, rb = torch_aminmax(_x, dim=dim, keepdim=True)
		scale = (rb - lb) / float(qmax - qmin)
		bias = lb - scale * float(qmin)

	return torch.stack([scale, bias], dim=0)

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
		qhyp = estimate_quant_hyp_core(_x, dim=_edim, qmin=qmin, qmax=qmax, **kwargs)

		return qhyp.view(rs.size(0), *(_n if _ == dim else 1 for _ in range(_xd)))
	else:

		return estimate_quant_hyp_core(x, dim=None if dim is None else (-1 if dim == 0 else 0), qmin=qmin, qmax=qmax, **kwargs)

def quant(x, qhyp, dtype=torch_quant_dtype, qmin=None, qmax=None, **kwargs):

	with torch_no_grad():
		scale, bias = qhyp.unbind(0)

		return x.sub(bias).div_(scale).clamp_(qmin, qmax).to(dtype, non_blocking=True)

def dequant(x, qhyp, **kwargs):

	scale, bias = qhyp.unbind(0)

	return bias.addcmul(x.to(scale.dtype, non_blocking=True), scale)
