#encoding: utf-8

import torch
from math import floor

from utils.torch.comp import torch_amin, torch_aminmax, torch_no_grad, torch_quant_dtype

# x = ((x + 1 - min(x)).log() + 1).log()
# bias = min(x) - 1
# x = ((x - bias).log() + 1).log()
def log_encode(x, dim=None, log_shift=1.0, log_bias=None):

	if log_shift >= 1.0:
		_log_bias = (torch_amin(x, dim=dim, keepdim=True).sub(log_shift)) if log_bias is None else log_bias
		rs = x.sub(_log_bias).log().add(log_shift).log()
	else:
		rs, _log_bias = x, log_bias

	return rs, _log_bias

def log_decode(x, log_bias=None, log_shift=1.0):

	rs = x.exp().add(-log_shift).exp().add(log_bias) if log_shift >= 1.0 else x

	return rs

# qrs = (x - lb) / (rb - lb) * (qmax - qmin) + qmin
# scale = (rb - lb) / (qmax - qmin)
# qrs = (x - lb) / scale + qmin
# qrs = (x - lb + scale * qmin) / scale
# bias = lb - scale * qmin
# qrs = (x - bias) / scale
# x = scale * qrs + bias
def estimate_quant_hyp_core(x, dim=None, qmin=None, qmax=None, log_shift=1.0, **kwargs):

	with torch_no_grad():
		_x, _dim = (x.view(-1), -1) if dim is None else (x, dim)
		_x, log_bias = log_encode(_x, dim=dim, log_shift=log_shift, log_bias=None)
		lb, rb = torch_aminmax(_x, dim=dim, keepdim=True)
		scale = (rb - lb) / float(qmax - qmin)
		bias = lb - scale * float(qmin)

	return torch.stack([scale, bias], dim=0) if log_bias is None else torch.stack([log_bias, scale, bias], dim=0)

def estimate_quant_hyp(x, dim=None, qmin=None, qmax=None, log_shift=1.0, estimate_quant_hyp_core=estimate_quant_hyp_core, **kwargs):

	_xd = x.dim()
	if (_xd > 2) and (dim is not None):
		_n = x.size(dim)
		if (dim == -1) or (dim == (_xd - 1)):
			_x, _edim = x.view(-1, _n), 0
		elif dim == 0:
			_x, _edim = x.view(_n, -1), -1
		else:
			_x, _edim = x.transpose(dim, 0).contiguous().view(_n, -1), -1
		qhyp = estimate_quant_hyp_core(_x, dim=_edim, qmin=qmin, qmax=qmax, log_shift=log_shift, **kwargs)

		return qhyp.view(rs.size(0), *(_n if _ == dim else 1 for _ in range(_xd)))
	else:

		return estimate_quant_hyp_core(x, dim=None if dim is None else (-1 if dim == 0 else 0), qmin=qmin, qmax=qmax, log_shift=log_shift, **kwargs)

def quant(x, qhyp, dtype=torch_quant_dtype, qmin=None, qmax=None, log_shift=1.0, **kwargs):

	with torch_no_grad():
		if qhyp.size(0) > 2:
			log_bias, scale, bias = qhyp.unbind(0)
		else:
			log_bias, (scale, bias) = None, qhyp.unbind(0)
		_x = x if log_bias is None else log_encode(x, dim=None, log_shift=log_shift, log_bias=log_bias)[0]

		return _x.sub(bias).div_(scale).clamp_(qmin, qmax).to(dtype, non_blocking=True)

def dequant(x, qhyp, log_shift=1.0, **kwargs):

	if qhyp.size(0) > 2:
		log_bias, scale, bias = qhyp.unbind(0)
	else:
		log_bias, (scale, bias) = None, qhyp.unbind(0)
	rs = bias.addcmul(x.to(scale.dtype, non_blocking=True), scale)
	if log_bias is not None:
		rs = log_decode(rs, log_bias=log_bias, log_shift=log_shift)

	return rs
