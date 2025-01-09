#encoding: utf-8

import sys
import torch
from math import floor
from time import time

from modules.base import ResSelfAttn
from modules.hplstm.snbase import ResHPLSTM
from transformer.Encoder import EncoderLayer
from utils.torch.comp import mask_tensor_type

cuda_device = torch.device("cuda", 0)
max_token = 4096
slen_iter = 64
niter = 1024
warm_iter = 8

isize = 512
fhsize = isize * 4
num_head = isize // 64
dropout = 0.1
attn_drop = dropout
act_drop = dropout

if cuda_device is not None:
	torch.cuda.set_device(cuda_device.index)

tdl = []
maskl = []
_ = slen_iter
_f_max_token = float(max_token)
while _ <= max_token:
	tdl.append(torch.randn(floor(_f_max_token / _), _, isize, requires_grad=True, device=cuda_device))
	maskl.append(torch.ones(_, _, requires_grad=False, dtype=mask_tensor_type, device=cuda_device).triu(1).unsqueeze(0))
	_ += slen_iter

rsam = ResSelfAttn(isize, hsize=None, num_head=num_head, dropout=attn_drop)
rslm = ResHPLSTM(isize, num_head=num_head, dropout=dropout, act_drop=act_drop)
rsem = EncoderLayer(isize, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head)
if cuda_device is not None:
	rsam.to(cuda_device, non_blocking=True)
	rslm.to(cuda_device, non_blocking=True)
	rsem.to(cuda_device, non_blocking=True)

t_af, t_lf, t_ef, t_ab, t_lb, t_eb = [], [], [], [], [], []
for _td, _m in zip(tdl, maskl):
	ntdata = float(niter * _td.size(0) * _td.size(1))
	tm, tl = rsam, t_af
	for _ in range(warm_iter):
		out = tm(_td, mask=_m)
		out = None
	_t = 0.0
	for _ in range(niter):
		_st = time()
		out = tm(_td, mask=_m)
		_et = time()
		out = None
		_t += (_et - _st)
	tl.append(ntdata / _t)
	tm, tl = rslm, t_lf
	for _ in range(warm_iter):
		out = tm(_td)
		out = None
	_t = 0.0
	for _ in range(niter):
		_st = time()
		out = tm(_td)
		_et = time()
		out = None
		_t += (_et - _st)
	tl.append(ntdata / _t)
	tm, tl = rsem, t_ef
	for _ in range(warm_iter):
		out = tm(_td, mask=_m)
		out = None
	_t = 0.0
	for _ in range(niter):
		_st = time()
		out = tm(_td, mask=_m)
		_et = time()
		out = None
		_t += (_et - _st)
	tl.append(ntdata / _t)
	tm, tl = rsam, t_ab
	for _ in range(warm_iter):
		tm(_td, mask=_m).sum().backward()
		tm.zero_grad(set_to_none=True)
		_td.grad = None
	_t = 0.0
	for _ in range(niter):
		_st = time()
		out = tm(_td, mask=_m).sum().backward()
		_et = time()
		tm.zero_grad(set_to_none=True)
		_td.grad = None
		_t += (_et - _st)
	tl.append(ntdata / _t)
	tm, tl = rslm, t_lb
	for _ in range(warm_iter):
		out = tm(_td).sum().backward()
		tm.zero_grad(set_to_none=True)
		_td.grad = None
	_t = 0.0
	for _ in range(niter):
		_st = time()
		out = tm(_td).sum().backward()
		_et = time()
		tm.zero_grad(set_to_none=True)
		_td.grad = None
		_t += (_et - _st)
	tl.append(ntdata / _t)
	tm, tl = rsem, t_eb
	for _ in range(warm_iter):
		out = tm(_td, mask=_m).sum().backward()
		tm.zero_grad(set_to_none=True)
		_td.grad = None
	_t = 0.0
	for _ in range(niter):
		_st = time()
		out = tm(_td, mask=_m).sum().backward()
		_et = time()
		tm.zero_grad(set_to_none=True)
		_td.grad = None
		_t += (_et - _st)
	tl.append(ntdata / _t)

ens = "\n".encode("utf-8")
with open(sys.argv[1], "wb") as f:
	for _l in zip(t_af, t_lf, t_ef, t_ab, t_lb, t_eb):
		f.write("\t".join(str(_) for _ in _l).encode("utf-8"))
		f.write(ens)
