#encoding: utf-8

import torch
from time import time

from utils.hplstm.cumsum import cumsumFunc

cuda_device = None#torch.device("cuda", 0)
if not torch.cuda.is_available():
	cuda_device = None
if cuda_device is not None:
	torch.cuda.set_device(cuda_device.index)

x = torch.randn(2, 3, 2, 5, requires_grad=True, device=cuda_device)
_x = x.clone()
rs = cumsumFunc(_x, True)
rs.sum().backward()
xg = x.grad.clone()
x.grad = None
rsl = x.cumsum(1)
print(rs)
print(rsl)
rsl.sum().backward()
print(xg)
print(x.grad)

nrun = 128
x = torch.randn(64, 128, 8, 64, requires_grad=True, device=cuda_device)
_st = time()
for _ in range(nrun):
	rs = x.cumsum(1)
	rs.sum().backward()
	x.grad = None
_et = time()
print(_et - _st)

_st = time()
for _ in range(nrun):
	rs = cumsumFunc(x, False)
	rs.sum().backward()
	x.grad = None
_et = time()
print(_et - _st)
