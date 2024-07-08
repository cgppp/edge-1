#encoding: utf-8

import torch

from utils.hplstm.RS1MvAvg import RS1MvAvgFunc

cuda_device = None#torch.device("cuda", 0)
if not torch.cuda.is_available():
	cuda_device = None
if cuda_device is not None:
	torch.cuda.set_device(cuda_device.index)

beta = 0.9
x = torch.randn(2, 3, 2, 5, requires_grad=True, device=cuda_device)
rs = RS1MvAvgFunc(x, beta)
rs.sum().backward()
xg = x.grad.clone()
x.grad = None
rsl = [x.new_zeros(x.select(1, 0).size())]
mbeta = 1.0 - beta
for i in range(x.size(1) - 1):
		rsl.append(rsl[-1] * beta + x.select(1, i) * mbeta)
rsl = torch.stack(rsl, 1)
print(rs)
print(rsl)
rsl.sum().backward()
print(xg)
print(x.grad)
