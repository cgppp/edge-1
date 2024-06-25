#encoding: utf-8

import torch

from modules.hplstm.MvAvg import MvAvgFunc

cuda_device = None#torch.device("cuda", 0)
if cuda_device is not None:
	torch.cuda.set_device(cuda_device.index)

beta = 0.9
x = torch.randn(2, 3, 2, 5, requires_grad=True, device=cuda_device)
_x = x.clone()
rs = MvAvgFunc(_x, beta, True)
rs.sum().backward()
xg = x.grad.clone()
x.grad = None
rsl = []
mbeta = 1.0 - beta
for i in range(x.size(1)):
	if i == 0:
		rsl.append(x * mbeta)
	else:
		rsl.append(rsl[-1] * beta + x.select(1, i) * mbeta)
rsl = torch.stack(rsl, 1)
print(rs)
print(rsl)
rsl.sum().backward()
print(xg)
print(x.grad)
