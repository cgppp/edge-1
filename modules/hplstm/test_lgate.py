#encoding: utf-8

import torch

from modules.hplstm.LGate import LGateFunc

cuda_device = None#torch.device("cuda", 0)
if cuda_device is not None:
	torch.cuda.set_device(cuda_device.index)

a = torch.randn(2, 3, 2, 5, requires_grad=True, device=cuda_device)
b = torch.randn(2, 3, 2, 5, requires_grad=True, device=cuda_device)
c = torch.randn(2, 5, requires_grad=True, device=cuda_device)
rs = LGateFunc(a.sigmoid(), b.clone(), c, True)
rs.sum().backward()
ag = a.grad.clone()
bg = b.grad.clone()
cg = c.grad.clone()
a.grad = b.grad = c.grad = None
rsl = []
_ag = a.sigmoid()
for i in range(b.size(1)):
	if i == 0:
		rsl.append(c * _ag.select(1, 0) + b.select(1,0))
	else:
		rsl.append(rsl[-1] * _ag.select(1, i) + b.select(1,i))
rsl = torch.stack(rsl, 1)
print(rs)
print(rsl)
rsl.sum().backward()
print(ag)
print(a.grad)
print(bg)
print(b.grad)
print(cg)
print(c.grad)
