#encoding: utf-8

import torch

from utils.hplstm.RS1cumsum import RS1cumsumFunc

cuda_device = None#torch.device("cuda", 0)
if not torch.cuda.is_available():
	cuda_device = None
if cuda_device is not None:
	torch.cuda.set_device(cuda_device.index)

x = torch.randn(2, 3, 2, 5, requires_grad=True, device=cuda_device)
rs = RS1cumsumFunc(x)
rs.sum().backward()
xg = x.grad.clone()
x.grad = None
rsl = torch.cat((x.new_zeros(x.narrow(1, 0, 1).size()), x.narrow(1, 0, x.size(1) - 1).cumsum(1),), dim=1)
print(rs)
print(rsl)
rsl.sum().backward()
print(xg)
print(x.grad)
