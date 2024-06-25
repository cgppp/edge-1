#encoding: utf-8

import torch

from utils.torch.c import MvAvgFunc

a = torch.randn(2,3,5,requires_grad=True)
beta = 0.9
rs = MvAvgFunc(a.clone(),1,beta,True)
rs.sum().backward()
ag = a.grad.clone()
a.grad.zero_()
mbeta = 1.0-beta
rsl = []
for i in range(3):
	if i == 0:
		tmp = mbeta * a.select(1, 0)
	else:
		tmp = tmp * beta + a.select(1, i) * mbeta
	rsl.append(tmp)
rsl = torch.stack(rsl,1)
print(rs)
print(rsl)
rsl.sum().backward()
print(ag)
print(a.grad)
