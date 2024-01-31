#encoding: utf-8

import torch
from tqdm import tqdm

from modules.hplstm.LGate import LGateFunc
from modules.hplstm.LGateC import LGateFunc as LGateFuncC

cuda_device = torch.device("cuda", 0)
torch.cuda.set_device(cuda_device.index)

bsize, seql, nhead, isize = 32, 128, 8, 64
inplace = False

ntest = 5000

igh = torch.randn(bsize, seql, nhead, isize, requires_grad=True, device=cuda_device)
fg = torch.randn(bsize, seql, nhead, isize, requires_grad=True, device=cuda_device)
ic = torch.randn(nhead, isize, requires_grad=True, device=cuda_device)

ol = LGateFunc(fg.sigmoid(), igh.clone(), ic, 1, inplace)
#print(ol)
ol.sum().backward()
#print(igh.grad)
#print(fg.grad)
#print(ic.grad)
igh.grad = fg.grad = ic.grad = None
oc = LGateFuncC(fg.sigmoid(), igh.clone(), ic, inplace)
#print(oc)
oc.sum().backward()
#print(igh.grad)
#print(fg.grad)
#print(ic.grad)
igh.grad = fg.grad = ic.grad = None
#exit()

for _ in tqdm(range(ntest)):
	LGateFunc(fg.sigmoid(), igh.clone(), ic, 1, inplace).sum().backward()
	igh.grad = fg.grad = ic.grad = None

for _ in tqdm(range(ntest)):
	LGateFuncC(fg.sigmoid(), igh.clone(), ic, inplace).sum().backward()
	igh.grad = fg.grad = ic.grad = None
