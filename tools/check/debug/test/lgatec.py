#encoding: utf-8

import torch
from tqdm import tqdm

from modules.hplstm.LGate import LGateFunc
from modules.hplstm.LGateC import LGateFunc as LGateFuncC

bsize, seql, nhead, isize = 128, 64, 8, 64
inplace = False

ntest = 100

igh = torch.randn(bsize, seql, nhead, isize, requires_grad=True)
fg = torch.randn(bsize, seql, nhead, isize, requires_grad=True)
ic = torch.randn(nhead, isize, requires_grad=True)

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

for _ in tqdm(range(ntest)):
	LGateFunc(fg.sigmoid(), igh.clone(), ic, 1, inplace)#.sum().backward()
	igh.grad = fg.grad = ic.grad = None

for _ in tqdm(range(ntest)):
	LGateFuncC(fg.sigmoid(), igh.clone(), ic, inplace)#.sum().backward()
	igh.grad = fg.grad = ic.grad = None
