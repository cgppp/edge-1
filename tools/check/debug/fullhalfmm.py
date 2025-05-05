#encoding: utf-8

import torch
from tqdm import tqdm

n, a, b, device = 1000, 1024, 1024, None

af = torch.randn(a, b, dtype=torch.float32, device=device)
bf = torch.randn(b, a, dtype=torch.float32, device=device)

ah, bh = af.to(torch.float16, non_blocking=True), bf.to(torch.float16, non_blocking=True)

o = af.mm(bf)
for _ in tqdm(range(n)):
	o = af.mm(bf)

o = ah.mm(bh)
for _ in tqdm(range(n)):
	o = ah.mm(bh)
