#encoding: utf-8

import torch

from modules.normer import MRNormer, MinNormer
from utils.torch.comp import mask_tensor_type

normer = torch.nn.Softmax(-1)

def pos2p(num_pos, length, scale=1.0, sid=0, device=None, dtype=torch.float32, normer=normer):

	_inds = torch.arange(length, device=device, dtype=dtype).unsqueeze(0)
	_ = torch.arange(sid, length, device=device, dtype=dtype)
	if sid == 0:
		_[0] = 1.0
	_d = ((_inds / _.unsqueeze(-1)).unsqueeze(-1) - torch.arange(num_pos, device=device, dtype=dtype).div_(num_pos - 1)).abs_()
	_m = torch.ones(length - sid, length, device=device, dtype=mask_tensor_type).triu_(sid + 1)

	return normer(_d.neg_() if scale == 1.0 else _d.mul_(-scale)).masked_fill_(_m.unsqueeze(-1), 0.0)
