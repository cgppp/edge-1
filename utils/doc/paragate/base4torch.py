#encoding: utf-8

import torch

def clear_pad_mask(batch_in, mask, dim=-1, mask_dim=-1):

	npad = mask.to(torch.int32, non_blocking=True).sum(mask_dim).min().item()
	if npad > 0:
		seql = batch_in.size(dim)
		return batch_in.narrow(dim, 0, seql - npad), mask.narrow(mask_dim, 0, seql - npad)
	else:
		return batch_in, mask
