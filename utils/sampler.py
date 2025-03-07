#encoding: utf-8

import torch

from utils.torch.ext import multinomial

def SampleMax(x, dim=-1, keepdim=False, sample=False, top_k=1, top_p=0.0, temp=1.0):

	_legal_top_p = (top_p > 0.0) and (top_p < 1.0)
	out = x
	if (temp > 0.0) and (sample or _legal_top_p or (top_k > 1)):
		inds = None
		if top_k > 1:
			out, inds = out.topk(top_k)
		if temp != 1.0:
			out = out / temp
		out = out.softmax(dim)
		if _legal_top_p:
			if inds is None:
				out, inds = out.sort(dim=dim, descending=True)
			out = out.masked_fill(out.cumsum(dim).ge(top_p).to(torch.int32, non_blocking=True).cumsum(dim).gt(1), 0.0)
		out = multinomial(out, 1, replacement=True, dim=dim)
		if inds is not None:
			out = inds.gather(dim, out)
		if not keepdim:
			out = out.squeeze(dim)
	else:
		out = out.argmax(dim=dim, keepdim=keepdim)

	return out
