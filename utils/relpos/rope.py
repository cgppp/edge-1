#encoding: utf-8

import torch

def apply_rope_rot(x, sp, cp):

	_ = x.size()
	_x1, _x2 = x.view(*_[:-1], _[-1] // 2, 2).unbind(-1)

	return torch.stack((_x1 * cp - _x2 * sp, (_x2 * cp).addcmul(_x1, sp),), dim=-1).view_as(x)

def apply_rope_split(x, sp, cp):

	_x1, _x2 = x.tensor_split(2, -1)

	return torch.cat((_x1 * cp - _x2 * sp, (_x2 * cp).addcmul(_x1, sp),), dim=-1)

apply_rope = apply_rope_split
