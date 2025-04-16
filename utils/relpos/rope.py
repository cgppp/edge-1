#encoding: utf-8

import torch
from functools import wraps

def wrapper_partial_rope(rope_apply_func):

	@wraps(rope_apply_func)
	def apply_rope_partial(x, sp, cp, *args, **kwargs):

		_xd, _rd = x.size(-1), sp.size(-1)
		_rd += _rd

		return rope_apply_func(x, sp, cp, *args, **kwargs) if _xd == _rd else torch.cat((rope_apply_func(x.narrow(-1, 0, _rd), sp, cp, *args, **kwargs), x.narrow(-1, _rd, _xd - _rd),), dim=-1)

	return apply_rope_partial

@wrapper_partial_rope
def apply_rope_rot(x, sp, cp, *args, **kwargs):

	_ = x.size()
	_x1, _x2 = x.view(*_[:-1], _[-1] // 2, 2).unbind(-1)

	return torch.stack((_x1 * cp - _x2 * sp, (_x2 * cp).addcmul(_x1, sp),), dim=-1).view_as(x)

@wrapper_partial_rope
def apply_rope_split(x, sp, cp, *args, **kwargs):

	_x1, _x2 = x.tensor_split(2, -1)

	return torch.cat((_x1 * cp - _x2 * sp, (_x2 * cp).addcmul(_x1, sp),), dim=-1)

apply_rope = apply_rope_split
