#encoding: utf-8

import torch
from functools import wraps

def split_wrapper(block_size=4096):

	def wrapper_builder(func):

		@wraps(func)
		def split_core(m, iQ, mask=None, states=None, mask_func=None, **kwargs):

			_nquery = iQ.size(1)
			if _nquery > block_size:
				rs = []
				_state = (None, None,) if states is None else states
				_sid = 0
				while _sid < _nquery:
					_eid = min(_sid + block_size, _nquery)
					_ = _eid - _sid
					_out, _state = func(m, iQ.narrow(1, _sid, _), mask=(mask if mask_func is None else mask_func(_eid, sid=_sid)) if mask is None else mask.narrow(-2, _sid, _).narrow(-1, 0, _eid), states=_state, **kwargs)
					rs.append(_out)
					_sid = _eid
				rs = torch.cat(rs, dim=1)

				return rs if states is None else (rs, _state,)

			return func(m, iQ, mask=mask, states=states, **kwargs)

		return split_core

	return wrapper_builder
