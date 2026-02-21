#encoding: utf-8

import torch

from utils.hplstm.RS1cumsum import RS1cumsumFunc
from utils.hplstm.RS1cumsumstat import RS1cumsumstatFunc

def RS1cumsumstatnorm(x, normer, states=None, detach=True, contiguous_state=False, **kwargs):

	bsize, seql, nheads, adim = x.size()
	_x = x.detach() if detach else x
	if states is None:
		csum, csum_state_return = normer(RS1cumsumFunc(_x)), None
	else:
		if states == "init":
			if seql > 1:
				csum, csum_state_return = RS1cumsumstatFunc(_x)
				csum = normer(csum)
				if cont_state:
					csum_state_return = csum_state_return.contiguous()
			else:
				csum, csum_state_return = normer(_x.new_zeros(1, 1, nheads, adim)).expand(bsize, 1, nheads, adim), _x
		else:
			_csum_state = states[0]
			if seql > 1:
				_ = torch.cat((_csum_state, _x,), dim=1).cumsum_(dim=1)
				csum, csum_state_return = normer(_.narrow(1, 0, seql)), _.narrow(1, seql, 1)
				if cont_state:
					csum_state_return = csum_state_return.contiguous()
			else:
				csum, csum_state_return = normer(_csum_state), _csum_state + _x

	return csum, csum_state_return
