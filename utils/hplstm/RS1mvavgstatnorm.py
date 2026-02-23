#encoding: utf-8

import torch

from utils.hplstm.MvAvgis import MvAvgiSFunc
from utils.hplstm.RS1MvAvg import RS1MvAvgFunc
from utils.hplstm.RS1MvAvgstat import RS1MvAvgstatFunc

def RS1mvavgstatnorm(x, normer, states=None, ma_beta=None, detach=False, contiguous_state=False, **kwargs):

	bsize, seql, nheads, adim = x.size()
	_x = x.detach() if detach else x
	if states is None:
		csum, csum_state_return = normer(RS1MvAvgFunc(_x, ma_beta)), None
	else:
		if states == "init":
			if seql > 1:
				csum, csum_state_return = RS1MvAvgstatFunc(_x, ma_beta)
				csum = normer(csum)
				if cont_state:
					csum_state_return = csum_state_return.contiguous()
			else:
				csum, csum_state_return = normer(_x.new_zeros(1, 1, nheads, adim)).expand(bsize, 1, nheads, adim), _x * (1.0 - ma_beta)
		else:
			_csum_state = states[0]
			if seql > 1:
				_ = MvAvgiSFunc(torch.cat((_csum_state, _x,), dim=1), ma_beta, True)
				csum, csum_state_return = normer(_.narrow(1, 0, seql)), _.narrow(1, seql, 1)
				if cont_state:
					csum_state_return = csum_state_return.contiguous()
			else:
				csum, csum_state_return = normer(_csum_state), _csum_state.mul_(ma_beta).add_(_x, alpha=1.0 - ma_beta)

	return csum, csum_state_return
