#encoding: utf-8

""" usage:
	python tools/average_model.py $averaged_model_file.h5 $model1.h5 $model2.h5 ...
"""

import sys
import torch

from utils.h5serial import h5File, h5save
from utils.torch.comp import secure_type_map

from cnfg.ihyp import h5_fileargs, h5zipargs

def handle(srcfl, rsf):

	src_type = {}
	map_type = {}
	sec_rsm = {}
	nmodel = {}
	for modelf in srcfl:
		with h5File(modelf, "r", **h5_fileargs) as _nmp:
			for _n, _p in sec_rsm.items():
				if _n in _nmp:
					_ = torch.from_numpy(_nmp[_n][()])
					_m_type = map_type[_n]
					_p.add_(_ if _m_type is None else _.to(_m_type, non_blocking=True))
					nmodel[_n] += 1
			for _n, _p in _nmp.items():
				if _n not in sec_rsm:
					_p = torch.from_numpy(_p[()])
					src_type[_n] = _p_dtype = _p.dtype
					map_type[_n] = _m_type = secure_type_map.get(_p_dtype, None)
					sec_rsm[_n] = _p if _m_type is None else _p.to(_m_type, non_blocking=True)
					nmodel[_n] = 1

	for _n, _p in sec_rsm.items():
		_p.div_(float(nmodel[_n]))

	h5save({_n: _p if map_type[_n] is None else _p.to(src_type[_n], non_blocking=True) for _n, _p in sec_rsm.items()}, rsf, h5args=h5zipargs)

if __name__ == "__main__":
	handle(sys.argv[2:], sys.argv[1])
