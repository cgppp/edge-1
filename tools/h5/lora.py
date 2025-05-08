#encoding: utf-8

import sys

from utils.h5serial import h5load, h5save
from utils.lora.para import merge_lora

from cnfg.ihyp import *

def handle(srcfl, rsf, h5args=h5zipargs, **kwargs):

	d = None
	for _ in srcfl:
		_d = h5load(_, restore_list=False)
		if d is None:
			d = _d
		else:
			d.update(_d)
		_d = None
	d = merge_lora(d, inplace=True, transpose=True)
	h5save(d, rsf, h5args=h5args)

if __name__ == "__main__":
	handle(sys.argv[1:-1], sys.argv[-1])
