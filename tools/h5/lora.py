#encoding: utf-8

import sys

from utils.h5serial import h5Dataset, h5File, h5FileList, h5write_data
from utils.lora.para import merge_lora

from cnfg.ihyp import *

def handle(srcfl, rsf, h5args=h5zipargs, inplace=True, transpose=True, h5_fileargs=h5_fileargs, **kwargs):

	d = {}
	with h5FileList(srcfl, "r", **h5_fileargs) as fsrc, h5File(rsf, "w", **h5_fileargs) as frs:
		for _ in fsrc:
			for k, v in _.items():
				d[k] = v
		d = merge_lora(d, inplace=True, transpose=True)
		for k, v in d.items():
			h5write_data(frs, k, v[()] if isinstance(v, h5Dataset) else v, h5args=h5args)

if __name__ == "__main__":
	handle(sys.argv[1:-1], sys.argv[-1])
