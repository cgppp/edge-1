#encoding: utf-8

import sys
from safetensors.torch import load_file, save_file

from utils.h5serial import h5File, h5load, h5write_dict

from cnfg.ihyp import *

def safetensors_to_h5(srcfl, rsf, h5args=h5zipargs):

	with h5File(fname, "w", **h5_fileargs) as h5f:
		for srcf in srcfl:
			h5write_dict(h5f, load_file(srcf), h5args=h5args)

def h5_to_safetensors(srcf, rsf):

	save_file(h5load(srcf, restore_list=False), rsf)

def handle(srcf, rsf):

	_execf = h5_to_safetensors if srcf.endswith(".h5") else safetensors_to_h5
	_execf(srcf, rsf)

if __name__ == "__main__":
	handle(sys.argv[1:-1], sys.argv[-1])
