#encoding: utf-8

import sys
from safetensors.torch import load_file, save_file

from utils.h5serial import h5File, h5load, h5write_dict

from cnfg.ihyp import *

def safetensors_to_h5(srcfl, rsf, h5args=h5zipargs):

	with h5File(rsf, "w", **h5_fileargs) as h5f:
		for srcf in srcfl:
			h5write_dict(h5f, load_file(srcf), h5args=h5args)

def h5_to_safetensors(srcfl, rsf):

	_ = h5load(srcfl[0], restore_list=False)
	for srcf in srcfl[1:]:
		_.update(h5load(srcf, restore_list=False))
	save_file(_, rsf)

def handle(srcfl, rsf):

	(safetensors_to_h5 if rsf.endswith(".h5") else h5_to_safetensors)(srcfl, rsf)

if __name__ == "__main__":
	handle(sys.argv[1:-1], sys.argv[-1])
