#encoding: utf-8

import sys
import torch

from utils.h5serial import h5File, h5load, h5write_dict

from cnfg.ihyp import *

def torch_to_h5(srcfl, rsf, h5args=h5zipargs):

	with h5File(rsf, "w", **h5_fileargs) as h5f:
		for srcf in srcfl:
			h5write_dict(h5f, torch.load(srcf, map_location="cpu"), h5args=h5args)

def h5_to_torch(srcfl, rsf):

	_ = h5load(srcfl[0], restore_list=False)
	for srcf in srcfl[1:]:
		_.update(h5load(srcf, restore_list=False))
	torch.save(_, rsf)

def handle(srcfl, rsf, **kwargs):

	(torch_to_h5 if rsf.endswith(".h5") else h5_to_torch)(srcfl, rsf)

if __name__ == "__main__":
	handle(sys.argv[1:-1], sys.argv[-1])
