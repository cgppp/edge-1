#encoding: utf-8

import sys

from utils.h5serial import h5File

from cnfg.ihyp import *

def handle(srcfl, rsf, h5args=h5zipargs):

	with h5File(rsf, "w", **h5_fileargs) as h5fr:
		for srcf in srcfl:
			with h5File(srcf, "r", **h5_fileargs) as h5fs:
				for _k, _v in h5fs.items():
					h5fr.create_dataset(k, data=_v[()], **h5args)

if __name__ == "__main__":
	handle(sys.argv[1:-1], sys.argv[-1])
