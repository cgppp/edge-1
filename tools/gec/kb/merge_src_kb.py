#encoding: utf-8

import sys

from utils.fmt.base import iter_to_int, iter_to_str, sys_open
from utils.fmt.gec.kb.base import merge_src_kb

def handle(srcf, kbf, rssf, rskf, **kwargs):

	ens = "\n".encode("utf-8")
	with sys_open(srcf, "rb") as frds, sys_open(kbf, "rb") as frdk, sys_open(rssf, "wb") as fwrts, sys_open(rskf, "wb") as fwrtk:
		for s, k in zip(frds, frdk):
			_s, _k = s.strip(), k.strip()
			if _s:
				_s, _k = merge_src_kb(tuple(iter_to_int(_s.decode("utf-8").split())), tuple(iter_to_int(_k.decode("utf-8").split())))
				fwrts.write(" ".join(iter_to_str(_s)).encode("utf-8"))
				fwrts.write(ens)
				fwrtk.write(" ".join(iter_to_str(_k)).encode("utf-8"))
				fwrtk.write(ens)

if __name__ == "__main__":
	handle(*sys.argv[1:5])
