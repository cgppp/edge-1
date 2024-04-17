#encoding: utf-8

import sys

from utils.fmt.base import iter_to_int, iter_to_str, sys_open
from utils.fmt.gec.kb.base import merge_src_kb

from cnfg.vocab.gec.edit import blank_id

def handle(srcf, kbf, rssf, rskf):

	ens = "\n".encode("utf-8")
	with sys_open(srcf, "rb") as frds, sys_open(kbf, "rb") as frdk, sys_open(rssf, "wb") as fwrts, sys_open(rskf, "wb") as fwrtk:
		for s, k in zip(frds, frdk):
			_s, _k = s.strip(), k.strip()
			if _s:
				if _k:
					_s, _k = tuple(iter_to_int(_s.decode("utf-8").split())), tuple(iter_to_int(_k.decode("utf-8").split()))
					_src, _kb = merge_src_kb(_s, _k)
				else:
					_src, _kb = _s, (blank_id for _ in range(len(_s)))
				fwrts.write(" ".join(iter_to_str(_src)).encode("utf-8"))
				fwrts.write(ens)
				fwrtk.write(" ".join(iter_to_str(_kb)).encode("utf-8"))
				fwrtk.write(ens)

if __name__ == "__main__":
	handle(*sys.argv[1:5])
