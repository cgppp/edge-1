#encoding: utf-8

# usage: python tools/clean/rank.py srcf tgtf rankf rssf rstf threshold

import sys

from utils.fmt.base import FileList, sys_open

def handle(srcfl, rankf, rsfl, threshold, descend=False, **kwargs):

	ndata = nkeep = 0
	ens = "\n".encode("utf-8")
	_comp_func = (lambda a, b: a >= b) if descend else (lambda a, b: a <= b)
	with FileList(srcfl, "rb") as fsrc, sys_open(rankf, "rb") as fs, FileList(rsfl, "wb") as fwrt:
		for lines in zip(*fsrc, fs):
			srcl = [_.strip() for _ in lines]
			if all(srcl):
				srcl, s = [_.decode("utf-8") for _ in srcl[:-1]], float(srcl[-1].decode("utf-8"))
				if _comp_func(s, threshold):
					for _, _f in zip(srcl, fwrt):
						_f.write(_.encode("utf-8"))
						_f.write(ens)
					nkeep += 1
				ndata += 1

	print("%d in %d data keeped with ratio %.2f" % (nkeep, ndata, float(nkeep) / float(ndata) * 100.0 if ndata > 0 else 0.0))

if __name__ == "__main__":
	_ = (len(sys.argv) - 1) // 2
	handle(sys.argv[1:_], sys.argv[_], sys.argv[_ + 1:-1], float(sys.argv[-1]))
