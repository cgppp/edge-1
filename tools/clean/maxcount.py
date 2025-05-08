#encoding: utf-8

# usage: python tools/clean/maxcount.py srcf tgtf rssf rstf count

import sys

from utils.fmt.base import FileList

def count_token(strin):

	rsd = {}
	for _ in strin.split():
		rsd[_] = rsd.get(_, 0) + 1

	return rsd

def count_char(strin):

	rsd = {}
	for _ in strin:
		rsd[_] = rsd.get(_, 0) + 1

	return rsd

def handle(srcfl, rsfl, max_count=8, is_token=True, **kwargs):

	counts = [{} for _ in range(len(srcfl))]
	nkeep = ndata = 0
	_count_func = count_token if is_token else count_char
	ens = "\n".encode("utf-8")
	with FileList(srcfl, "rb") as fsrc, FileList(rsfl, "wb") as fwrt:
		for lines in zip(*fsrc):
			srcl = [_.strip() for _ in lines]
			if all(srcl):
				srcl = [_.decode("utf-8") for _ in srcl]
				_cl = [_count_func(_) for _ in srcl]
				if any(any(_fd.get(_, 0) < max_count for _ in _ld.keys()) for _ld, _fd in zip(_cl, counts)):
					for _ld, _fd in zip(_cl, counts):
						for _, _c in _ld.items():
							_fd[_] = _fd.get(_, 0) + _c
					for _, _f in zip(srcl, fwrt):
						_f.write(_.encode("utf-8"))
						_f.write(ens)
					nkeep += 1
				ndata += 1

	print("%d in %d data keeped with ratio %.2f" % (nkeep, ndata, float(nkeep) / float(ndata) * 100.0 if ndata > 0 else 0.0))

if __name__ == "__main__":
	_ = len(sys.argv)
	if (_ % 2) == 0:
		_ = _ // 2
		handle(sys.argv[1:_], sys.argv[_:-1], int(sys.argv[-1]))
	else:
		_ = (_ + 1) // 2
		handle(sys.argv[1:_], sys.argv[_:])
