#encoding: utf-8

import sys
from random import seed as rpyseed, shuffle

from utils.fmt.base import FileList, dict_insert_list, dict_insert_set, iter_dict_sort, sys_open

# remove_same: reduce same data in the corpus
# shuf: shuffle the data of same score
# descend: in descending order

def handle(srcfl, rankf, tgtfl, remove_same=True, shuf=True, descend=False, **kwargs):

	_insert_func = dict_insert_set if remove_same else dict_insert_list
	data = {}

	with FileList(srcfl, "rb") as fl, sys_open(rankf, "rb") as fs:
		for lines in zip(*fl, fs):
			lines = [line.strip() for line in lines]
			if all(lines):
				lines, s = [line.decode("utf-8") for line in lines[:-1]], float(lines[-1].decode("utf-8"))
				data = _insert_func(data, tuple(line.encode("utf-8") for line in lines), s)

	ens = "\n".encode("utf-8")
	with FileList(tgtfl, "wb") as fl:
		for tmp in iter_dict_sort(data, reverse=descend, free=True):
			tmp = list(tmp)
			if len(tmp) > 1:
				if shuf:
					shuffle(tmp)
			for du, f in zip(zip(*tmp), fl):
				f.write(ens.join(du))
				f.write(ens)

if __name__ == "__main__":
	rpyseed(666666)
	_ = len(sys.argv)
	if (_ % 2) == 1:
		_ = (_ - 1) // 2
		handle(sys.argv[1:_], sys.argv[_], sys.argv[_ + 1:-1], descend=bool(int(sys.argv[-1])))
	else:
		_ = _ // 2
		handle(sys.argv[1:_], sys.argv[_], sys.argv[_ + 1:])
