#encoding: utf-8

import sys

from utils.fmt.base import FileList, sys_open

def handle(srcf, osfl, interval=15, max_id=3, **kwargs):

	fwd = {}

	ens = "\n".encode("utf-8")
	with FileList([srcf] + osfl, "rb") as rdl:
		for lind in zip(*rdl):
			li = [lu.strip().decode("utf-8") for lu in lind]
			src = li[0]
			if src:
				lts = len(src.split())
				wid = min(lts // interval, max_id)
				if wid in fwd:
					fwrtl = fwd[wid]
				else:
					sind = str(wid) + "_"
					fwrtl = [sys_open(sind + srcf, "wb") for srcf in osfl]
					fwd[wid] = fwrtl
				for wl, fw in zip(li[1:], fwrtl):
					fw.write(wl.encode("utf-8"))
					fw.write(ens)

	for k, fwl in fwd.items():
		for f in fwl:
			f.close()

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2:])
