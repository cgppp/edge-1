#encoding: utf-8

# usage: python tools/check/rank.py rankf number_of_data_keeped

import sys

from utils.fmt.base import sys_open

def handle(rankf, dkeep, descend=False, **kwargs):

	scores = []

	with sys_open(rankf, "rb") as f:
		for line in f:
			tmp = line.strip()
			if tmp:
				scores.append(float(tmp.decode("utf-8")))

	scores.sort(reverse=descend)

	print(scores[dkeep - 1])

if __name__ == "__main__":
	if len(sys.argv) > 3:
		handle(sys.argv[1], int(sys.argv[2]), bool(int(sys.argv[-1])))
	else:
		handle(sys.argv[1], int(sys.argv[2]))
