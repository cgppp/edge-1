#encoding: utf-8

# usage: python comet-score.py $srcf $mtf [$ref] $model $rsf|/dev/null $ngpu ${load_bsize}

import sys
from comet import load_from_checkpoint

from utils.fmt.base import FileList, is_null_file, iter_to_str, sys_open
from utils.torch.comp import torch_inference_mode

def load_files(finputs, bsize=-1, keys=("src", "mt", "ref",)):

	rs = []
	_seg = bsize > 0
	_n = 0
	with FileList(finputs, "rb") as fl:
		for lines in zip(*fl):
			rs.append({_k: _v for _k, _v in zip(keys, [line.strip().decode("utf-8") for line in lines])})
			_n += 1
			if _seg and (_n >= bsize):
				yield rs
				rs.clear()
				_n = 0
	if rs:
		yield rs

def handle(finputs, fmodel, frs, ngpu=0, load_bsize=-1, model_bsize=8):

	model = load_from_checkpoint(fmodel)
	model.eval()
	model.half()
	sum_score = 0.0
	n = 0
	ens = "\n".encode("utf-8")
	with sys_open(frs, "wb") as fwrt, torch_inference_mode():
		_write = not is_null_file(fwrt)
		for data in load_files(finputs[:3], bsize=max(load_bsize, model_bsize) if load_bsize > 0 else load_bsize):
			scores = model.predict(data, batch_size=model_bsize, gpus=ngpu, progress_bar=False).scores
			if _write:
				fwrt.write("\n".join(iter_to_str(scores)).encode("utf-8"))
				fwrt.write(ens)
			sum_score += sum(scores)
			n += len(scores)

	return sum_score / float(n)

if __name__ == "__main__":
	print(handle(sys.argv[1:-4], sys.argv[-4], sys.argv[-3], int(sys.argv[-2]), int(sys.argv[-1])))
