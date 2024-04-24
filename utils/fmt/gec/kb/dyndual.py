#encoding: utf-8

from math import ceil

from utils.fmt.base import get_bsize, pad_batch

from cnfg.vocab.gec.edit import pad_id as edit_pad_id
from cnfg.vocab.gec.op import pad_id as op_pad_id
from cnfg.vocab.plm.custbert import pad_id as mlm_pad_id

pad_id = (mlm_pad_id, edit_pad_id)

def batch_loader(finput, bsize, maxpad, maxpart, maxtoken, minbsize, get_bsize=get_bsize, file_reader=None, iter_to_int=iter_to_int, **kwargs):

	_f_maxpart = float(maxpart)
	rsi = []
	rsk = []
	nd = maxlen = mlen = 0
	for i_d, kd in (finput if file_reader is None else file_reader(finput)):
		i_d, kd = list(i_d), list(kd)
		lgth = len(i_d)
		if maxlen == 0:
			maxlen = lgth + min(maxpad, ceil(lgth / _f_maxpart))
			_bsize = get_bsize(maxlen, maxtoken, bsize)
		if (nd < minbsize) or (lgth <= maxlen and nd < _bsize):
			rsi.append(i_d)
			rsk.append(kd)
			if lgth > mlen:
				mlen = lgth
			nd += 1
		else:
			yield rsi, rsk, mlen
			rsi = [i_d]
			rsk = [kd]
			mlen = lgth
			maxlen = lgth + min(maxpad, ceil(lgth / _f_maxpart))
			_bsize = get_bsize(maxlen, maxtoken, bsize)
			nd = 1
	if rsi:
		yield rsi, rsk, mlen

def batch_padder(finput, bsize, maxpad, maxpart, maxtoken, minbsize, pad_batch=pad_batch, batch_loader=batch_loader, pad_id=pad_id, **kwargs):

	_mlm_pad_id, _kb_pad_id = pad_id
	for i_d, kd, mlen in batch_loader(finput, bsize, maxpad, maxpart, maxtoken, minbsize, **kwargs):
		yield pad_batch(i_d, mlen, pad_id=_mlm_pad_id), pad_batch(kd, mlen, pad_id=_kb_pad_id)
