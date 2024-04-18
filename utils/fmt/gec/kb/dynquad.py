#encoding: utf-8

from math import ceil

from utils.fmt.base import get_bsize, pad_batch

from cnfg.vocab.gec.edit import pad_id as edit_pad_id
from cnfg.vocab.gec.op import pad_id as op_pad_id
from cnfg.vocab.plm.custbert import pad_id as mlm_pad_id

pad_id = (mlm_pad_id, edit_pad_id, edit_pad_id, op_pad_id,)

def batch_loader(finput, bsize, maxpad, maxpart, maxtoken, minbsize, get_bsize=get_bsize, file_reader=None, **kwargs):

	_f_maxpart = float(maxpart)
	rsi = []
	rsk = []
	rse = []
	rst = []
	nd = maxlen = mlen = 0
	for i_d, kd, ed, td in (finput if file_reader is None else file_reader(finput)):
		i_d, kd, ed, td = list(i_d), list(kd), list(ed), list(td)
		lgth = len(i_d)
		if maxlen == 0:
			maxlen = lgth + min(maxpad, ceil(lgth / _f_maxpart))
			_bsize = get_bsize(maxlen, maxtoken, bsize)
		if (nd < minbsize) or (lgth <= maxlen and nd < _bsize):
			rsi.append(i_d)
			rsk.append(kd)
			rse.append(ed)
			rst.append(td)
			if lgth > mlen:
				mlen = lgth
			nd += 1
		else:
			yield rsi, rsk, rse, rst, mlen
			rsi = [i_d]
			rsk = [kd]
			rse = [ed]
			rst = [td]
			mlen = lgth
			maxlen = lgth + min(maxpad, ceil(lgth / _f_maxpart))
			_bsize = get_bsize(maxlen, maxtoken, bsize)
			nd = 1
	if rsi:
		yield rsi, rsk, rse, rst, mlen

def batch_padder(finput, bsize, maxpad, maxpart, maxtoken, minbsize, pad_batch=pad_batch, batch_loader=batch_loader, pad_id=pad_id, **kwargs):

	_mlm_pad_id, _kb_pad_id, _edit_pad_id, _op_pad_id = pad_id
	for i_d, kd, ed, td, mlen in batch_loader(finput, bsize, maxpad, maxpart, maxtoken, minbsize, **kwargs):
		yield pad_batch(i_d, mlen, pad_id=_mlm_pad_id), pad_batch(kd, mlen, pad_id=_kb_pad_id), pad_batch(ed, mlen, pad_id=_edit_pad_id), pad_batch(td, mlen, pad_id=_op_pad_id)
