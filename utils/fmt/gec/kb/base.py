#encoding: utf-8

from utils.fmt.diff import seq_diff_reorder_insert as seq_diff

from cnfg.vocab.gec.edit import blank_id, delete_id, init_vocab as edit_vcb, insert_id, keep_id, mask_id as edit_mask_id
from cnfg.vocab.gec.op import append_id, pad_id as op_pad_id
from cnfg.vocab.plm.custbert import eos_id, mask_id, pad_id

edit_vcb_diff = edit_vcb.copy()
edit_vcb_diff["e"] = keep_id

def apply_op_ids(i, k, e, t, mask_id=mask_id, append_id=append_id, keep_id=keep_id, delete_id=delete_id, edit_mask_id=edit_mask_id, insert_id=insert_id, blank_id=blank_id, pad_id=pad_id, eos_id=eos_id):

	_k = k if k else (blank_id for _ in range(len(i)))
	rsi, rsk, rse = [], [], []
	for _iu, _ku, _eu, _tu in zip(i, _k, e, t):
		if _iu == pad_id:
			break
		if _tu == append_id:
			rsi.append(_iu)
			rsi.append(mask_id)
			rsk.append(_ku)
			rsk.append(mask_id)
			rse.append(keep_id if _eu == blank_id else _eu)
			rse.append(edit_mask_id)
		else:
			if _iu == mask_id:
				rsi.append(_tu)
				rsk.append(blank_id)
				rse.append(insert_id)
			else:
				rsi.append(_iu)
				rsk.append(_ku)
				rse.append(_tu if _eu == blank_id else _eu)
	if rsi[-1] == eos_id:
		rse[-1] = keep_id

	return rsi, rsk, rse

def merge_src_kb(src, kb=None, edit_vcb_diff=edit_vcb_diff):

	if kb:
		if src == kb:
			kbf, m = tuple(keep_id for _ in range(len(src))), src
		else:
			kbf, m = zip(*seq_diff(src, kb))
			kbf = tuple(edit_vcb_diff[_] for _ in kbf)
	else:
		kbf, m = tuple(blank_id for _ in range(len(src))), src

	return m, kbf

def generate_iter_data(src, kb, tgt, mask_id=mask_id, edit_mask_id=edit_mask_id, blank_id=blank_id, delete_id=delete_id, insert_id=insert_id, keep_id=keep_id, append_id=append_id, eos_id=eos_id, op_pad_id=op_pad_id):

	m_src, _kbf = merge_src_kb(src, kb)
	_src, _tgt, _il, _iu, _prev_op = [], [], [], [], None
	for _op, _token in seq_diff(m_src, tgt):
		if _op == "i":
			_iu.append(_token)
			if _prev_op == "e":
				_tgt[-1] = append_id
		else:
			if _iu:
				_il.append(_iu)
				_iu = []
			_src.append(_token)
			_tgt.append(delete_id if _op == "d" else keep_id)
		_prev_op = _op
	if _iu:
		_il.append(_iu)
		_iu = None
	_edit = [blank_id for _ in range(len(_src))]
	if _src[-1] == eos_id:
		_tgt[-1] = op_pad_id
	yield _src, _kbf, _edit, _tgt
	_handle_mask = True if _il else False
	while _il:
		_s, _k, _e, _t, _na = [], [], [], [], []
		_ril = 0
		for _su, _ku, _eu, _tu in zip(_src, _kbf, _edit, _tgt):
			if _tu == append_id:
				_s.append(_su)
				_s.append(mask_id)
				_k.append(_ku)
				_k.append(mask_id)
				_e.append(keep_id if _eu == blank_id else _eu)
				_e.append(edit_mask_id)
				_t.append(keep_id)
				_ = _il[_ril]
				_ril += 1
				_t.append(_.pop(0))
				_na.append(False)
				_na.append(True if _ else False)
			else:
				_s.append(_su)
				_k.append(_ku)
				_e.append(_tu if _eu == blank_id else _eu)
				_t.append(_tu)
				_na.append(False)
		if _e[-1] == op_pad_id:
			_e[-1] = keep_id
		_cid = []
		for _ind, _ in enumerate(_il):
			if not _:
				_cid.append(_ind)
		for _ in reversed(_cid):
			del _il[_]
		_src, _kbf, _edit, _tgt = _s, _k, _e, _t
		yield _src, _kbf, _edit, _tgt
		if any(_na):
			_s, _k, _e, _t = [], [], [], []
			for _su, _ku, _eu, _tu, _au in zip(_src, _kbf, _edit, _tgt, _na):
				if _su == mask_id:
					_s.append(_tu)
					_k.append(blank_id)
					_e.append(insert_id)
					_t.append(append_id if _au else keep_id)
				else:
					_s.append(_su)
					_k.append(_ku)
					_e.append(_eu)
					_t.append(append_id if _au else _tu)
			_src, _kbf, _edit, _tgt = _s, _k, _e, _t
			yield _src, _kbf, _edit, _tgt
	if _handle_mask:
		_s, _k, _e, _t = [], [], [], []
		for _su, _ku, _eu, _tu in zip(_src, _kbf, _edit, _tgt):
			if _su == mask_id:
				_s.append(_tu)
				_k.append(blank_id)
				_e.append(insert_id)
				_t.append(keep_id)
			else:
				_s.append(_su)
				_k.append(_ku)
				_e.append(_eu)
				_t.append(_tu)
		_src, _kbf, _edit, _tgt = _s, _k, _e, _t
		yield _src, _kbf, _edit, _tgt
