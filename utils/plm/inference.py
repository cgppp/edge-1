#encoding: utf-8

from torch import Tensor

from utils.fmt.base import clean_empl_iter, get_common_prefix_len

def find_common_prefix(seql):

	_a = _mv = None
	for _ in seql:
		if _a is None:
			_a = _
			_mv = len(_)
		else:
			_cmv = get_common_prefix_len(_a, _)
			if _cmv < _mv:
				_mv = _cmv
			if _mv == 0:
				break

	return _a[:_mv]

def get_of_common_prefix(of, line_func=None, reset=True):

	_reset = reset and hasattr(of, "seek")
	if _reset:
		of.seek(0)
	rs = find_common_prefix(clean_empl_iter(of)) if line_func is None else find_common_prefix(line_func(_) for _ in clean_empl_iter(of))
	if _reset:
		of.seek(0)

	return rs if rs else None

def h5g_reader(h5g):

	for dset in h5g.values():
		for _ in dset[()].tolist():
			yield _

def get_h5g_common_prefix(h5g):

	rs = find_common_prefix(h5g_reader(h5g))

	return rs if rs else None

def expand_bsize(*inputs, bsize=1):

	outputs = []
	for inputu in inputs:
		if isinstance(inputu, Tensor):
			outputs.append(inputu.expand(bsize, *inputu.size()[1:]))
		elif isinstance(inputu, dict):
			outputs.append({k: expand_bsize(v, bsize=bsize) for k, v in inputu.items()})
		elif isinstance(inputu, tuple):
			outputs.append(tuple(expand_bsize(tmpu, bsize=bsize) for tmpu in inputu))
		elif isinstance(inputu, list):
			outputs.append([expand_bsize(tmpu, bsize=bsize) for tmpu in inputu])
		else:
			outputs.append(inputu)

	return outputs[0] if len(inputs) == 1 else tuple(outputs)

def prepare_states_bsize(states, bsize=1):

	return states if (states is None) or (bsize == 1) else expand_bsize(states, bsize=bsize)
