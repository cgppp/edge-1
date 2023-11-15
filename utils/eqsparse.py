#encoding: utf-8

from math import ceil
from random import sample, shuffle

def gen_random_conn_inds(in_features, out_features, num_conn):

	_ = list(range(in_features))
	for _i in range(ceil(float(out_features * num_conn) / in_features)):
		shuffle(_)
		yield from _

def reorder_conn_ids_core(idl, num_conn):

	_cache = []
	_iset = set()
	_id = 0
	for _ in idl:
		if _ in _iset:
			_cache.append(_)
		else:
			_iset.add(_)
			_id += 1
			yield _
			if _id == num_conn:
				_iset.clear()
				_id = 0
				if _cache:
					while _cache:
						_rl = []
						_break = True
						for _i, _c in enumerate(_cache):
							if _c not in _iset:
								_iset.add(_c)
								_id += 1
								_rl.append(_i)
								yield _c
							if _id == num_conn:
								_iset.clear()
								_id = 0
								_break = False
								break
						if _rl:
							for _i in reversed(_rl):
								del _cache[_i]
						if _break:
							break
	if _cache:
		while _cache:
			_rl = []
			for _i, _c in enumerate(_cache):
				if _c not in _iset:
					_iset.add(_c)
					_rl.append(_i)
					yield _c
			if _rl:
				for _i in reversed(_rl):
					del _cache[_i]
			_iset.clear()

def reorder_conn_ids(idl, num_conn):

	return reorder_conn_ids_core(reversed(list(reorder_conn_ids_core(idl, num_conn))), num_conn)

def build_random_conn_inds_eq(in_features, out_features, num_conn):

	_ = list(reorder_conn_ids(gen_random_conn_inds(in_features, out_features, num_conn), num_conn))
	_nkeep = out_features * num_conn

	return _ if len(_) == _nkeep else _[:_nkeep]

def build_random_conn_inds_samp(in_features, out_features, num_conn):

	_ = list(range(in_features))
	rs = []
	for _i in range(out_features):
		rs.extend(sample(_, num_conn))

	return rs

def build_random_conn_inds(in_features, out_features, num_conn):

	if in_features == out_features:
		return build_random_conn_inds_eq(in_features, out_features, num_conn)
	else:
		return build_random_conn_inds_samp(in_features, out_features, num_conn)
