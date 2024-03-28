#encoding: utf-8

from random import random
from re import compile

from utils.fmt.base import sys_open
from utils.fmt.rext import SubPair, SubPairP

mdate_search = compile("\d+月\d+日").search
renminzhengfu_sub = compile("[省市县区]人民政府").sub
jiezhi_search = compile("截至\d+").search
didei_sub = compile("地|得").sub

tihuan_file = "custdata/随机替换.txt"
huhuan_file = "custdata/随机互换.txt"
shanchu_file = "custdata/随机删除.txt"

def reader(fname):

	with sys_open(fname, "rb") as f:
		for line in f:
			tmp = line.strip()
			if tmp:
				tmp = tmp.decode("utf-8")
				if not tmp.startswith("#"):
					tmp = tuple(_ for _ in tmp.split() if _)
					yield tmp

def load_pairs(fname):

	rs = set()
	for _ in reader(fname):
		if len(_) == 2:
			if _ not in rs:
				rs.add(_)
	rs = list(rs)
	rs.sort()

	return tuple(rs)

def make_pairs(lin):

	_ = tuple(set(lin))
	for i, s in enumerate(_):
		for j, t in enumerate(_):
			if i != j:
				yield s, t

def load_p_pairs(fname):

	prs = set()
	for _ in reader(fname):
		_ = tuple(set(_))
		if len(_) > 1:
			if _ not in prs:
				prs.add(_)
	rs = set()
	for _ in prs:
		if len(_) == 2:
			if _ not in rs:
				rs.add(_)
		else:
			rs |= set(make_pairs(_))

	return tuple(sorted(rs))

def load_del(fname):

	rs = set()
	for _ in reader(fname):
		if len(_) == 1:
			_ = _[0]
			if _ not in rs:
				rs.add(_)

	return tuple((_, "",) for _ in sorted(rs))

yicuo_sub = SubPair(load_pairs(tihuan_file), allow_re=False).handle
yicuo_p_sub = SubPairP(tuple(sorted(load_p_pairs(huhuan_file) + load_del(shanchu_file))), allow_re=False, p=0.1).handle

def mdate_err(x, *args, sfunc=mdate_search, **kwargs):

	_m = sfunc(x)
	if _m is not None:
		_ = _m.end()
		return "%s月%s" % (x[:_ - 1], mdate_err(x[_:], sfunc=sfunc),)

	return x

def renminzhengfu_err(x, *args, sfunc=renminzhengfu_sub, **kwargs):

	return sfunc("人民政府", x)

def jiezhi_err(x, *args, sfunc=jiezhi_search, **kwargs):

	_m = sfunc(x)
	if _m is not None:
		_ = _m.start()
		return "%s截止%s" % (x[:_], jiezhi_err(x[_ + 2:], sfunc=sfunc),)

	return x

def yicuo_err(x, *args, sfunc=yicuo_sub, **kwargs):

	return sfunc(x)

def yicuo_p_err(x, *args, sfunc=yicuo_p_sub, **kwargs):

	return sfunc(x)

def didei_err(x, *args, sfunc=didei_sub, **kwargs):

	return sfunc("的", x)

def dedidei_err(x, *args, p=0.1, p_di=0.7, **kwargs):

	if random() < p:
		return x.replace("的", "地" if random() < p_di else "得")

	return x

cust_err_funcs = [mdate_err, renminzhengfu_err, jiezhi_err, yicuo_err, yicuo_p_err, didei_err, dedidei_err]
