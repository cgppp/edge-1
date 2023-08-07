#encoding: utf-8

from re import compile

mdate_search = compile("\d+月\d+日").search

def mdate_err(x, *args, sfunc=mdate_search, **kwargs):

	_m = sfunc(x)
	if _m is not None:
		_ = _m.end()
		return "%s月%s" % (x[:_ - 1], mdate_err(x[_:], sfunc=sfunc),)

	return x

cust_err_funcs = [mdate_err]
