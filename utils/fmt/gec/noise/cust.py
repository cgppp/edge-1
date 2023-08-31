#encoding: utf-8

from re import compile

mdate_search = compile("\d+月\d+日").search
renminzhengfu_sub = compile("[省市县区]人民政府").sub
jiezhi_search = compile("截至\d+").search
didei_sub = compile("地|得").sub

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

def didei_err(x, *args, sfunc=didei_sub, **kwargs):

	return sfunc("的", x)

cust_err_funcs = [mdate_err, renminzhengfu_err, jiezhi_err, didei_err]
