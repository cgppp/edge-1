#encoding: utf-8

from random import random
from re import compile

from utils.fmt.rext import SubPair

mdate_search = compile("\d+月\d+日").search
renminzhengfu_sub = compile("[省市县区]人民政府").sub
jiezhi_search = compile("截至\d+").search
didei_sub = compile("地|得").sub

yicuo_pairs = (("登录", "登陆",),)
yicuo_sub = SubPair(yicuo_pairs, allow_re=False).handle

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

def didei_err(x, *args, sfunc=didei_sub, **kwargs):

	return sfunc("的", x)

def dedidei_err(x, *args, p=0.1, p_di=0.7, **kwargs):

	if random() < p:
		return x.replace("的", "地" if random() < p_di else "得")

	return x

cust_err_funcs = [mdate_err, renminzhengfu_err, jiezhi_err, yicuo_err, didei_err, dedidei_err]
