#encoding: utf-8

from utils.fmt.base import iter_all_eq

def lstrip(lin, xl):

	for _ in xl:
		if lin[0] == _:
			del lin[0]

	return lin

def add_prefix(lin, xl):

	if iter_all_eq(lin[:len(xt)], xl):

		return lin

	return xl + lin
