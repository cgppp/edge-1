#encoding: utf-8

from json import loads

from utils.fmt.base import sys_open
from utils.ifilter.utf8 import ifilter

def line_reader(fname, print_func=print, **kwargs):

	with sys_open(fname, "rb") as f:
		for _ in ifilter(f):
			if _:
				rs = _.rstrip()
				try:
					rs = loads(rs)
				except Exception as e:
					rs = ""
					if print_func is not None:
						print_func(e)
			else:
				rs = ""
			yield rs
