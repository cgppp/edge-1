#encoding: utf-8

from utils.fmt.base import sys_open
from utils.ifilter.decode import ifilter

def line_reader(fname, encoding="utf-8", print_func=print, **kwargs):

	with sys_open(fname, "rb") as f:
		for _ in ifilter(f, encoding=encoding):
			rs = _.rstrip() if _ else ""
			yield rs
