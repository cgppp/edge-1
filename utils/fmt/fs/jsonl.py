#encoding: utf-8

from json import loads

from utils.fmt.base import sys_open

def line_reader(fname, print_func=print, **kwargs):

	with sys_open(fname, "rb") as f:
		for _ in f:
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
