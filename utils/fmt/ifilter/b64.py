#encoding: utf-8

from base64 import b64decode

def ifilter(x, *args, validate=True, print_func=print, **kwargs):

	for _ in x:
		try:
			rs = b64decode(_, validate=validate)
		except Exception as e:
			rs = b""
			if print_func is not None:
				print_func(e)
		yield rs
