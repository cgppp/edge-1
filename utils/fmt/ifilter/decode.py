#encoding: utf-8

def ifilter(x, *args, encoding="utf-8", print_func=print, **kwargs):

	for _ in x:
		try:
			rs = _.decode(encoding=encoding, errors="strict")
		except Exception as e:
			rs = ""
			if print_func is not None:
				print_func(e)
		yield rs
