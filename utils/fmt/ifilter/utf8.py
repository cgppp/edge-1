#encoding: utf-8

def ifilter(x, *args, print_func=print, **kwargs):

	for _ in x:
		try:
			rs = _.decode("utf-8", errors="strict")
		except Exception as e:
			rs = ""
			if print_func is not None:
				print_func(e)
		yield rs
