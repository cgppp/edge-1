#encoding: utf-8

from functools import wraps

def wrap(outer):

	def inner_constructor(inner):

		@wraps(outer)
		def constructor(*args, **kwargs):

			return outer(inner, *args, **kwargs)

		return constructor

	return inner_constructor
