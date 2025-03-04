#encoding: utf-8

def wrap(outer):

	def inner_constructor(inner):

		def constructor(*args, **kwargs):

			return outer(inner, *args, **kwargs)

		return constructor

	return inner_constructor
