#encoding: utf-8

def shift_rms_weight(x, value=1.0, **kwargs):

	return x.add_(value)
