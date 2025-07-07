#encoding: utf-8

def identity_func(x, *args, **kwargs):

	return x

def always_true(*args, **kwargs):

	return True

def always_false(*args, **kwargs):

	return False

def try_set(obj, key, value, skip_none=False, check_existence=True):

	if skip_none and (value is None):

		return False

	_kl, _set, _obj = key.split("."), True, obj
	for _k in _kl[:-1]:
		if hasattr(_obj, _k):
			_obj = getattr(_obj, _k)
		else:
			_set = False
			break

	_k = _kl[-1]
	_set = _set and (hasattr(_obj, _k) or (not check_existence))
	if _set:
		setattr(_obj, _k, value)

	return _set
