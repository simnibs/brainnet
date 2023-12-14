def recursively_apply_function(d, fn, out=None):
    """Recursively call .item() on values."""
    if out is None:
        out = {}
    for k,v in d.items():
        if isinstance(v, dict):
            recursively_apply_function(v, fn, out)
        else:
            out[k] = fn(v)
    return out

def recursive_dict_update_(d0, d1):
    """`d1` is a subset of `d0`."""
    for k,v in d1.items():
        if isinstance(v, dict):
            recursive_dict_update_(d0[k], v)
        else:
            d0[k] = v

def recursive_dict_sum(d):
    """Recursive sum values of a dict containing only int/float values."""
    total = 0.0
    for v in d.values():
        if isinstance(v, dict):
            total += recursive_dict_sum(v)
        else:
            total += v
    return total

def recursive_dict_multiply(d, factor): # iop() # inplace operator from operator module
    """Multiply all entries of a dictionary by `factor`."""
    for k,v in d.items():
        if isinstance(v, dict):
            recursive_dict_multiply(v, factor)
        else:
            d[k] *= factor # iop(d[k], factor)  ---- operator.imul, operator.iadd, ...

def flatten_dict(d: dict, out: None | dict = None, prefix=None): # sep=":"
    if out is None:
        out = {}
    for k,v in d.items():
        # key = k if prefix is None else sep.join((prefix, k))
        if prefix is None:
            key = k
        elif isinstance(prefix, tuple):
            key = (*prefix, k)
        else:
            key = (prefix, k)
        if isinstance(v, dict):
            flatten_dict(v, out, key)
        else:
            out[key] = v
    return out

def multiply_dicts(values, weights, out: None | dict = None, threshold=1e-8):
    """Multiply two dicts entry-wise. All entries much match exact."""
    if out is None:
        out = {}
    for k,v in values.items():
        if isinstance(v, dict):
            out[k] = {}
            multiply_dicts(v, weights[k], out[k], threshold)
        else:
            out[k] = v * weights[k] if (weights[k] >= threshold) else 0
    return out
