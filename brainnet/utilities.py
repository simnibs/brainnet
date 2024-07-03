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
    """Multiply two dicts entry-wise. All entries much match exactly."""
    if out is None:
        out = {}
    for k,v in values.items():
        if isinstance(v, dict):
            out[k] = {}
            multiply_dicts(v, weights[k], out[k], threshold)
        else:
            out[k] = v * weights[k] if (weights[k] >= threshold) else 0
    return out


def add_dict(a: dict, b: dict):
    """Add `b` to `a` entrywise (in place). If an entry in `b` doesn't exist in
    `a`, it is created.
    """
    for k,v in b.items():
        if k in a:
            if isinstance(v, dict):
                add_dict(a[k], v)
            else:
                a[k] += v
        else:
            a[k] = v
    return a


def increment_dict_count(a: dict, b: dict, value: int = 1):
    """For each entry in `b`, increment the value of the corresponding entry in
    `a` by 1 (in place). If an entry in `b` doesn't exist in `a`, it is
    created with a value of 1.

    NOTE a and b should match in the sense that if

        a = dict(a=1)
        b = dict(a=dict(x=2))

    this will overwrite a=1 with a=dict(x=1)

    but this works

        a = dict(a=dict(y=1))
        b = dict(a=dict(x=2))

    """
    for k,v in b.items():
        if k in a: # error if this is a leaf node in a but not in b!
            if isinstance(v, dict):
                increment_dict_count(a[k], v, value)
            else:
                a[k] += value
        else:
            if isinstance(v, dict):
                a[k] = {}
                increment_dict_count(a[k], v, value)
            else:
                a[k] = value
    return a


def divide_dict(a: dict, b: dict):
    """Divide values of `a` with those in `b`. Entries should match."""
    for k,v in a.items():
        if isinstance(v, dict):
            divide_dict(a[k], b[k])
        else:
            a[k] /= b[k]
    return a
