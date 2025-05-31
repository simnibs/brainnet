def unsqueeze_and_expand(a, b):
    """Unsqueeze and expand `a` to match the remaining dimensions of
    `b`, e.g., if

        a.shape = (2,5)
        b.shape = (2,10,3)

    will return (2,5,3) where `a` is expanded in the last dim.

    """
    n = a.ndim
    for _ in b.shape[n:]:
        a = a.unsqueeze(-1)
    return a.expand(*a.shape[:n], *b.shape[n:])


def expand_and_gather(t, index, dim):
    return t.gather(dim, unsqueeze_and_expand(index, t))


def atleast_nd_prepend(t, n):
    if t.ndim >= n:
        return t
    else:
        return atleast_nd_prepend(t[None], n)


def atleast_nd_append(t, n):
    if t.ndim >= n:
        return t
    else:
        return atleast_nd_append(t[..., None], n)
