from functools import reduce
from itertools import accumulate
from operator import mul
from typing import Iterable


def product(iterable: Iterable[int]) -> int:
    return reduce(mul, iterable, 1)


def ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = tuple(accumulate(shape[::-1], mul, initial=1))[-2::-1]
    return strides


def all_eq(*args):
    return all([args[0] == arg for arg in args[1:]])


def broadcast_shapes_strides(
    *pairs: tuple[tuple[int, ...], tuple[int, ...]],
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    if not pairs:
        return []
    ndim = max(len(s) for s, _ in pairs)
    shapes = [list((1,) * (ndim - len(s)) + s) for s, _ in pairs]
    strides = [list((1,) * (ndim - len(st)) + st) for _, st in pairs]
    for d in range(ndim):
        dim_shapes = [sh[d] for sh in shapes]
        common = max(dim_shapes)
        for i, s in enumerate(dim_shapes):
            if s != common:
                if s != 1:
                    raise ValueError("Not Broadcastable Arrays")
                shapes[i][d] = common
                strides[i][d] = 0
    return [(tuple(shapes[i]), tuple(strides[i])) for i in range(len(pairs))]
