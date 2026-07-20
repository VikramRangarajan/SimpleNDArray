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
