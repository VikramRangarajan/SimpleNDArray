from __future__ import annotations

import array
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from .dtypes import Device


class Buffer:
    def __init__(self, data: array.array):
        self.data = data
        self.address, self.num_bytes = data.buffer_info()
        self.typecode = data.typecode
        self.num_bytes *= data.itemsize
        self.device: Device = "cpu"

    @classmethod
    def empty(cls, size: int, dtype: str) -> Buffer:
        # Not empty, full of zeros.
        return cls(array.array(dtype, [0]) * size)

    @classmethod
    def from_iterable(cls, data: Iterable[int | float | bool], dtype: str) -> Buffer:
        data = array.array(dtype, data)
        return cls(data)

    def __repr__(self) -> str:
        return repr(self.data)
