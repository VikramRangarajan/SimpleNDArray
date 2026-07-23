from __future__ import annotations

from typing import Iterable

from simplendarray.dtypes import Device, DType

from .buffer import Buffer
from .buffer_cuda import BufferCuda
from .dtypes import get_dtype, typecode
from .kernels import (
    dispatch_arange,
    dispatch_bmm,
    dispatch_element_wise_binary,
    dispatch_element_wise_unary,
    dispatch_reduction,
    dispatch_reshape_copy,
)
from .utils import broadcast_shapes_strides, ceildiv, contiguous_strides, product

type BufferType = Buffer | BufferCuda

type Scalar = int | float | bool
type NestedIterable = Iterable[Scalar] | Iterable[NestedIterable]

buf_cls: dict[Device, type[Buffer | BufferCuda]] = {"cpu": Buffer, "gpu": BufferCuda}


def flatten_and_get_shape(
    data: NestedIterable | Scalar,
) -> tuple[list[Scalar], tuple[int, ...]]:
    if isinstance(data, (int, float, bool)):
        return [data], ()

    if not data:
        return [], (0,)

    flat: list[Scalar] = []

    cur_dim = list(data)
    first_elem = cur_dim[0]
    if isinstance(first_elem, Iterable):
        first_list = list(first_elem)
        first_flat, first_shape = flatten_and_get_shape(first_list)
    else:
        first_flat, first_shape = [first_elem], ()
    flat.extend(first_flat)
    shape = (len(cur_dim), *first_shape)

    for item in cur_dim[1:]:
        if isinstance(item, Iterable):
            sub_list = list(item)
            sub_flat, sub_shape = flatten_and_get_shape(sub_list)
        else:
            sub_flat, sub_shape = [item], ()
        if sub_shape != first_shape:
            raise ValueError(f"Jagged array: expected shape {first_shape}, got {sub_shape}")
        flat.extend(sub_flat)

    return flat, shape


class Array:
    @classmethod
    def from_iterable(cls, data: NestedIterable | Scalar, dtype: str | type[DType], device: Device = "cpu"):
        flat, shape = flatten_and_get_shape(data)
        _typecode = typecode(get_dtype(dtype))
        buffer = buf_cls[device].from_iterable(flat, _typecode)
        strides = contiguous_strides(shape)
        offset = 0
        return cls(buffer, shape, strides, offset)

    @classmethod
    def empty(cls, numel: int, dtype: str | type[DType], device: Device = "cpu") -> "Array":
        if numel < 0:
            raise ValueError("numel must be >= 0")
        _typecode = typecode(get_dtype(dtype))
        buffer = buf_cls[device].empty(numel, _typecode)
        return cls(buffer, (numel,), (1,), 0)

    @classmethod
    def arange(cls, numel: int, dtype: str | type[DType], device: Device = "cpu") -> "Array":
        if numel < 0:
            raise ValueError("numel must be >= 0")
        _typecode = typecode(get_dtype(dtype))
        buffer = buf_cls[device].empty(numel, _typecode)
        dispatch_arange(buffer, 0, 1, numel)
        return cls(buffer, (numel,), (1,), 0)

    def __init__(self, buffer: BufferType, shape: tuple[int, ...], strides: tuple[int, ...], offset: int):
        self.data = buffer
        self.shape = shape
        self.strides = strides
        self.offset = offset
        self.dtype = get_dtype(self.data.typecode)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def device(self) -> Device:
        return self.data.device

    @property
    def size(self) -> int:
        return product(self.shape)

    @property
    def is_contiguous(self):
        if self.ndim == 0 or 0 in self.shape:
            return True
        real_shape = tuple(x for x in self.shape if x > 1)
        real_stride = tuple(y for x, y in zip(self.shape, self.strides) if x > 1)
        return real_stride == contiguous_strides(real_shape)

    def __repr__(self) -> str:
        return f"Array({self.to_python()}, shape={self.shape}, strides={self.strides}, offset={self.offset})"

    def to_python(self, buf=None) -> NestedIterable:
        if buf is None:
            buf = self.data.data
        if self.ndim == 0:
            return buf[self.offset]
        nested = []
        for i in range(self.shape[0]):
            # Does [self[0, :, :, ...], self[1, :, :, ...], ..., self[shape[0] - 1, :, :, ...]]
            # Then recursively calls to_python on each of these children, until base case reached
            indexed = self[i, *(slice(None) for _ in range(self.ndim - 1))]
            nested.append(indexed.to_python(buf))
        return nested

    def squeeze(self, dims: int | Iterable[int]) -> "Array":
        if isinstance(dims, int):
            dims = [dims]
        dims = [d % self.ndim if self.ndim > 0 else d for d in dims]
        if self.ndim == 0 or any(self.shape[d] != 1 for d in dims):
            raise ValueError("Can only squeeze a non scalar array with length 1 dims, but shape is", self.shape)
        dims_set = set(dims)
        new_shape = tuple(s for i, s in enumerate(self.shape) if i not in dims_set)
        new_strides = tuple(stride for i, stride in enumerate(self.strides) if i not in dims_set)
        return Array(self.data, new_shape, new_strides, self.offset)

    def unsqueeze(self, dim: int | tuple[int, ...]) -> "Array":
        if isinstance(dim, int):
            dim = (dim,)
        if self.ndim == 0:
            raise ValueError("Cannot unsqueeze a 0-dimensional array")
        dims = sorted(d % (self.ndim + 1) for d in dim)
        new_shape = self.shape
        new_strides = self.strides
        shift = 0
        for d in dims:
            pos = d + shift
            new_shape = new_shape[:pos] + (1,) + new_shape[pos:]
            new_strides = new_strides[:pos] + (1,) + new_strides[pos:]
            shift += 1
        return Array(self.data, new_shape, new_strides, self.offset)

    def __getitem__(self, items: tuple[int | slice | ..., ...] | int | slice | ...):
        if not isinstance(items, tuple):
            items = (items,)
        num_ellipses = sum(1 if x == ... else 0 for x in items)
        if num_ellipses > 1:
            raise ValueError("Can only have at most 1 ellipsis")
        if num_ellipses == 1:
            idx = items.index(...)
            middle = (slice(None),) * (self.ndim - (len(items) - 1))
            items = items[:idx] + middle + items[idx + 1 :]
        elif len(items) < self.ndim:
            items = items + (slice(None),) * (self.ndim - len(items))
        if len(items) != self.ndim:
            raise ValueError("Must index the same number of dimensions as the array")
        new_shape = []
        new_strides = []
        new_offset = self.offset
        for shape, stride, item in zip(self.shape, self.strides, items):
            if isinstance(item, int):
                item = slice(item, item + 1).indices(shape)[0]
                new_offset += stride * item
            elif isinstance(item, slice):
                start, stop, step = item.indices(shape)
                if step > 0:
                    # Num elements in [start, stop) = stop - start
                    new_shape.append(max(0, ceildiv(stop - start, step)))
                else:
                    new_shape.append(max(0, ceildiv(start - stop, -step)))
                new_strides.append(stride * step)
                new_offset += stride * start
            else:
                raise TypeError(f"Index must be int or slice, got {type(item).__name__}")
        return Array(self.data, tuple(new_shape), tuple(new_strides), new_offset)

    def transpose(self, dims: Iterable[int]) -> "Array":
        dims = list(dims)
        if len(dims) != self.ndim:
            raise ValueError("Transpose dims needs to be the same length as array ndim")
        new_shape = tuple(self.shape[i] for i in dims)
        new_strides = tuple(self.strides[i] for i in dims)
        return Array(self.data, new_shape, new_strides, self.offset)

    @property
    def T(self) -> "Array":
        return self.transpose(range(self.ndim - 1, -1, -1))

    @property
    def mT(self) -> "Array":
        if self.ndim < 2:
            raise ValueError("matrix transpose with ndim < 2 is undefined")
        dims = tuple(range(self.ndim - 2)) + (-1, -2)
        return self.transpose(dims)

    def reshape(self, new_shape: int | tuple[int, ...]) -> "Array":
        numel = product(self.shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape,)
        num_n1s = sum(1 for s in new_shape if s == -1)
        for s in new_shape:
            if s < -1:
                raise ValueError("Dimension must be non-negative, or with a single -1")
        if num_n1s > 1:
            raise ValueError("only one dimension can be -1")
        if num_n1s == 1:
            known = product(x for x in new_shape if x != -1)
            if numel % known != 0:
                raise ValueError(
                    f"Cannot reshape array of size {numel} into shape "
                    f"{tuple(numel // known if s == -1 else s for s in new_shape)}"
                )
            replacement = numel // known
            new_shape = tuple(replacement if s == -1 else s for s in new_shape)
        new_strides = reshape_strides(self.shape, new_shape, self.strides)
        if new_strides is None:
            out_ndim = len(new_shape)
            new_strides = contiguous_strides(new_shape)
            new_buffer = buf_cls[self.device].empty(numel, self.data.typecode)
            inp_shape_buffer = buf_cls[self.device].from_iterable(self.shape, "l")
            inp_strides_buffer = buf_cls[self.device].from_iterable(self.strides, "l")
            out_shape_buffer = buf_cls[self.device].from_iterable(new_shape, "l")
            out_strides_buffer = buf_cls[self.device].from_iterable(new_strides, "l")
            inp_work_buffer = buf_cls[self.device].empty(self.ndim, "l")
            out_work_buffer = buf_cls[self.device].empty(out_ndim, "l")

            dispatch_reshape_copy(
                self,
                new_shape,
                new_buffer,
                inp_shape_buffer,
                inp_strides_buffer,
                inp_work_buffer,
                out_shape_buffer,
                out_strides_buffer,
                out_work_buffer,
            )
            return Array(new_buffer, new_shape, new_strides, 0)
        return Array(self.data, new_shape, new_strides, self.offset)

    @staticmethod
    def unary_op(op: str):
        def fn(self: Array):
            out = Array.empty(product(self.shape), self.dtype, self.device).reshape(self.shape)
            dispatch_element_wise_unary(self, out, op)
            return out

        return fn

    relu = unary_op("relu")
    exp = unary_op("exp")
    exp2 = unary_op("exp2")
    log = unary_op("log")
    log2 = unary_op("log2")
    log10 = unary_op("log10")
    relu = unary_op("relu")
    square = unary_op("square")
    sqrt = unary_op("sqrt")
    sin = unary_op("sin")
    cos = unary_op("cos")
    tan = unary_op("tan")
    asin = unary_op("asin")
    acos = unary_op("acos")
    atan = unary_op("atan")
    sinh = unary_op("sinh")
    cosh = unary_op("cosh")
    tanh = unary_op("tanh")

    @staticmethod
    def binary_op(op: str):
        def fn(self: Array, other) -> Array:
            if not isinstance(other, Array):
                other = Array.from_iterable(other, self.dtype, self.device)
            [(s1, st1), (s2, st2)] = broadcast_shapes_strides((self.shape, self.strides), (other.shape, other.strides))
            self = Array(self.data, s1, st1, self.offset)
            other = Array(other.data, s2, st2, other.offset)
            out = Array.empty(product(self.shape), self.dtype, self.device).reshape(self.shape)
            dispatch_element_wise_binary(self, other, out, op)
            return out

        return fn

    __add__ = binary_op("add")
    __radd__ = binary_op("add")
    __sub__ = binary_op("sub")
    __rsub__ = binary_op("sub")
    __mul__ = binary_op("mul")
    __rmul__ = binary_op("mul")
    __div__ = binary_op("div")
    __rdiv__ = binary_op("div")
    __truediv__ = binary_op("div")
    __rtruediv__ = binary_op("div")
    atan2 = binary_op("atan2")

    @staticmethod
    def reduction_op(op: str):
        def fn(self: Array, dims: tuple[int, ...] | None = None):
            if dims is None:
                dims = tuple(range(self.ndim))
            dims = tuple(d % self.ndim for d in dims)
            reduced_shape = tuple(x for i, x in enumerate(self.shape) if i not in dims)
            out = Array.empty(product(reduced_shape), self.dtype, self.device)
            dispatch_reduction(self, out, op, dims)
            return out.reshape(reduced_shape)

        return fn

    sum = reduction_op("add")
    min = reduction_op("min")
    max = reduction_op("max")
    prod = reduction_op("mul")

    def __matmul__(self, other: Array):
        b, m, n = self.shape[0], self.shape[1], other.shape[2]
        out = Array.empty(b * m * n, self.dtype, self.device).reshape((b, m, n))
        dispatch_bmm(self, other, out)
        return out


def reshape_strides(
    old_shape: tuple[int, ...], new_shape: tuple[int, ...], old_strides: tuple[int, ...]
) -> tuple[int, ...] | None:
    """If the reshape can be done with a zero-copy memory view, return the new strides.
    Otherwise, return None. Based on numpy's _attempt_nocopy_reshape in _core/src/multiarray/shape.c"""
    squeezed = [(x, y) for x, y in zip(old_shape, old_strides) if x != 1]
    old_shape = tuple(p[0] for p in squeezed)
    old_strides = tuple(p[1] for p in squeezed)
    oldnd = len(old_shape)
    newnd = len(new_shape)
    numel = product(old_shape)
    if numel != product(new_shape):
        raise ValueError("Shapes do not share the same number of elements")
    if numel == 0:
        return (0,) * newnd

    oi = 0
    oj = 1
    ni = 0
    nj = 1
    new_strides = [0] * newnd
    while ni < newnd and oi < oldnd:
        np = new_shape[ni]
        op = old_shape[oi]

        while np != op:
            if np < op:
                # Misses trailing 1s, these are handled later
                np *= new_shape[nj]
                nj += 1
            else:
                op *= old_shape[oj]
                oj += 1

        # Check whether the original axes can be combined
        for ok in range(oi, oj - 1):
            # C order
            if old_strides[ok] != old_shape[ok + 1] * old_strides[ok + 1]:
                # Not contiguous enough
                return None

        # Calculate new strides for all axes currently worked with
        new_strides[nj - 1] = old_strides[oj - 1]
        for nk in range(nj - 1, ni, -1):
            new_strides[nk - 1] = new_strides[nk] * new_shape[nk]
        ni = nj
        nj += 1
        oi = oj
        oj += 1

    # Set strides corresponding to trailing 1s of the new shape
    if ni >= 1:
        last_stride = new_strides[ni - 1]
    else:
        last_stride = 1
    for nk in range(ni, newnd):
        new_strides[nk] = last_stride

    return tuple(new_strides)
