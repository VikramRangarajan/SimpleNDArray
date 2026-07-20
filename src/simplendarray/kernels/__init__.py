from typing import TYPE_CHECKING

from simplendarray.dtypes import cname, ctype, get_dtype
from simplendarray.utils import all_eq, product

from .cpu import element_wise_module, reduction_module
from .cuda import element_wise_module_cuda

if TYPE_CHECKING:
    from simplendarray import Array, Buffer, BufferCuda

__all__ = ["element_wise_module", "element_wise_module_cuda"]

elem_wise_modules = {
    "gpu": element_wise_module_cuda,
    "cpu": element_wise_module,
}

reduction_modules = {"cpu": reduction_module}


def dispatch_element_wise_unary(array: Array, out: Array, op: str):

    dt = get_dtype(array.data.typecode)

    dispatch_key = (("T", ctype(dt)), ("Op", f"_{op}_{cname(dt)}"))
    array = array.reshape(-1)
    out = out.reshape(-1)

    a = array.data.address
    a_off = array.offset
    a_stride = array.strides[0]
    c = out.data.address
    c_off = out.offset
    c_stride = out.strides[0]
    n = array.size
    elem_wise_modules[array.device].DISPATCH_DICT_element_wise_unary[dispatch_key](
        a, a_off, a_stride, c, c_off, c_stride, n
    )


def dispatch_element_wise_binary(a_arr: Array, b_arr: Array, c_arr: Array, op: str):
    if (
        not all_eq(a_arr.size, b_arr.size, c_arr.size)
        or not all_eq(a_arr.data.typecode, b_arr.data.typecode, c_arr.data.typecode)
        or not all_eq(a_arr.device, b_arr.device, c_arr.device)
    ):
        raise ValueError("All arrays must be the same size, dtype, and device")
    dt = get_dtype(a_arr.data.typecode)

    dispatch_key = (("T", ctype(dt)), ("Op", f"_{op}_{cname(dt)}"))
    a_arr = a_arr.reshape(-1)
    b_arr = b_arr.reshape(-1)
    c_arr = c_arr.reshape(-1)
    if not c_arr.is_contiguous:
        raise ValueError("Output not contiguous")

    a = a_arr.data.address
    a_off = a_arr.offset
    a_stride = a_arr.strides[0]
    b = b_arr.data.address
    b_off = b_arr.offset
    b_stride = b_arr.strides[0]
    c = c_arr.data.address
    c_off = c_arr.offset
    c_stride = c_arr.strides[0]
    n = a_arr.size

    elem_wise_modules[a_arr.device].DISPATCH_DICT_element_wise_binary[dispatch_key](
        a, a_off, a_stride, b, b_off, b_stride, c, c_off, c_stride, n
    )


def dispatch_reshape_copy(
    a_arr: Array,
    new_shape: tuple[int, ...],
    new_buffer: Buffer | BufferCuda,
    inp_shape_buffer: Buffer | BufferCuda,
    inp_strides_buffer: Buffer | BufferCuda,
    inp_work_buffer: Buffer | BufferCuda,
    out_shape_buffer: Buffer | BufferCuda,
    out_strides_buffer: Buffer | BufferCuda,
    out_work_buffer: Buffer | BufferCuda,
):
    dt = get_dtype(a_arr.data.typecode)
    numel = a_arr.size

    out_ndim = len(new_shape)

    if a_arr.device == "gpu":
        ker_name = f"reshape_copy_kernel_{cname(dt)}"
        dispatch_key = (("T", ctype(dt)), ("Kernel", ker_name))
    else:
        dispatch_key = (("T", ctype(dt)),)

    elem_wise_modules[a_arr.device].DISPATCH_DICT_reshape_copy[dispatch_key](
        a_arr.data.address,
        inp_strides_buffer.address,
        inp_shape_buffer.address,
        inp_work_buffer.address,
        a_arr.offset,
        a_arr.ndim,
        new_buffer.address,
        out_strides_buffer.address,
        out_shape_buffer.address,
        out_work_buffer.address,
        0,
        out_ndim,
        numel,
    )


def dispatch_arange(buf: Buffer | BufferCuda, offset: int, stride: int, n: int):
    dt = get_dtype(buf.typecode)

    if buf.device == "gpu":
        dispatch_key = (("T", ctype(dt)), ("Kernel", f"arange_kernel_{cname(dt)}"))
    else:
        dispatch_key = (("T", ctype(dt)),)

    a = buf.address
    elem_wise_modules[buf.device].DISPATCH_DICT_arange[dispatch_key](a, offset, stride, n)


def dispatch_reduction(a_arr: Array, b_arr: Array, op: str, dims: tuple[int]):
    if a_arr.data.typecode != b_arr.data.typecode or a_arr.device != b_arr.device:
        raise ValueError("All arrays must be the same size, dtype, and device")
    dt = get_dtype(a_arr.data.typecode)

    dispatch_key = (("T", ctype(dt)), ("Op", f"_{op}_{cname(dt)}"))

    if a_arr.ndim < 2:
        raise ValueError("Unsqueeze not implemented yet")

    # if a_arr.ndim != 2:
    not_reduce_dims = tuple(set(range(a_arr.ndim)) - set(dims))
    print(not_reduce_dims, dims)
    a_arr = a_arr.transpose(not_reduce_dims + dims)
    a_arr = a_arr.reshape((product(a_arr.shape[i] for i in range(len(not_reduce_dims))), -1))

    print(b_arr.shape, a_arr.shape)
    if b_arr.shape != (a_arr.shape[0],):
        raise ValueError("Output array is invalid shape")
    a = a_arr.data.address
    a_off = a_arr.offset
    a_row_stride, a_col_stride = a_arr.strides
    b = b_arr.data.address
    b_off = b_arr.offset
    b_stride = b_arr.strides[0]
    n, d = a_arr.shape

    reduction_modules[a_arr.device].DISPATCH_DICT_reduction[dispatch_key](
        a, a_off, a_row_stride, a_col_stride, b, b_off, b_stride, n, d
    )
