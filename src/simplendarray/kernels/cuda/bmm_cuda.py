from math import sqrt
from typing import Callable

from simplendarray.dtypes import all_float_dtypes, cname, ctype, i64, u32
from simplendarray.transpiler.runtime import DType, SpecItem

from ._bmm_cuda_stubs import _BmmModuleClass
from .helpers import blockDim, blockIdx, dim3, threadIdx

void = None

bmm_module_cuda = _BmmModuleClass(
    includes=["#include <math.h>"],
    stub_path=__file__,
    stub_var="bmm_module",
)


bmm_kernel_specs: list[SpecItem] = []
for dt in all_float_dtypes:
    bmm_kernel_specs.append(SpecItem({"T": ctype(dt)}, f"bmm_kernel_{cname(dt)}"))


@bmm_module_cuda.compile_fn(bmm_kernel_specs, c_attrs=["__global__"])
def bmm_kernel[T: DType](
    a_ptr: list[T],
    a_off: i64,
    a_stride_b: i64,
    a_stride_m: i64,
    a_stride_k: i64,
    b_ptr: list[T],
    b_off: i64,
    b_stride_b: i64,
    b_stride_k: i64,
    b_stride_n: i64,
    c_ptr: list[T],
    c_off: i64,
    c_stride_b: i64,
    c_stride_m: i64,
    c_stride_n: i64,
    B: i64,
    M: i64,
    K: i64,
    N: i64,
    alpha: T,
    beta: T,
) -> void:
    b_idx: i64 = blockDim.x * blockIdx.x + threadIdx.x  # Batch idx
    m_idx: i64 = blockDim.y * blockIdx.y + threadIdx.y
    n_idx: i64 = blockDim.z * blockIdx.z + threadIdx.z
    if b_idx < B and m_idx < M and n_idx < N:
        acc: T = 0.0  # type: ignore
        for k in range(K):
            a_val: T = a_ptr[a_off + b_idx * a_stride_b + m_idx * a_stride_m + k * a_stride_k]
            b_val: T = b_ptr[b_off + b_idx * b_stride_b + k * b_stride_k + n_idx * b_stride_n]
            acc += a_val * b_val
        c_val: T = c_ptr[c_off + b_idx * c_stride_b + m_idx * c_stride_m + n_idx * c_stride_n]
        c_ptr[c_off + b_idx * c_stride_b + m_idx * c_stride_m + n_idx * c_stride_n] = alpha * acc + beta * c_val


numerical_unary_specs: list[SpecItem] = []
for dt in all_float_dtypes:
    numerical_unary_specs.append(SpecItem({"T": ctype(dt), "Op": f"bmm_kernel_{cname(dt)}"}, f"bmm_{cname(dt)}"))


@bmm_module_cuda.compile_fn()
def ceil_sqrt(i: i64) -> i64:
    r: int = sqrt(i)  # type: ignore
    while 1 * r * r < i:
        r += 1
    return r


"""
Fixed block dim 1024
Fixed number of kernel calls 2
Elements per thread = T
Elements per block = T * 1024
Number of blocks = ceildiv(d, 1024 * T)
For length d, we want num blocks = ceildiv(d, 1024 * T) <= 1024 * T => (1024^2) T^2 = d, T = ceil(sqrt(d/1024^2))
"""


@bmm_module_cuda.compile_fn(numerical_unary_specs, pybind=True)
def bmm[T: DType, Op: Callable](
    a_ptr: list[T],
    a_off: i64,
    a_stride_b: i64,
    a_stride_m: i64,
    a_stride_k: i64,
    b_ptr: list[T],
    b_off: i64,
    b_stride_b: i64,
    b_stride_k: i64,
    b_stride_n: i64,
    c_ptr: list[T],
    c_off: i64,
    c_stride_b: i64,
    c_stride_m: i64,
    c_stride_n: i64,
    B: i64,
    M: i64,
    K: i64,
    N: i64,
    alpha: T,
    beta: T,
) -> void:
    block: dim3 = [8, 16, 8]  # for b, m, n # type: ignore
    grid_x: u32 = (B + block.x - 1) // block.x
    grid_y: u32 = (M + block.y - 1) // block.y
    grid_z: u32 = (M + block.z - 1) // block.z
    grid: dim3 = [grid_x, grid_y, grid_z]  # type: ignore

    Op[[[grid, block]]](  # type: ignore
        a_ptr,
        a_off,
        a_stride_b,
        a_stride_m,
        a_stride_k,
        b_ptr,
        b_off,
        b_stride_b,
        b_stride_k,
        b_stride_n,
        c_ptr,
        c_off,
        c_stride_b,
        c_stride_m,
        c_stride_n,
        B,
        M,
        K,
        N,
        alpha,
        beta,
    )


bmm_module_cuda = bmm_module_cuda.compile("nvcc")
