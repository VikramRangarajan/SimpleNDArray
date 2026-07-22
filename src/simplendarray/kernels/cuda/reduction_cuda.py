from math import sqrt
from typing import Annotated, Callable

from simplendarray.dtypes import all_dtypes, cname, ctype, i64, u32
from simplendarray.transpiler.runtime import DType, SpecItem
from simplendarray.transpiler.transpiler import ref, sizeof

from ._reduction_cuda_stubs import _ReductionModuleClass
from .helpers import __syncthreads, blockDim, blockIdx, cudaFree, cudaMalloc, dim3, threadIdx

void = None

reduction_module_cuda = _ReductionModuleClass(
    includes=["#include <math.h>"],
    stub_path=__file__,
    stub_var="reduction_module",
)


def _math_id[T](x: T) -> T: ...


@reduction_module_cuda.compile_fn(
    types=[SpecItem({"T": ctype(dt)}, f"_add_{cname(dt)}") for dt in all_dtypes], c_attrs=["__device__"]
)
def _add[T: DType](x: T, y: T) -> T:
    return x + y


@reduction_module_cuda.compile_fn(
    types=[SpecItem({"T": ctype(dt)}, f"_mul_{cname(dt)}") for dt in all_dtypes], c_attrs=["__device__"]
)
def _mul[T: DType](x: T, y: T) -> T:
    return x * y


@reduction_module_cuda.compile_fn(
    types=[SpecItem({"T": ctype(dt)}, f"_max_{cname(dt)}") for dt in all_dtypes], c_attrs=["__device__"]
)
def _max[T: DType](x: T, y: T) -> T:
    return x if x > y else y


@reduction_module_cuda.compile_fn(
    types=[SpecItem({"T": ctype(dt)}, f"_min_{cname(dt)}") for dt in all_dtypes], c_attrs=["__device__"]
)
def _min[T: DType](x: T, y: T) -> T:
    return x if x < y else y


reduction_kernel_specs: list[SpecItem] = []
for dt in all_dtypes:
    for op in ["add", "max", "min", "mul"]:
        default_value = {"add": "0", "max": "-INFINITY", "min": "INFINITY", "mul": "1"}[op]
        reduction_kernel_specs.append(
            SpecItem(
                {"T": ctype(dt), "Op": f"_{op}_{cname(dt)}", "T_DEFAULT_VALUE": default_value},
                f"reduction_kernel_{op}_{cname(dt)}",
            )
        )


@reduction_module_cuda.compile_fn(reduction_kernel_specs, c_attrs=["__global__"])
def reduction_kernel[T: DType, Op: Callable, T_DEFAULT_VALUE: DType](
    a: list[T],
    a_off: i64,
    a_row_stride: i64,
    a_col_stride: i64,
    c: list[T],
    c_off: i64,
    c_stride: i64,
    d: i64,
    COARSE_FACTOR: i64,
) -> void:
    elems: Annotated[list[T], "__shared__", 1024] = []
    i: i64 = threadIdx.x + blockIdx.x * blockDim.x * COARSE_FACTOR
    row: i64 = blockIdx.y
    local_reduction: T = T_DEFAULT_VALUE  # type: ignore
    # Each thread will process COARSE_FACTOR elements sequentially, and store result in shared memory
    for tile in range(COARSE_FACTOR):
        col_idx: i64 = i + tile * blockDim.x
        if col_idx < d:
            local_reduction = Op(local_reduction, a[a_off + row * a_row_stride + a_col_stride * col_idx])  # type: ignore
    elems[threadIdx.x] = local_reduction
    __syncthreads()
    # Now parallel tree reduction will be done in shared memory, elems[0] will contain sum for block
    stride: i64 = blockDim.x // 2
    while stride > 0:
        if threadIdx.x < stride:
            elems[threadIdx.x] = Op(elems[threadIdx.x], elems[threadIdx.x + stride])  # type: ignore
        __syncthreads()
        stride = stride // 2

    # Block output saved to gmem
    if threadIdx.x == 0:
        c[c_off + row * c_stride + blockIdx.x] = elems[0]


numerical_unary_specs: list[SpecItem] = []
for dt in all_dtypes:
    for op in ["add", "max", "min", "mul"]:
        numerical_unary_specs.append(
            SpecItem({"T": ctype(dt), "Op": f"reduction_kernel_{op}_{cname(dt)}"}, f"reduction_{op}_{cname(dt)}")
        )


@reduction_module_cuda.compile_fn()
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


@reduction_module_cuda.compile_fn(numerical_unary_specs, pybind=True)
def reduction[T: DType, Op: Callable](
    a: list[T], a_off: i64, a_row_stride: i64, a_col_stride: i64, c: list[T], c_off: i64, c_stride: i64, n: i64, d: i64
) -> void:
    coarse_factor: i64 = ceil_sqrt(d // (1024 * 1024))
    if coarse_factor < 1:
        coarse_factor = 1
    # Really this coarse factor can be any number >= 1. course_factor_2 makes up for any extra blocks > 1024.
    block_dim_x: u32 = (d + (coarse_factor * 1024) - 1) // (coarse_factor * 1024)
    block_dim_y: u32 = n
    grid: dim3 = [block_dim_x, block_dim_y]  # type: ignore
    work_arr: list[T] = []
    cudaMalloc(ref(work_arr), block_dim_x * block_dim_y * sizeof(T))
    Op[[[grid, 1024]]](a, a_off, a_row_stride, a_col_stride, work_arr, 0, block_dim_x, d, coarse_factor)  # type: ignore[unsupported-operation]
    grid = [1, block_dim_y]  # type: ignore
    coarse_factor_2: i64 = (block_dim_x + 1024 - 1) // 1024
    Op[[[grid, 1024]]](work_arr, 0, block_dim_x, 1, c, c_off, c_stride, block_dim_x, coarse_factor_2)  # type: ignore[unsupported-operation]
    cudaFree(work_arr)


reduction_module = reduction_module_cuda.compile("nvcc")
