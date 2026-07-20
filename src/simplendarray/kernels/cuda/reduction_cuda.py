from typing import Annotated, Callable

from simplendarray.dtypes import all_dtypes, cname, ctype, i64, u32
from simplendarray.transpiler.runtime import DType, SpecItem

from ._reduction_cuda_stubs import _ReductionModuleClass
from .helpers import __syncthreads, blockDim, blockIdx, dim3, threadIdx

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
    for op in ["add", "max", "min"]:  # TODO: mul
        default_value = {"add": "0", "max": "-INFINITY", "min": "INFINITY"}[op]
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
    for op in ["add", "max", "min"]:  # TODO: mul
        numerical_unary_specs.append(
            SpecItem({"T": ctype(dt), "Op": f"reduction_kernel_{op}_{cname(dt)}"}, f"reduction_{op}_{cname(dt)}")
        )


@reduction_module_cuda.compile_fn(numerical_unary_specs, pybind=True)
def reduction[T: DType, Op: Callable](
    a: list[T], a_off: i64, a_row_stride: i64, a_col_stride: i64, c: list[T], c_off: i64, c_stride: i64, n: i64, d: i64
) -> void:
    block_dim_x: u32 = (d + 1024 - 1) // 1024
    block_dim_y: u32 = n
    grid: dim3 = [block_dim_x, block_dim_y]  # type: ignore
    Op[[[grid, 1024]]](a, a_off, a_row_stride, a_col_stride, c, c_off, c_stride, d, 16)  # type: ignore[unsupported-operation]


reduction_module = reduction_module_cuda.compile("nvcc")
