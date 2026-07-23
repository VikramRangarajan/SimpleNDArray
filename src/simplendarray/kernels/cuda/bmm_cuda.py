from typing import Annotated, Callable

from simplendarray.dtypes import all_float_dtypes, cname, ctype, i64, u32
from simplendarray.transpiler import ref
from simplendarray.transpiler.runtime import DType, SpecItem

from ._bmm_cuda_stubs import _BmmModuleClass
from .helpers import __syncthreads, blockDim, blockIdx, dim3, threadIdx

void = None

bmm_module_cuda = _BmmModuleClass(
    includes=["#include <math.h>"],
    stub_path=__file__,
    stub_var="bmm_module",
    module_name="bmm_module_cuda",
)


bmm_kernel_specs: list[SpecItem] = []
for dt in all_float_dtypes[:1]:
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
    b_idx: i64 = blockDim.z * blockIdx.z + threadIdx.z  # Batch idx
    m_idx: i64 = threadIdx.y + blockIdx.y * blockDim.y
    n_idx: i64 = threadIdx.x + blockIdx.x * blockDim.x
    tM: i64 = threadIdx.y
    tN: i64 = threadIdx.x
    zero: Annotated[T, "const"] = 0.0  # type: ignore
    if b_idx >= B:
        return
    BS: Annotated[int, "const"] = 32  # Same as blockDim.x,y
    a_shared: Annotated[list[T], "__shared__", BS * BS] = []
    b_shared: Annotated[list[T], "__shared__", BS * BS] = []

    # a_ptr, b_ptr, c_ptr to the start of the arrays at the batch index
    # a_tile, b_tile, c_tile will be the top left of the tile assigned to this block
    # a_tile and b_tile will slide 32 right/down along the k dimension until the end
    a_ptr = ref(a_ptr[a_off + b_idx * a_stride_b])
    b_ptr = ref(b_ptr[b_off + b_idx * b_stride_b])
    c_ptr = ref(c_ptr[c_off + b_idx * c_stride_b])
    a_tile: list[T] = ref(a_ptr[blockIdx.y * BS * a_stride_m])
    b_tile: list[T] = ref(b_ptr[blockIdx.x * BS * b_stride_n])
    c_tile: list[T] = ref(c_ptr[blockIdx.y * BS * c_stride_m + blockIdx.x * BS * c_stride_n])
    acc: T = zero

    for k_tile in range((K + BS - 1) // BS):
        # Load A and B tiles into smem, then sync
        a_val: T = (
            a_tile[tM * a_stride_m + threadIdx.x * a_stride_k] if m_idx < M and k_tile * BS + threadIdx.x < K else zero
        )
        b_val: T = (
            b_tile[threadIdx.y * b_stride_k + tN * b_stride_n] if n_idx < N and k_tile * BS + threadIdx.y < K else zero
        )
        a_shared[tM * BS + threadIdx.x] = a_val
        b_shared[threadIdx.y * BS + tN] = b_val
        __syncthreads()

        # Perform tile matmul in shared memory
        for k_dot in range(BS):
            acc += a_shared[tM * BS + k_dot] * b_shared[k_dot * BS + tN]

        # Advance a and b tiles to their next location along k dimension (go BS right/down). Then sync.
        a_tile = ref(a_tile[a_stride_k * BS])
        b_tile = ref(b_tile[b_stride_k * BS])
        __syncthreads()
    if m_idx < M and n_idx < N:
        c_tile[tM * c_stride_m + tN] = alpha * acc + beta * c_tile[tM * c_stride_m + tN * c_stride_n]


numerical_unary_specs: list[SpecItem] = []
for dt in all_float_dtypes[:1]:
    numerical_unary_specs.append(SpecItem({"T": ctype(dt), "Op": f"bmm_kernel_{cname(dt)}"}, f"bmm_{cname(dt)}"))


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
    block: dim3 = [32, 32, 1]  # for n, m, b # type: ignore
    grid_x: u32 = (N + block.x - 1) // block.x
    grid_y: u32 = (M + block.y - 1) // block.y
    grid_z: u32 = (B + block.z - 1) // block.z
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


bmm_module_cuda = bmm_module_cuda.compile("nvcc", ["-O3"])
