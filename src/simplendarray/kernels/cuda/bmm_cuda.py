from typing import Annotated, Callable

from simplendarray.dtypes import all_float_dtypes, cname, ctype, i64, u32
from simplendarray.transpiler import ref
from simplendarray.transpiler.runtime import DType, SpecItem

from ._bmm_cuda_stubs import _BmmModuleClass
from .helpers import (
    __syncthreads,
    blockDim,
    blockIdx,
    cudaError_t,
    cudaGetErrorString,
    cudaGetLastError,
    cudaSuccess,
    dim3,
    printf,
    static_assert,
    threadIdx,
)

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
    BDX: Annotated[i64, "const"] = 16
    BDY: Annotated[i64, "const"] = 16
    BM: Annotated[int, "const"] = 128
    BN: Annotated[int, "const"] = 128
    BK: Annotated[int, "const"] = 16
    b_idx: i64 = blockDim.z * blockIdx.z + threadIdx.z  # Batch idx
    tM: i64 = threadIdx.y
    tN: i64 = threadIdx.x
    zero: Annotated[T, "const"] = 0.0  # type: ignore
    if b_idx >= B:
        return
    static_assert(BM % BDY == 0, "BM must be divisible by blockDim.y")
    static_assert(BN % BDX == 0, "BN must be divisible by blockDim.x")
    static_assert(BK % BDY == 0, "BK must be divisible by blockDim.y")
    static_assert(BK % BDX == 0, "BK must be divisible by blockDim.x")
    a_shared: Annotated[list[T], "__shared__", BM * BK] = []
    b_shared: Annotated[list[T], "__shared__", BK * BN] = []

    mpt: Annotated[i64, "const"] = BM // BDY  # M per thread
    npt: Annotated[i64, "const"] = BN // BDX  # N per thread
    a_reg: Annotated[list[T], mpt] = []
    b_reg: Annotated[list[T], npt] = []

    # a_ptr, b_ptr, c_ptr will be the top left of the tile assigned to this block
    # a_ptr and b_ptr will slide 32 right/down along the k dimension until the end
    a_ptr = ref(a_ptr[a_off + b_idx * a_stride_b + blockIdx.y * BM * a_stride_m])
    b_ptr = ref(b_ptr[b_off + b_idx * b_stride_b + blockIdx.x * BN * b_stride_n])
    c_ptr = ref(c_ptr[c_off + b_idx * c_stride_b + blockIdx.y * BM * c_stride_m + blockIdx.x * BN * c_stride_n])
    acc: Annotated[list[T], mpt * npt] = []
    for i in range(mpt * npt):
        acc[i] = zero

    for k_tile in range((K + BK - 1) // BK):
        # Load A and B tiles into smem, then sync
        for sub_m in range(mpt):
            for sub_k in range(BK // BDX):
                a_val: T = zero
                if (sub_m * BDY + tM) + blockIdx.y * BM < M and k_tile * BK + (sub_k * BDX + threadIdx.x) < K:
                    a_val = a_ptr[(sub_m * BDY + tM) * a_stride_m + (sub_k * BDX + threadIdx.x) * a_stride_k]

                a_shared[(sub_m * BDY + tM) * BK + (sub_k * BDX + threadIdx.x)] = a_val

        for sub_k in range(BK // BDY):
            for sub_n in range(npt):
                b_val: T = zero
                if (sub_n * BDX + tN) + blockIdx.x * BN < N and k_tile * BK + (sub_k * BDY + threadIdx.y) < K:
                    b_val = b_ptr[(sub_k * BDY + threadIdx.y) * b_stride_k + (sub_n * BDX + tN) * b_stride_n]
                b_shared[(sub_k * BDY + threadIdx.y) * BN + (sub_n * BDX + tN)] = b_val
        __syncthreads()

        # Tile matmul in registers. First move shared memory into registers, then FMA inside registers
        for k_dot in range(BK):
            for j in range(npt):
                b_reg[j] = b_shared[k_dot * BN + (j * BDX + tN)]
            for i in range(mpt):
                a_reg[i] = a_shared[(i * BDY + tM) * BK + k_dot]
                for j in range(npt):
                    acc[i * npt + j] += a_reg[i] * b_reg[j]

        # Advance a and b tiles to their next location along k dimension (go BS right/down). Then sync.
        a_ptr = ref(a_ptr[a_stride_k * BK])
        b_ptr = ref(b_ptr[b_stride_k * BK])
        __syncthreads()
    for m in range(mpt):
        for n in range(npt):
            if (m * BDY + tM) + blockIdx.y * BM < M and (n * BDX + tN) + blockIdx.x * BN < N:
                c_ptr[(m * BDY + tM) * c_stride_m + (n * BDX + tN) * c_stride_n] = (
                    alpha * acc[m * npt + n] + beta * c_ptr[(m * BDY + tM) * c_stride_m + (n * BDX + tN) * c_stride_n]
                )


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
    block: dim3 = [16, 16, 1]  # for n, m, b # type: ignore
    grid_x: u32 = (N + 8 * block.x - 1) // (8 * block.x)
    grid_y: u32 = (M + 8 * block.y - 1) // (8 * block.y)
    grid_z: u32 = (B + block.z - 1) // (block.z)
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

    err: cudaError_t = cudaGetLastError()
    if err != cudaSuccess:
        printf("Launch failed: %s\\n", cudaGetErrorString(err))
        exit(123)


bmm_module_cuda = bmm_module_cuda.compile("nvcc", ["-O3"])
