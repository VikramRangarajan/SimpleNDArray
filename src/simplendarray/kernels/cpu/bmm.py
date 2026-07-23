from simplendarray.dtypes import all_float_dtypes, cname, ctype, i64
from simplendarray.transpiler.runtime import DType, SpecItem

from ._bmm_stubs import _BmmModuleClass

void = None

bmm_module = _BmmModuleClass(
    includes=["#include <math.h>"],
    stub_path=__file__,
    stub_var="bmm_module",
    module_name="bmm_module",
)

numerical_unary_specs: list[SpecItem] = []
for dt in all_float_dtypes:
    numerical_unary_specs.append(SpecItem({"T": ctype(dt)}, f"bmm_{cname(dt)}"))

"""
bmm does C = alpha * (A @ B) + beta * C, like cublasgemmBatched
A: b x m x k
B: b x k x n
C: b x m x n
3 array pointers, 4 shape params, 9 strides, 3 offsets, 2 alpha/beta params
"""


@bmm_module.compile_fn(numerical_unary_specs, pybind=True)
def bmm[T: DType](
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
    for b in range(B):
        for i in range(M):
            for j in range(N):
                acc: T = 0.0  # type: ignore
                for k in range(K):
                    a_val: T = a_ptr[a_off + b * a_stride_b + i * a_stride_m + k * a_stride_k]
                    b_val: T = b_ptr[b_off + b * b_stride_b + j * b_stride_n + k * b_stride_k]
                    acc += a_val * b_val
                c_orig: T = c_ptr[c_off + b * c_stride_b + i * c_stride_m + j * c_stride_n]
                c_ptr[c_off + b * c_stride_b + i * c_stride_m + j * c_stride_n] = alpha * acc + beta * c_orig


bmm_module = bmm_module.compile()
