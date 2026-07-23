from typing import Callable

from simplendarray.dtypes import all_dtypes, cname, ctype, i64
from simplendarray.transpiler.runtime import DType, SpecItem

from ._reduction_stubs import _ReductionModuleClass

void = None

reduction_module = _ReductionModuleClass(
    includes=["#include <math.h>"],
    stub_path=__file__,
    stub_var="reduction_module",
    module_name="reduction_module",
)


def _math_id[T](x: T) -> T: ...


@reduction_module.compile_fn(types=[SpecItem({"T": ctype(dt)}, f"_add_{cname(dt)}") for dt in all_dtypes])
def _add[T: DType](x: T, y: T) -> T:
    return x + y


@reduction_module.compile_fn(types=[SpecItem({"T": ctype(dt)}, f"_mul_{cname(dt)}") for dt in all_dtypes])
def _mul[T: DType](x: T, y: T) -> T:
    return x * y


@reduction_module.compile_fn(types=[SpecItem({"T": ctype(dt)}, f"_max_{cname(dt)}") for dt in all_dtypes])
def _max[T: DType](x: T, y: T) -> T:
    return x if x > y else y


@reduction_module.compile_fn(types=[SpecItem({"T": ctype(dt)}, f"_min_{cname(dt)}") for dt in all_dtypes])
def _min[T: DType](x: T, y: T) -> T:
    return x if x < y else y


numerical_unary_specs: list[SpecItem] = []
for dt in all_dtypes:
    for op in ["add", "mul", "max", "min"]:
        numerical_unary_specs.append(
            SpecItem({"T": ctype(dt), "Op": f"_{op}_{cname(dt)}"}, f"reduction_{op}_{cname(dt)}")
        )


@reduction_module.compile_fn(numerical_unary_specs, pybind=True)
def reduction[T: DType, Op: Callable](
    a: list[T], a_off: i64, a_row_stride: i64, a_col_stride: i64, c: list[T], c_off: i64, c_stride: i64, n: i64, d: i64
) -> void:
    for i in range(n):
        red: T = a[a_off + a_row_stride * i]
        for j in range(1, d):
            red = Op(red, a[a_off + a_row_stride * i + a_col_stride * j])  # pyrefly: ignore [not-callable]
        c[c_off + c_stride * i] = red


reduction_module = reduction_module.compile()
