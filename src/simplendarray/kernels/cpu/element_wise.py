from typing import Callable

from simplendarray.dtypes import all_dtypes, cname, ctype, i64
from simplendarray.transpiler.runtime import DType, SpecItem

from ._element_wise_stubs import _ElementWiseModuleClass as _ElementWiseModule

void = None

element_wise_module = _ElementWiseModule(
    includes=["#include <math.h>"],
    stub_path=__file__,
    stub_var="element_wise_module",
    module_name="element_wise_module",
)


@element_wise_module.compile_fn(types=[SpecItem({"T": ctype(dt)}, f"_add_{cname(dt)}") for dt in all_dtypes])
def _add[T: DType](x: T, y: T) -> T:
    return x + y


@element_wise_module.compile_fn(types=[SpecItem({"T": ctype(dt)}, f"_sub_{cname(dt)}") for dt in all_dtypes])
def _sub[T: DType](x: T, y: T) -> T:
    return x - y


@element_wise_module.compile_fn(types=[SpecItem({"T": ctype(dt)}, f"_mul_{cname(dt)}") for dt in all_dtypes])
def _mul[T: DType](x: T, y: T) -> T:
    return x * y


@element_wise_module.compile_fn(types=[SpecItem({"T": ctype(dt)}, f"_div_{cname(dt)}") for dt in all_dtypes])
def _div[T: DType](x: T, y: T) -> T:
    return x / y


def atan2[T: DType](x: T, y: T) -> T: ...


@element_wise_module.compile_fn(types=[SpecItem({"T": ctype(dt)}, f"_atan2_{cname(dt)}") for dt in all_dtypes])
def _atan2[T: DType](y: T, x: T) -> T:
    return atan2(y, x)


numerical_binary_specs: list[SpecItem] = []
for dt in all_dtypes:
    for op in ["add", "sub", "mul", "div", "atan2"]:
        numerical_binary_specs.append(
            SpecItem({"T": ctype(dt), "Op": f"_{op}_{cname(dt)}"}, f"element_wise_binary_{cname(dt)}__{op}")
        )


@element_wise_module.compile_fn(numerical_binary_specs, pybind=True)
def element_wise_binary[T: DType, Op: Callable](
    a: list[T],
    a_off: i64,
    a_stride: i64,
    b: list[T],
    b_off: i64,
    b_stride: i64,
    c: list[T],
    c_off: i64,
    c_stride: i64,
    n: i64,
) -> void:
    for i in range(n):
        c[c_off + i * c_stride] = Op(a[a_off + i * a_stride], b[b_off + i * b_stride])  # pyrefly: ignore [not-callable]


def _math_id[T](x: T) -> T: ...


exp = _math_id
exp2 = _math_id
log = _math_id
log2 = _math_id
log10 = _math_id
sqrt = _math_id
sin = _math_id
cos = _math_id
tan = _math_id
asin = _math_id
acos = _math_id
atan = _math_id
sinh = _math_id
cosh = _math_id
tanh = _math_id


@element_wise_module.compile_fn(types=[SpecItem({"T": ctype(dt)}, f"_exp_{cname(dt)}") for dt in all_dtypes])
def _exp[T: DType](x: T) -> T:
    return exp(x)


@element_wise_module.compile_fn(types=[SpecItem({"T": ctype(dt)}, f"_exp2_{cname(dt)}") for dt in all_dtypes])
def _exp2[T: DType](x: T) -> T:
    return exp2(x)


@element_wise_module.compile_fn(types=[SpecItem({"T": ctype(dt)}, f"_log_{cname(dt)}") for dt in all_dtypes])
def _log[T: DType](x: T) -> T:
    return log(x)


@element_wise_module.compile_fn(types=[SpecItem({"T": ctype(dt)}, f"_log2_{cname(dt)}") for dt in all_dtypes])
def _log2[T: DType](x: T) -> T:
    return log2(x)


@element_wise_module.compile_fn(types=[SpecItem({"T": ctype(dt)}, f"_log10_{cname(dt)}") for dt in all_dtypes])
def _log10[T: DType](x: T) -> T:
    return log10(x)


@element_wise_module.compile_fn(types=[SpecItem({"T": ctype(dt)}, f"_relu_{cname(dt)}") for dt in all_dtypes])
def _relu[T: DType](x: T) -> T:
    return x if x > 0 else 0  # pyrefly: ignore [bad-return,unsupported-operation]


@element_wise_module.compile_fn(types=[SpecItem({"T": ctype(dt)}, f"_square_{cname(dt)}") for dt in all_dtypes])
def _square[T: DType](x: T) -> T:
    return x * x


@element_wise_module.compile_fn(types=[SpecItem({"T": ctype(dt)}, f"_sqrt_{cname(dt)}") for dt in all_dtypes])
def _sqrt[T: DType](x: T) -> T:
    return sqrt(x)


@element_wise_module.compile_fn(types=[SpecItem({"T": ctype(dt)}, f"_sin_{cname(dt)}") for dt in all_dtypes])
def _sin[T: DType](x: T) -> T:
    return sin(x)


@element_wise_module.compile_fn(types=[SpecItem({"T": ctype(dt)}, f"_cos_{cname(dt)}") for dt in all_dtypes])
def _cos[T: DType](x: T) -> T:
    return cos(x)


@element_wise_module.compile_fn(types=[SpecItem({"T": ctype(dt)}, f"_tan_{cname(dt)}") for dt in all_dtypes])
def _tan[T: DType](x: T) -> T:
    return tan(x)


@element_wise_module.compile_fn(types=[SpecItem({"T": ctype(dt)}, f"_asin_{cname(dt)}") for dt in all_dtypes])
def _asin[T: DType](x: T) -> T:
    return asin(x)


@element_wise_module.compile_fn(types=[SpecItem({"T": ctype(dt)}, f"_acos_{cname(dt)}") for dt in all_dtypes])
def _acos[T: DType](x: T) -> T:
    return acos(x)


@element_wise_module.compile_fn(types=[SpecItem({"T": ctype(dt)}, f"_atan_{cname(dt)}") for dt in all_dtypes])
def _atan[T: DType](x: T) -> T:
    return atan(x)


@element_wise_module.compile_fn(types=[SpecItem({"T": ctype(dt)}, f"_sinh_{cname(dt)}") for dt in all_dtypes])
def _sinh[T: DType](x: T) -> T:
    return sinh(x)


@element_wise_module.compile_fn(types=[SpecItem({"T": ctype(dt)}, f"_cosh_{cname(dt)}") for dt in all_dtypes])
def _cosh[T: DType](x: T) -> T:
    return cosh(x)


@element_wise_module.compile_fn(types=[SpecItem({"T": ctype(dt)}, f"_tanh_{cname(dt)}") for dt in all_dtypes])
def _tanh[T: DType](x: T) -> T:
    return tanh(x)


numerical_unary_specs: list[SpecItem] = []
for dt in all_dtypes:
    for op in [
        "exp",
        "exp2",
        "log",
        "log2",
        "log10",
        "relu",
        "square",
        "sqrt",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "sinh",
        "cosh",
        "tanh",
    ]:
        numerical_unary_specs.append(
            SpecItem({"T": ctype(dt), "Op": f"_{op}_{cname(dt)}"}, f"element_wise_unary_{cname(dt)}__{op}")
        )


@element_wise_module.compile_fn(numerical_unary_specs, pybind=True)
def element_wise_unary[T: DType, Op: Callable](
    a: list[T],
    a_off: i64,
    a_stride: i64,
    c: list[T],
    c_off: i64,
    c_stride: i64,
    n: i64,
) -> void:
    for i in range(n):
        c[c_off + i * c_stride] = Op(a[a_off + i * a_stride])  # pyrefly: ignore [not-callable]


@element_wise_module.compile_fn(
    types=[SpecItem({"T": ctype(dt)}, f"arange_{cname(dt)}") for dt in all_dtypes], pybind=True
)
def arange[T: DType](out: list[T], out_offset: i64, out_stride: i64, numel: i64) -> void:
    for i in range(numel):
        out[out_offset + i * out_stride] = i  # pyrefly: ignore [unsupported-operation]


@element_wise_module.compile_fn(
    types=[SpecItem({"T": ctype(dt)}, f"reshape_copy_{cname(dt)}") for dt in all_dtypes], pybind=True
)
def reshape_copy[T: DType](
    inp: list[T],
    inp_strides: list[i64],
    inp_shape: list[i64],
    inp_index: list[i64],
    inp_offset: i64,
    inp_ndim: i64,
    out: list[T],
    out_strides: list[i64],
    out_shape: list[i64],
    out_index: list[i64],
    out_offset: i64,
    out_ndim: i64,
    numel: i64,
) -> void:
    inp_idx: i64 = inp_offset
    out_idx: i64 = out_offset
    for i in range(inp_ndim):
        inp_index[i] = 0
    for i in range(out_ndim):
        out_index[i] = 0
    for i in range(numel):
        out[out_idx] = inp[inp_idx]

        # Now we increment ND index of input
        inp_index[inp_ndim - 1] += 1
        inp_idx += inp_strides[inp_ndim - 1]
        for j in range(inp_ndim - 1, -1, -1):
            if inp_index[j] == inp_shape[j]:
                inp_index[j] = 0
                inp_idx -= inp_strides[j] * inp_shape[j]
                if j > 0:
                    inp_index[j - 1] += 1
                    inp_idx += inp_strides[j - 1]

        # Increment ND index of output
        out_index[out_ndim - 1] += 1
        out_idx += out_strides[out_ndim - 1]
        for j in range(out_ndim - 1, -1, -1):
            if out_index[j] == out_shape[j]:
                out_index[j] = 0
                out_idx -= out_strides[j] * out_shape[j]
                if j > 0:
                    out_index[j - 1] += 1
                    out_idx += out_strides[j - 1]


element_wise_module = element_wise_module.compile()
