from typing import Annotated, Callable

from simplendarray.dtypes import all_dtypes, all_float_dtypes, cname, ctype, i64, u64
from simplendarray.transpiler.runtime import DType, SpecItem

from ._element_wise_cuda_stubs import _ElementWiseModuleClass
from .helpers import __syncthreads, blockDim, blockIdx, printf, threadIdx

void = None

element_wise_module_cuda = _ElementWiseModuleClass(
    includes=["#include <math.h>", "#include <cuda_runtime.h>"],
    stub_path=__file__,
    stub_var="element_wise_module",
)


@element_wise_module_cuda.compile_fn(
    types=[SpecItem({"T": ctype(dt)}, f"_prim_add_{cname(dt)}") for dt in all_dtypes], c_attrs=["__device__"]
)
def _add[T: DType](x: T, y: T) -> T:
    return x + y


@element_wise_module_cuda.compile_fn(
    types=[SpecItem({"T": ctype(dt)}, f"_prim_sub_{cname(dt)}") for dt in all_dtypes], c_attrs=["__device__"]
)
def _sub[T: DType](x: T, y: T) -> T:
    return x - y


@element_wise_module_cuda.compile_fn(
    types=[SpecItem({"T": ctype(dt)}, f"_prim_mul_{cname(dt)}") for dt in all_dtypes], c_attrs=["__device__"]
)
def _mul[T: DType](x: T, y: T) -> T:
    return x * y


@element_wise_module_cuda.compile_fn(
    types=[SpecItem({"T": ctype(dt)}, f"_prim_div_{cname(dt)}") for dt in all_dtypes], c_attrs=["__device__"]
)
def _div[T: DType](x: T, y: T) -> T:
    return x / y


def atan2[T: DType](x: T, y: T) -> T: ...


@element_wise_module_cuda.compile_fn(
    types=[SpecItem({"T": ctype(dt)}, f"_prim_atan2_{cname(dt)}") for dt in all_float_dtypes], c_attrs=["__device__"]
)
def _atan2[T: DType](y: T, x: T) -> T:
    return atan2(y, x)


numerical_binary_kernel_specs: list[SpecItem] = []
for dt in all_dtypes:
    for op in ["add", "sub", "mul", "div", "atan2"]:
        if op == "atan2" and ctype(dt) not in ("float", "double"):
            continue
        numerical_binary_kernel_specs.append(
            SpecItem({"T": ctype(dt), "Op": f"_prim_{op}_{cname(dt)}"}, f"_{op}_{cname(dt)}")
        )


@element_wise_module_cuda.compile_fn(numerical_binary_kernel_specs, c_attrs=["__global__"])
def element_wise_binary_kernel[T: DType, Op: Callable](
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
    i: u64 = threadIdx.x + blockIdx.x * blockDim.x
    if i < n:
        c[c_off + i * c_stride] = Op(a[a_off + i * a_stride], b[b_off + i * b_stride])  # pyrefly: ignore [not-callable]


numerical_binary_specs: list[SpecItem] = []
for dt in all_dtypes:
    for op in ["add", "sub", "mul", "div", "atan2"]:
        if op == "atan2" and ctype(dt) not in ("float", "double"):
            continue
        numerical_binary_specs.append(
            SpecItem(
                {"T": ctype(dt), "Op": f"_{op}_{cname(dt)}"},
                f"element_wise_binary_{cname(dt)}__{op}",
            )
        )


@element_wise_module_cuda.compile_fn(numerical_binary_specs, pybind=True)
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
) -> None:
    NUM_THREADS: int = 1024
    NUM_BLOCKS: int = (n + NUM_THREADS - 1) // NUM_THREADS
    Op[[[NUM_BLOCKS, NUM_THREADS]]](a, a_off, a_stride, b, b_off, b_stride, c, c_off, c_stride, n)  # pyrefly: ignore[unsupported-operation]


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


@element_wise_module_cuda.compile_fn(
    types=[SpecItem({"T": ctype(dt)}, f"_primitive_exp_{cname(dt)}") for dt in all_float_dtypes], c_attrs=["__device__"]
)
def _exp[T: DType](x: T) -> T:
    return exp(x)


@element_wise_module_cuda.compile_fn(
    types=[SpecItem({"T": ctype(dt)}, f"_primitive_exp2_{cname(dt)}") for dt in all_float_dtypes],
    c_attrs=["__device__"],
)
def _exp2[T: DType](x: T) -> T:
    return exp2(x)


@element_wise_module_cuda.compile_fn(
    types=[SpecItem({"T": ctype(dt)}, f"_primitive_log_{cname(dt)}") for dt in all_float_dtypes], c_attrs=["__device__"]
)
def _log[T: DType](x: T) -> T:
    return log(x)


@element_wise_module_cuda.compile_fn(
    types=[SpecItem({"T": ctype(dt)}, f"_primitive_log2_{cname(dt)}") for dt in all_float_dtypes],
    c_attrs=["__device__"],
)
def _log2[T: DType](x: T) -> T:
    return log2(x)


@element_wise_module_cuda.compile_fn(
    types=[SpecItem({"T": ctype(dt)}, f"_primitive_log10_{cname(dt)}") for dt in all_float_dtypes],
    c_attrs=["__device__"],
)
def _log10[T: DType](x: T) -> T:
    return log10(x)


@element_wise_module_cuda.compile_fn(
    types=[SpecItem({"T": ctype(dt)}, f"_primitive_relu_{cname(dt)}") for dt in all_float_dtypes],
    c_attrs=["__device__"],
)
def _relu[T: DType](x: T) -> T:
    return x if x > 0 else 0  # pyrefly: ignore [bad-return,unsupported-operation]


@element_wise_module_cuda.compile_fn(
    types=[SpecItem({"T": ctype(dt)}, f"_primitive_square_{cname(dt)}") for dt in all_float_dtypes],
    c_attrs=["__device__"],
)
def _square[T: DType](x: T) -> T:
    return x * x


@element_wise_module_cuda.compile_fn(
    types=[SpecItem({"T": ctype(dt)}, f"_primitive_sqrt_{cname(dt)}") for dt in all_float_dtypes],
    c_attrs=["__device__"],
)
def _sqrt[T: DType](x: T) -> T:
    return sqrt(x)


@element_wise_module_cuda.compile_fn(
    types=[SpecItem({"T": ctype(dt)}, f"_primitive_sin_{cname(dt)}") for dt in all_float_dtypes], c_attrs=["__device__"]
)
def _sin[T: DType](x: T) -> T:
    return sin(x)


@element_wise_module_cuda.compile_fn(
    types=[SpecItem({"T": ctype(dt)}, f"_primitive_cos_{cname(dt)}") for dt in all_float_dtypes], c_attrs=["__device__"]
)
def _cos[T: DType](x: T) -> T:
    return cos(x)


@element_wise_module_cuda.compile_fn(
    types=[SpecItem({"T": ctype(dt)}, f"_primitive_tan_{cname(dt)}") for dt in all_float_dtypes], c_attrs=["__device__"]
)
def _tan[T: DType](x: T) -> T:
    return tan(x)


@element_wise_module_cuda.compile_fn(
    types=[SpecItem({"T": ctype(dt)}, f"_primitive_asin_{cname(dt)}") for dt in all_float_dtypes],
    c_attrs=["__device__"],
)
def _asin[T: DType](x: T) -> T:
    return asin(x)


@element_wise_module_cuda.compile_fn(
    types=[SpecItem({"T": ctype(dt)}, f"_primitive_acos_{cname(dt)}") for dt in all_float_dtypes],
    c_attrs=["__device__"],
)
def _acos[T: DType](x: T) -> T:
    return acos(x)


@element_wise_module_cuda.compile_fn(
    types=[SpecItem({"T": ctype(dt)}, f"_primitive_atan_{cname(dt)}") for dt in all_float_dtypes],
    c_attrs=["__device__"],
)
def _atan[T: DType](x: T) -> T:
    return atan(x)


@element_wise_module_cuda.compile_fn(
    types=[SpecItem({"T": ctype(dt)}, f"_primitive_sinh_{cname(dt)}") for dt in all_float_dtypes],
    c_attrs=["__device__"],
)
def _sinh[T: DType](x: T) -> T:
    return sinh(x)


@element_wise_module_cuda.compile_fn(
    types=[SpecItem({"T": ctype(dt)}, f"_primitive_cosh_{cname(dt)}") for dt in all_float_dtypes],
    c_attrs=["__device__"],
)
def _cosh[T: DType](x: T) -> T:
    return cosh(x)


@element_wise_module_cuda.compile_fn(
    types=[SpecItem({"T": ctype(dt)}, f"_primitive_tanh_{cname(dt)}") for dt in all_float_dtypes],
    c_attrs=["__device__"],
)
def _tanh[T: DType](x: T) -> T:
    return tanh(x)


unary_ops = [
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
]

numerical_unary_kernel_specs: list[SpecItem] = []
for dt in all_float_dtypes:
    for op in unary_ops:
        numerical_unary_kernel_specs.append(
            SpecItem({"T": ctype(dt), "Op": f"_primitive_{op}_{cname(dt)}"}, f"_{op}_{cname(dt)}")
        )


@element_wise_module_cuda.compile_fn(numerical_unary_kernel_specs, c_attrs=["__global__"])
def element_wise_unary_kernel[T: DType, Op: Callable](
    a: list[T],
    a_off: i64,
    a_stride: i64,
    c: list[T],
    c_off: i64,
    c_stride: i64,
    n: i64,
) -> void:
    i: u64 = threadIdx.x + blockIdx.x * blockDim.x
    if i < n:
        c[c_off + i * c_stride] = Op(a[a_off + i * a_stride])  # pyrefly: ignore [not-callable]


numerical_unary_specs: list[SpecItem] = []
for dt in all_float_dtypes:
    for op in unary_ops:
        numerical_unary_specs.append(
            SpecItem({"T": ctype(dt), "Op": f"_{op}_{cname(dt)}"}, f"element_wise_unary_{cname(dt)}__{op}")
        )


@element_wise_module_cuda.compile_fn(numerical_unary_specs, pybind=True)
def element_wise_unary[T: DType, Op: Callable](
    a: list[T], a_off: i64, a_stride: i64, c: list[T], c_off: i64, c_stride: i64, n: i64
) -> None:
    NUM_THREADS: int = 1024
    NUM_BLOCKS: int = (n + NUM_THREADS - 1) // NUM_THREADS
    Op[[[NUM_BLOCKS, NUM_THREADS]]](a, a_off, a_stride, c, c_off, c_stride, n)  # pyrefly: ignore[unsupported-operation]


@element_wise_module_cuda.compile_fn(
    types=[SpecItem({"T": ctype(dt)}, f"arange_kernel_{cname(dt)}") for dt in all_dtypes], c_attrs=["__global__"]
)
def arange_kernel[T: DType](out: list[T], out_offset: i64, out_stride: i64, numel: i64) -> void:
    i: u64 = threadIdx.x + blockIdx.x * blockDim.x
    if i < numel:
        out[out_offset + i * out_stride] = i  # pyrefly: ignore [unsupported-operation]


@element_wise_module_cuda.compile_fn(
    [SpecItem({"T": ctype(dt), "Kernel": f"arange_kernel_{cname(dt)}"}, f"arange_{cname(dt)}") for dt in all_dtypes],
    pybind=True,
)
def arange[T: DType, Kernel: Callable](out: list[T], out_offset: i64, out_stride: i64, numel: i64) -> None:
    NUM_THREADS: int = 1024
    NUM_BLOCKS: int = (numel + NUM_THREADS - 1) // NUM_THREADS
    Kernel[[[NUM_BLOCKS, NUM_THREADS]]](out, out_offset, out_stride, numel)  # pyrefly: ignore[unsupported-operation]


@element_wise_module_cuda.compile_fn(
    types=[SpecItem({"T": ctype(dt)}, f"reshape_copy_kernel_{cname(dt)}") for dt in all_dtypes], c_attrs=["__global__"]
)
def reshape_copy_kernel[T: DType](
    inp: list[T],
    inp_strides: list[i64],
    inp_shape: list[i64],
    inp_offset: i64,
    inp_ndim: i64,
    out: list[T],
    out_strides: list[i64],
    out_shape: list[i64],
    out_offset: i64,
    out_ndim: i64,
    numel: i64,
) -> void:
    inp_idx: i64 = inp_offset
    out_idx: i64 = out_offset
    contiguous_inp_shape: Annotated[list[i64], "__shared__", 16] = []
    shared_inp_shape: Annotated[list[i64], "__shared__", 16] = []
    shared_inp_strides: Annotated[list[i64], "__shared__", 16] = []
    contiguous_out_shape: Annotated[list[i64], "__shared__", 16] = []
    shared_out_shape: Annotated[list[i64], "__shared__", 16] = []
    shared_out_strides: Annotated[list[i64], "__shared__", 16] = []

    i: u64 = threadIdx.x + blockIdx.x * blockDim.x
    if threadIdx.x < inp_ndim:
        shared_inp_shape[threadIdx.x] = inp_shape[threadIdx.x]
        shared_inp_strides[threadIdx.x] = inp_strides[threadIdx.x]
    if threadIdx.x < out_ndim:
        shared_out_shape[threadIdx.x] = out_shape[threadIdx.x]
        shared_out_strides[threadIdx.x] = out_strides[threadIdx.x]
    __syncthreads()
    if threadIdx.x == 0:
        contiguous_inp_shape[inp_ndim - 1] = 1
        for i_in in range(inp_ndim - 2, -1, -1):
            contiguous_inp_shape[i_in] = contiguous_inp_shape[i_in + 1] * shared_inp_shape[i_in + 1]
        contiguous_out_shape[out_ndim - 1] = 1
        for i_out in range(out_ndim - 2, -1, -1):
            contiguous_out_shape[i_out] = contiguous_out_shape[i_out + 1] * shared_out_shape[i_out + 1]
    __syncthreads()
    if i < numel:
        i_copy: u64 = i
        for i_in in range(inp_ndim):
            dim_index: u64 = i_copy // contiguous_inp_shape[i_in]
            i_copy %= contiguous_inp_shape[i_in]
            inp_idx += dim_index * shared_inp_strides[i_in]
        i_copy = i
        for i_out in range(out_ndim):
            dim_index: u64 = i_copy // contiguous_out_shape[i_out]
            i_copy %= contiguous_out_shape[i_out]
            out_idx += dim_index * shared_out_strides[i_out]
        out[out_idx] = inp[inp_idx]


@element_wise_module_cuda.compile_fn(
    [
        SpecItem({"T": ctype(dt), "Kernel": f"reshape_copy_kernel_{cname(dt)}"}, f"reshape_copy_{cname(dt)}")
        for dt in all_dtypes
    ],
    pybind=True,
)
def reshape_copy[T: DType, Kernel: Callable](
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
) -> None:
    NUM_THREADS: i64 = 1024
    NUM_BLOCKS: i64 = (numel + NUM_THREADS - 1) // NUM_THREADS
    if inp_ndim > 16 or out_ndim > 16:
        printf("Max ndim is 16 for reshape_copy on gpu")
        exit(1)
    Kernel[[[NUM_BLOCKS, NUM_THREADS]]](  # pyrefly: ignore[unsupported-operation]
        inp,
        inp_strides,
        inp_shape,
        inp_offset,
        inp_ndim,
        out,
        out_strides,
        out_shape,
        out_offset,
        out_ndim,
        numel,
    )


element_wise_module_cuda = element_wise_module_cuda.compile("nvcc", ldflags=["-lcudart"])
