from typing import Annotated, Literal

type f32 = Annotated[float, "f", "float", "float"]
type f64 = Annotated[float, "d", "double", "double"]
type u8 = Annotated[int, "B", "unsigned_char", "unsigned char"]
type i8 = Annotated[int, "b", "char", "char"]
type u16 = Annotated[int, "H", "unsigned_short", "unsigned short"]
type i16 = Annotated[int, "h", "short", "short"]
type u32 = Annotated[int, "I", "unsigned_int", "unsigned int"]
type i32 = Annotated[int, "i", "int", "int"]
type u64 = Annotated[int, "Q", "unsigned_long_long", "unsigned long long"]
type i64 = Annotated[int, "q", "long_long", "long long"]


def typecode(x) -> str:
    if x not in all_dtypes:
        raise ValueError("Not a SNDA DType")  # pragma: no cover
    return x.__value__.__metadata__[0]


def cname(x) -> str:
    if x not in all_dtypes:
        raise ValueError("Not a SNDA DType")  # pragma: no cover
    return x.__value__.__metadata__[1]


def ctype(x) -> str:
    if x not in all_dtypes:
        raise ValueError("Not a SNDA DType")  # pragma: no cover
    return x.__value__.__metadata__[2]


DType = f32 | f64 | u8 | i8 | u16 | i16 | u32 | i32 | u64 | i64

all_dtypes = [f32, f64, u8, i8, u16, i16, u32, i32, u64, i64]
all_float_dtypes = [f32, f64]
all_int_dtypes = [u8, i8, u16, i16, u32, i32, u64, i64]
all_typecodes: list[str] = ["f", "d", "B", "b", "H", "h", "I", "i", "Q", "q"]
dtype_maps = {
    "f32": f32,
    "float": f32,
    "float32": f32,
    "double": f64,
    "f64": f64,
    "float64": f64,
    "u8": u8,
    "uint8": u8,
    "i8": i8,
    "int8": i8,
    "u16": u16,
    "i16": i16,
    "int16": i16,
    "u32": u32,
    "uint32": u32,
    "i32": i32,
    "int32": i32,
    "u64": u64,
    "uint64": u64,
    "i64": i64,
    "int64": i64,
} | {typecode(dtype): dtype for dtype in all_dtypes}


def get_dtype(dtype: str | type[DType]) -> type[DType]:
    """Given a DType class or array.array typecode, return the corresponding DType class."""
    if dtype in all_dtypes:
        return dtype  # pragma: no cover # type: ignore
    if dtype in dtype_maps:
        return dtype_maps[dtype]  # type: ignore
    raise KeyError(f"Unknown dtype: {dtype}")  # pragma: no cover


type Device = Literal["cpu", "gpu"]
type void_ptr = list[None]
type const_char_ptr = str
NULL: void_ptr = []
