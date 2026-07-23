import ast
from typing import Annotated

import pytest

from simplendarray import compile_str, ref
from simplendarray.transpiler.transpiler import _c_type, _expr

int32 = int
int64 = int
float32 = float
float64 = float
double = float
void = None


class char:
    pass


class MyType:
    pass


class Obj:
    pass


def test_ref():
    assert ref(42) == [42]


def test_c_type_int64():
    def f(x: int64) -> int64:
        return x

    expected = "long long f(long long x) {\n    return x;\n}"
    assert compile_str(f) == expected


def test_c_type_float():
    def f(x: float) -> float:
        return x

    expected = "float f(float x) {\n    return x;\n}"
    assert compile_str(f) == expected


def test_c_type_float32():
    def f(x: float32) -> float32:
        return x

    expected = "float f(float x) {\n    return x;\n}"
    assert compile_str(f) == expected


def test_c_type_float64():
    def f(x: float64) -> float64:
        return x

    expected = "double f(double x) {\n    return x;\n}"
    assert compile_str(f) == expected


def test_c_type_double():
    def f(x: double) -> double:
        return x

    expected = "double f(double x) {\n    return x;\n}"
    assert compile_str(f) == expected


def test_c_type_bool():
    def f(x: bool) -> bool:
        return x

    expected = "int f(int x) {\n    return x;\n}"
    assert compile_str(f) == expected


def test_c_type_void():
    def f() -> void:
        pass

    expected = "void f() {}"
    assert compile_str(f) == expected


def test_c_type_none_return():
    def f() -> None:
        pass

    expected = "void f() {}"
    assert compile_str(f) == expected


def test_c_type_none_return_with_return():
    def f() -> None:
        return

    expected = "void f() {\n    return;\n}"
    assert compile_str(f) == expected


def test_c_type_none_param():
    def f(x: None) -> int:
        return 0

    with pytest.raises(ValueError, match="parameter 'x' has no type annotation"):
        compile_str(f)


def test_c_type_char():
    def f(x: char) -> char:
        return x

    expected = "char f(char x) {\n    return x;\n}"
    assert compile_str(f) == expected


def test_c_type_str():
    def f(x: str) -> str:
        return x

    expected = "char* f(char* x) {\n    return x;\n}"
    assert compile_str(f) == expected


def test_c_type_list():
    def f(x: list[int]) -> list[int]:
        return x

    expected = "int* f(int* x) {\n    return x;\n}"
    assert compile_str(f) == expected


def test_c_type_unsigned():
    for name, expected in [
        ("u8", "unsigned char"),
        ("u16", "unsigned short"),
        ("u32", "unsigned int"),
        ("u64", "unsigned long long"),
    ]:
        node = ast.Name(id=name)
        assert _c_type(node) == expected


def test_c_type_default():
    class Custom:
        pass

    def f(x: Custom) -> Custom:
        return x

    expected = "Custom f(Custom x) {\n    return x;\n}"
    assert compile_str(f) == expected


def test_none_constant():
    def f() -> int:
        # pyrefly: ignore [bad-return]
        return None

    expected = "int f() {\n    return (NULL);\n}"
    assert compile_str(f) == expected


def test_bool_constant():
    def f() -> int:
        return False

    expected = "int f() {\n    return 0;\n}"
    assert compile_str(f) == expected


def test_str_constant():
    def f() -> int:
        # pyrefly: ignore [bad-return]
        return "hello"

    expected = 'int f() {\n    return "hello";\n}'
    assert compile_str(f) == expected


def test_unary_neg():
    def f(x: int) -> int:
        return -x

    expected = "int f(int x) {\n    return (-x);\n}"
    assert compile_str(f) == expected


def test_unary_not():
    def f(x: bool) -> int:
        return not x

    expected = "int f(int x) {\n    return (!x);\n}"
    assert compile_str(f) == expected


def test_unary_invert():
    def f(x: int) -> int:
        return ~x

    expected = "int f(int x) {\n    return (~x);\n}"
    assert compile_str(f) == expected


def test_subscript():
    def f(a: int) -> int:
        # pyrefly: ignore [bad-index]
        return a[0]

    expected = "int f(int a) {\n    return (a[0]);\n}"
    assert compile_str(f) == expected


def test_non_ref_call():
    def bar(x: int) -> int:
        return x

    def foo(x: int) -> int:
        return bar(x)

    expected = "int foo(int x) {\n    return bar(x);\n}"
    assert compile_str(foo) == expected


def test_ref_call():
    def f(x: int) -> list[int]:
        return ref(x)

    expected = "int* f(int x) {\n    return (&x);\n}"
    assert compile_str(f) == expected


def test_attribute():
    def f(o: Obj) -> int:
        # pyrefly: ignore [missing-attribute]
        return o.attr

    expected = "int f(Obj o) {\n    return (o.attr);\n}"
    assert compile_str(f) == expected


def test_compare_single():
    def f(x: int) -> int:
        return x < 10

    expected = "int f(int x) {\n    return (x < 10);\n}"
    assert compile_str(f) == expected


def test_compare_chained():
    def f(x: int) -> int:
        return 1 < x < 10

    expected = "int f(int x) {\n    return ((1 < x) && (x < 10));\n}"
    assert compile_str(f) == expected


def test_compare_unsupported():
    def f(x: int) -> int:
        return x in [1, 2, 3]

    with pytest.raises(ValueError, match="unsupported comparison"):
        compile_str(f)


def test_bool_op_and():
    def f(x: int, y: int) -> int:
        return x and y

    expected = "int f(int x, int y) {\n    return (x && y);\n}"
    assert compile_str(f) == expected


def test_bool_op_or():
    def f(x: int, y: int) -> int:
        return x or y

    expected = "int f(int x, int y) {\n    return (x || y);\n}"
    assert compile_str(f) == expected


def test_list_expr():
    def f() -> int:
        # pyrefly: ignore [bad-return]
        return [1, 2, 3]

    expected = "int f() {\n    return {1, 2, 3};\n}"
    assert compile_str(f) == expected


def test_tuple_expr():
    def f() -> int:
        # pyrefly: ignore [bad-return]
        return (1, 2, 3)

    expected = "int f() {\n    return 1, 2, 3;\n}"
    assert compile_str(f) == expected


def test_if_exp():
    def f(x: int, y: int) -> int:
        return x if y > 0 else 0

    expected = "int f(int x, int y) {\n    return ((y > 0) ? x : 0);\n}"
    assert compile_str(f) == expected


def test_unsupported_expr():
    def f() -> int:
        # pyrefly: ignore [bad-return]
        return {1: 2}

    with pytest.raises(ValueError, match="unsupported expression"):
        compile_str(f)


def test_unsupported_binary_op():
    def f(x: int, y: int) -> int:
        return x**y

    with pytest.raises(ValueError, match="unsupported binary op"):
        compile_str(f)


def test_unsupported_unary_op():
    class FakeOp:
        pass

    # pyrefly: ignore [bad-argument-type]
    node = ast.UnaryOp(op=FakeOp(), operand=ast.Constant(value=1))
    with pytest.raises(ValueError, match="unsupported unary op"):
        _expr(node)


def test_unsupported_c_type():
    node = ast.Constant(value="test")
    with pytest.raises(ValueError, match="Unsupported type"):
        _c_type(node)


def test_missing_param_annotation():
    def f(x) -> int:
        return x

    with pytest.raises(ValueError, match="parameter 'x' has no type annotation"):
        compile_str(f)


def test_missing_return_annotation():
    def f(x: int):
        return x

    with pytest.raises(ValueError, match="function has no return type annotation"):
        compile_str(f)


def test_bare_return():
    def f() -> void:
        return

    expected = "void f() {\n    return;\n}"
    assert compile_str(f) == expected


def test_variadic_error():
    # pyrefly: ignore [bad-return]
    def f(*args: int) -> int:
        pass

    with pytest.raises(ValueError, match="keyword / variadic / default args not supported"):
        compile_str(f)


def test_kwarg_error():
    # pyrefly: ignore [bad-return]
    def f(**kwargs: int) -> int:
        pass

    with pytest.raises(ValueError, match="keyword / variadic / default args not supported"):
        compile_str(f)


def test_empty_body():
    # pyrefly: ignore [bad-return]
    def f(x: int) -> int:
        pass

    expected = "int f(int x) {}"
    assert compile_str(f) == expected


def test_assign():
    # pyrefly: ignore [bad-return]
    def f() -> int:
        x = 1  # noqa: F841

    expected = "int f() {\n    x = 1;\n}"
    assert compile_str(f) == expected


def test_ann_assign_with_value():
    # pyrefly: ignore [bad-return]
    def f() -> int:
        x: int = 1  # noqa: F841

    expected = "int f() {\n    int x = 1;\n}"
    assert compile_str(f) == expected


def test_ann_assign_without_value():
    # pyrefly: ignore [bad-return]
    def f() -> int:
        x: int  # noqa: F842

    expected = "int f() {\n    int x;\n}"
    assert compile_str(f) == expected


def test_unsupported_aug_assign():
    # pyrefly: ignore [bad-return]
    def f(x: int) -> int:  # noqa: F841
        x **= 1

    with pytest.raises(ValueError, match="unsupported aug-assign op"):
        compile_str(f)


def test_for_orelse_error():
    # pyrefly: ignore [bad-return]
    def f() -> int:
        for i in range(3):
            pass
        else:
            pass

    with pytest.raises(ValueError, match="for-else not supported"):
        compile_str(f)


def test_for_non_range_error():
    # pyrefly: ignore [bad-return]
    def f() -> int:
        for i in [1, 2, 3]:
            pass

    with pytest.raises(ValueError, match=r"only range\(\) supported as for-loop iter"):
        compile_str(f)


def test_for_range_2_args():
    # pyrefly: ignore [bad-return]
    def f() -> int:
        for i in range(1, 10):
            pass

    expected = "int f() {\n    for (int i = 1; i < 10; i++) {\n\n    }\n}"
    assert compile_str(f) == expected


def test_for_range_3_args():
    # pyrefly: ignore [bad-return]
    def f() -> int:
        for i in range(1, 10, 2):
            pass

    expected = "int f() {\n    for (int i = 1; i < 10; i += 2) {\n\n    }\n}"
    assert compile_str(f) == expected


def test_for_range_3_args_neg_step():
    # pyrefly: ignore [bad-return]
    def f() -> int:
        for i in range(10, 0, -1):
            pass

    expected = "int f() {\n    for (int i = 10; i > 0; i += (-1)) {\n\n    }\n}"
    assert compile_str(f) == expected


def test_for_range_variable_step_falls_back_to_less_than():
    # pyrefly: ignore [bad-return]
    def f(step: int) -> int:
        for i in range(10, 0, step):
            pass

    expected = "int f(int step) {\n    for (int i = 10; i < 0; i += step) {\n\n    }\n}"
    assert compile_str(f) == expected


def test_while_orelse_error():
    def f() -> int:
        while True:
            pass
        else:
            pass

    with pytest.raises(ValueError, match="while-else not supported"):
        compile_str(f)


def test_while_basic():
    def f() -> int:
        while True:
            pass

    expected = "int f() {\n    while (1) {\n\n    }\n}"
    assert compile_str(f) == expected


def test_if_only():
    def f(x: int) -> int:
        if x > 0:
            return 1
        return 0

    expected = "int f(int x) {\n    if ((x > 0)) {\n        return 1;\n    }\n    return 0;\n}"
    assert compile_str(f) == expected


def test_if_else():
    def f(x: int) -> int:
        if x > 0:
            return 1
        else:
            return 0

    expected = "int f(int x) {\n    if ((x > 0)) {\n        return 1;\n    } else {\n        return 0;\n    }\n}"
    assert compile_str(f) == expected


def test_if_elif():
    # pyrefly: ignore [bad-return]
    def f(x: int) -> int:
        if x > 0:
            return 1
        elif x < 0:
            return -1

    expected = (
        "int f(int x) {\n    if ((x > 0)) {\n        return 1;"
        "\n    } else if ((x < 0)) {\n        return (-1);\n    }\n}"
    )
    assert compile_str(f) == expected


def test_if_elif_else():
    def f(x: int) -> int:
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            return 0

    expected = (
        "int f(int x) {\n    if ((x > 0)) {\n        return 1;\n    } "
        "else if ((x < 0)) {\n        return (-1);\n    } else {\n        return 0;\n    }\n}"
    )
    assert compile_str(f) == expected


def test_break():
    # pyrefly: ignore [bad-return]
    def f() -> int:
        for i in range(3):
            break

    expected = "int f() {\n    for (int i = 0; i < 3; i++) {\n        break;\n    }\n}"
    assert compile_str(f) == expected


def test_continue():
    # pyrefly: ignore [bad-return]
    def f() -> int:
        for i in range(3):
            continue

    expected = "int f() {\n    for (int i = 0; i < 3; i++) {\n        continue;\n    }\n}"
    assert compile_str(f) == expected


def test_expr_stmt():
    # pyrefly: ignore [bad-return]
    def f(x: int) -> int:
        x

    expected = "int f(int x) {\n    x;\n}"
    assert compile_str(f) == expected


def test_pass():
    def f() -> void:
        pass

    expected = "void f() {}"
    assert compile_str(f) == expected


def test_unsupported_stmt():
    # pyrefly: ignore [bad-return]
    def f() -> int:
        with open("file"):
            pass

    with pytest.raises(ValueError, match="unsupported statement"):
        compile_str(f)


def test_compile_str_error():
    class NotAFunction:
        pass

    with pytest.raises(ValueError, match="only function definitions can be compiled"):
        compile_str(NotAFunction)


def test_annotated_array_attr_with_size():
    def f(x: list[int]) -> list[int]:
        y: Annotated[list[int], "__shared__", 32] = x
        return y

    expected = "int* f(int* x) {\n    __shared__ int y[32];\n    return y;\n}"
    assert compile_str(f) == expected


def test_annotated_array_attr_no_size():
    def f(x: list[int]) -> list[int]:
        y: Annotated[list[int], "__shared__"] = x
        return y

    expected = "int* f(int* x) {\n    __shared__ int y[];\n    return y;\n}"
    assert compile_str(f) == expected


def test_annotated_regular_array():
    def f(x: list[int]) -> list[int]:
        y: Annotated[list[int], 32] = []
        return y

    expected = "int* f(int* x) {\n    int y[32] = {};\n    return y;\n}"
    assert compile_str(f) == expected


def test_annotated_array_attr_size_no_value():
    def f(x: list[int]) -> list[int]:
        y: Annotated[list[int], "__shared__", 32]  # noqa: F842
        return x

    expected = "int* f(int* x) {\n    __shared__ int y[32];\n    return x;\n}"
    assert compile_str(f) == expected


def test_annotated_volatile_with_value():
    def f(x: int) -> int:
        y: Annotated[int, "volatile"] = x
        return y

    expected = "int f(int x) {\n    volatile int y = x;\n    return y;\n}"
    assert compile_str(f) == expected


def test_annotated_volatile_no_value():
    def f(x: int) -> int:
        y: Annotated[int, "volatile"]  # noqa: F842
        return x

    expected = "int f(int x) {\n    volatile int y;\n    return x;\n}"
    assert compile_str(f) == expected


def test_annotated_non_array_multi_attr():
    def f(x: int) -> int:
        y: Annotated[int, "const", "volatile"] = x
        return y

    expected = "int f(int x) {\n    const volatile int y = x;\n    return y;\n}"
    assert compile_str(f) == expected


def test_annotated_array_multi_attr():
    def f(x: list[int]) -> list[int]:
        y: Annotated[list[int], "extern", "__shared__", 64] = x
        return y

    expected = "int* f(int* x) {\n    extern __shared__ int y[64];\n    return y;\n}"
    assert compile_str(f) == expected


def test_annotated_non_array_int_attr():
    def f(x: int) -> int:
        y: Annotated[int, 42] = x  # noqa: F841
        return 0

    expected = "int f(int x) {\n    42 int y = x;\n    return 0;\n}"
    assert compile_str(f) == expected


def test_original():
    def add(x: int, y: int) -> int:
        for i in range(3):
            for j in range(3):
                x += i + j
        return x + y

    expected = (
        "int add(int x, int y) {\n"
        "    for (int i = 0; i < 3; i++) {\n"
        "        for (int j = 0; j < 3; j++) {\n"
        "            x += (i + j);\n"
        "        }\n"
        "    }\n"
        "    return (x + y);\n"
        "}"
    )
    assert compile_str(add) == expected
