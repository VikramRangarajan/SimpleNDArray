"""Tests for simplendarray.transpiler.runtime"""

import shutil
import sysconfig

import pytest

from simplendarray.buffer_cuda import BufferCuda
from simplendarray.transpiler.runtime import CFunction, DType, PythonModule, SpecItem

cuda_available = shutil.which("nvcc") is not None

double = float


def _func_entry(name, params, ret_c_type, ret_is_bool, c_source):
    return CFunction(
        name=name,
        c_source=c_source,
        params=params,
        ret_c_type=ret_c_type,
        ret_is_bool=ret_is_bool,
        c_attrs=[],
        pybind=True,
        group=None,
        dispatch_key=None,
    )


def _param(name, c_type, is_bool=False, is_string=False):
    return {"name": name, "c_type": c_type, "is_bool": is_bool, "is_string": is_string}


def test_module_init():
    m = PythonModule()
    assert m._funcs == []
    assert m._module is None
    assert not m._compiled


def test_compile_fn_registers_multiple():
    m = PythonModule()

    @m.compile_fn(pybind=True)
    def f1() -> int:
        return 1

    @m.compile_fn(pybind=True)
    def f2() -> int:
        return 2

    assert len(m._funcs) == 2
    assert m._funcs[0].name == "f1"
    assert m._funcs[1].name == "f2"


def test_register_normal():
    m = PythonModule()

    def add(a: int, b: int) -> int:
        return a + b

    m._register(add, pybind=True, c_attrs=[])

    assert len(m._funcs) == 1
    func = m._funcs[0]
    assert func.name == "add"
    assert func.params == [
        {"name": "a", "c_type": "int", "is_bool": False, "is_string": False},
        {"name": "b", "c_type": "int", "is_bool": False, "is_string": False},
    ]
    assert func.ret_c_type == "int"
    assert not func.ret_is_bool
    assert "int add(int a, int b)" in func.c_source
    assert "return (a + b);" in func.c_source


def test_register_bool_param():
    m = PythonModule()

    def f(x: bool) -> int:
        return x

    m._register(f, pybind=True, c_attrs=[])

    assert m._funcs[0].params[0] == {"name": "x", "c_type": "int", "is_bool": True, "is_string": False}


def test_register_bool_return():
    m = PythonModule()

    def is_even(x: int) -> bool:
        return (x % 2) == 0

    m._register(is_even, pybind=True, c_attrs=[])

    assert m._funcs[0].ret_c_type == "int"
    assert m._funcs[0].ret_is_bool


def test_register_void():
    m = PythonModule()

    def noop() -> None:
        pass

    m._register(noop, pybind=True, c_attrs=[])

    assert m._funcs[0].ret_c_type == "void"
    assert not m._funcs[0].ret_is_bool


def test_register_str():
    m = PythonModule()

    def greet(name: str) -> str:
        return name

    m._register(greet, pybind=True, c_attrs=[])

    assert m._funcs[0].params[0] == {"name": "name", "c_type": "char*", "is_bool": False, "is_string": True}
    assert m._funcs[0].ret_c_type == "char*"


def test_register_raises_for_non_function():
    m = PythonModule()

    class NotAFunction:
        pass

    with pytest.raises(ValueError, match="only function definitions can be compiled"):
        m._register(NotAFunction, pybind=True, c_attrs=[])


def test_register_raises_missing_param_annotation():
    m = PythonModule()

    def f(x) -> int:
        return x

    with pytest.raises(ValueError, match="parameter 'x' has no type annotation"):
        m._register(f, pybind=True, c_attrs=[])


INT_SRC = "int add(int a, int b) {\n    return (a + b);\n}"


def test_generate_wrapper_int():
    m = PythonModule()
    m._funcs = [_func_entry("add", [_param("a", "int"), _param("b", "int")], "int", False, INT_SRC)]
    wrapper = m._generate_wrapper(m._funcs[0])

    assert "static PyObject* add_wrapper(PyObject* self, PyObject* args)" in wrapper
    assert "int a;" in wrapper
    assert "int b;" in wrapper
    assert 'PyArg_ParseTuple(args, "ii", &a, &b)' in wrapper
    assert "int result = add(a, b);" in wrapper
    assert "PyLong_FromLong(result);" in wrapper


def test_generate_wrapper_no_args():
    m = PythonModule()
    m._funcs = [_func_entry("noop", [], "void", False, "void noop() {}")]
    wrapper = m._generate_wrapper(m._funcs[0])

    assert "(void)self;" in wrapper
    assert 'PyArg_ParseTuple(args, "")' in wrapper
    assert "noop();" in wrapper
    assert "Py_RETURN_NONE;" in wrapper


def test_generate_wrapper_void_return():
    m = PythonModule()
    m._funcs = [_func_entry("noop", [_param("x", "int")], "void", False, "void noop(int x) {}")]
    wrapper = m._generate_wrapper(m._funcs[0])

    assert "int x;" in wrapper
    assert "noop(x);" in wrapper
    assert "Py_RETURN_NONE;" in wrapper


def test_generate_wrapper_bool_param():
    m = PythonModule()
    m._funcs = [_func_entry("f", [_param("x", "int", is_bool=True)], "int", False, INT_SRC)]
    wrapper = m._generate_wrapper(m._funcs[0])

    assert "int x;" in wrapper
    assert 'PyArg_ParseTuple(args, "p", &x)' in wrapper


def test_generate_wrapper_bool_return():
    m = PythonModule()
    m._funcs = [_func_entry("is_even", [_param("x", "int")], "int", True, INT_SRC)]
    wrapper = m._generate_wrapper(m._funcs[0])

    assert "int result = is_even(x);" in wrapper
    assert "PyBool_FromLong(result);" in wrapper


def test_generate_wrapper_float():
    m = PythonModule()
    src = "float add(float a, float b) {\n    return (a + b);\n}"
    m._funcs = [_func_entry("add", [_param("a", "float"), _param("b", "float")], "float", False, src)]
    wrapper = m._generate_wrapper(m._funcs[0])

    assert "float a;" in wrapper
    assert "float b;" in wrapper
    assert 'PyArg_ParseTuple(args, "ff", &a, &b)' in wrapper
    assert "float result = add(a, b);" in wrapper
    assert "PyFloat_FromDouble(result);" in wrapper


def test_generate_wrapper_double():
    m = PythonModule()
    src = "double add(double a, double b) {\n    return (a + b);\n}"
    m._funcs = [_func_entry("add", [_param("a", "double"), _param("b", "double")], "double", False, src)]
    wrapper = m._generate_wrapper(m._funcs[0])

    assert "double a;" in wrapper
    assert "double b;" in wrapper
    assert 'PyArg_ParseTuple(args, "dd", &a, &b)' in wrapper
    assert "double result = add(a, b);" in wrapper
    assert "PyFloat_FromDouble(result);" in wrapper


def test_generate_wrapper_long_long():
    m = PythonModule()
    src = "long long add(long long a, long long b) {\n    return (a + b);\n}"
    m._funcs = [_func_entry("add", [_param("a", "long long"), _param("b", "long long")], "long long", False, src)]
    wrapper = m._generate_wrapper(m._funcs[0])

    assert "long long a;" in wrapper
    assert "long long b;" in wrapper
    assert 'PyArg_ParseTuple(args, "LL", &a, &b)' in wrapper
    assert "long long result = add(a, b);" in wrapper
    assert "PyLong_FromLongLong(result);" in wrapper


def test_generate_wrapper_char():
    m = PythonModule()
    src = "char f(char x) {\n    return x;\n}"
    m._funcs = [_func_entry("f", [_param("x", "char")], "char", False, src)]
    wrapper = m._generate_wrapper(m._funcs[0])

    assert "char x;" in wrapper
    assert 'PyArg_ParseTuple(args, "c", &x)' in wrapper
    assert "char result = f(x);" in wrapper
    assert "PyLong_FromLong(result);" in wrapper


def test_generate_wrapper_str():
    m = PythonModule()
    src = "char* greet(char* name) {\n    return name;\n}"
    m._funcs = [_func_entry("greet", [_param("name", "char*", is_string=True)], "char*", False, src)]
    wrapper = m._generate_wrapper(m._funcs[0])

    assert "char* name;" in wrapper
    assert 'PyArg_ParseTuple(args, "s", &name)' in wrapper
    assert "char* result = greet(name);" in wrapper
    assert "PyUnicode_FromString(result);" in wrapper


def test_generate_wrapper_ptr_param():
    m = PythonModule()
    src = ("void add(float* a, float* b, float* c, int n) {\n    for (int i = 0; i < n; i++) {\n",)
    "        c[i] = (a[i] + b[i]);\n    }\n}"
    m._funcs = [
        _func_entry(
            "add",
            [_param("a", "float*"), _param("b", "float*"), _param("c", "float*"), _param("n", "int")],
            "void",
            False,
            src,
        )
    ]
    wrapper = m._generate_wrapper(m._funcs[0])

    assert "unsigned long long a;" in wrapper
    assert "unsigned long long b;" in wrapper
    assert "unsigned long long c;" in wrapper
    assert "int n;" in wrapper
    assert 'PyArg_ParseTuple(args, "KKKi", &a, &b, &c, &n)' in wrapper
    assert "add((float*)a, (float*)b, (float*)c, n);" in wrapper
    assert "Py_RETURN_NONE;" in wrapper


def test_generate_wrapper_ptr_param_double():
    m = PythonModule()
    src = (
        "void scale(double* x, double s, int n) {\n    for (int i = 0; i < n; i++) {\n",
        "        x[i] = (x[i] * s);\n    }\n}",
    )
    m._funcs = [
        _func_entry("scale", [_param("x", "double*"), _param("s", "double"), _param("n", "int")], "void", False, src)
    ]
    wrapper = m._generate_wrapper(m._funcs[0])

    assert "unsigned long long x;" in wrapper
    assert "double s;" in wrapper
    assert "int n;" in wrapper
    assert 'PyArg_ParseTuple(args, "Kdi", &x, &s, &n)' in wrapper
    assert "scale((double*)x, s, n);" in wrapper
    assert "Py_RETURN_NONE;" in wrapper


def test_generate_wrapper_ptr_param_int():
    m = PythonModule()
    src = ("int sum(int* data, int n) {\n    int result = 0;\n    for (int i = 0; i < n; i++) {\n",)
    "        result = (result + data[i]);\n    }\n    return result;\n}"
    m._funcs = [_func_entry("sum", [_param("data", "int*"), _param("n", "int")], "int", False, src)]
    wrapper = m._generate_wrapper(m._funcs[0])

    assert "unsigned long long data;" in wrapper
    assert "int n;" in wrapper
    assert 'PyArg_ParseTuple(args, "Ki", &data, &n)' in wrapper
    assert "int result = sum((int*)data, n);" in wrapper


def test_generate_wrapper_unsupported_param_type():
    m = PythonModule()
    m._funcs = [_func_entry("f", [_param("x", "SomeType*")], "int", False, INT_SRC)]

    with pytest.raises(ValueError, match="unsupported param type"):
        m._generate_wrapper(m._funcs[0])


def test_generate_wrapper_unsupported_return_type():
    m = PythonModule()
    m._funcs = [_func_entry("f", [_param("x", "int")], "int*", False, INT_SRC)]

    with pytest.raises(ValueError, match="unsupported return type"):
        m._generate_wrapper(m._funcs[0])


def test_generate_extension_single_function():
    m = PythonModule()
    m._funcs = [_func_entry("add", [_param("a", "int"), _param("b", "int")], "int", False, INT_SRC)]
    ext = m._generate_extension("_test_mod")

    assert "#define PY_SSIZE_T_CLEAN" in ext
    assert "#include <Python.h>" in ext
    assert INT_SRC in ext
    assert "_test_mod_methods" in ext
    assert "_test_mod_module" in ext
    assert "PyInit__test_mod" in ext
    assert "PyModule_Create" in ext
    assert "{NULL, NULL, 0, NULL}" in ext
    assert '"add"' in ext
    assert "add_wrapper" in ext


def test_generate_extension_multiple_functions():
    m = PythonModule()
    src2 = "int mul(int a, int b) {\n    return (a * b);\n}"
    m._funcs = [
        _func_entry("add", [_param("a", "int"), _param("b", "int")], "int", False, INT_SRC),
        _func_entry("mul", [_param("a", "int"), _param("b", "int")], "int", False, src2),
    ]
    ext = m._generate_extension("_test_mod")

    assert '"add"' in ext
    assert '"mul"' in ext
    assert "add_wrapper" in ext
    assert "mul_wrapper" in ext
    assert src2 in ext
    # Both method defs should be present
    assert ext.count("PyModuleDef_HEAD_INIT") == 1


def test_compile_raises_when_no_functions():
    m = PythonModule()
    with pytest.raises(ValueError, match="no functions registered"):
        m.compile()


def test_compile_raises_when_already_compiled():
    m = PythonModule()

    @m.compile_fn(pybind=True)
    def add(a: int, b: int) -> int:
        return a + b

    m.compile()
    with pytest.raises(RuntimeError, match="module already compiled"):
        m.compile()


def test_compile_fails_gracefully_with_bad_compiler():
    m = PythonModule()

    @m.compile_fn(pybind=True)
    def add(a: int, b: int) -> int:
        return a + b

    with pytest.raises(OSError):
        m.compile(compiler="nonexistent_compiler_xyz")


def test_compile_skips_nvcc_when_not_available(monkeypatch):
    import shutil

    monkeypatch.setattr(shutil, "which", lambda cmd: None if cmd == "nvcc" else shutil.which(cmd))
    m = PythonModule()

    @m.compile_fn(pybind=True)
    def add(a: int, b: int) -> int:
        return a + b

    result = m.compile(compiler="nvcc")
    assert result is m
    assert not m._compiled
    with pytest.raises(AttributeError, match="has no attribute 'add'"):
        m.add(1, 2)


def test_compile_fails_with_bad_c_source():
    m = PythonModule()
    bad_src = "int add(int a, int b) {\n    bad stuff here\n}"
    m._funcs = [_func_entry("add", [_param("a", "int"), _param("b", "int")], "int", False, bad_src)]
    with pytest.raises(RuntimeError, match="compilation failed"):
        m.compile()


# Module.compile -- integration (end-to-end)


def test_compile_and_call_int():
    m = PythonModule()

    @m.compile_fn(pybind=True)
    def add(a: int, b: int) -> int:
        return a + b

    m.compile()
    assert m.add(3, 4) == 7
    assert m.add(-1, 1) == 0
    assert m.add(100, 200) == 300


def test_compile_and_call_bool():
    m = PythonModule()

    @m.compile_fn(pybind=True)
    def is_even(x: int) -> bool:
        return (x % 2) == 0

    m.compile()
    assert m.is_even(4) is True
    assert m.is_even(7) is False


def test_compile_and_call_void():
    m = PythonModule()

    @m.compile_fn(pybind=True)
    def noop() -> None:
        pass

    m.compile()
    assert m.noop() is None


def test_compile_and_call_str():
    m = PythonModule()

    @m.compile_fn(pybind=True)
    def greet(name: str) -> str:
        return name

    m.compile()
    assert m.greet("world") == "world"
    assert m.greet("hello") == "hello"


def test_compile_and_call_float():
    m = PythonModule()

    @m.compile_fn(pybind=True)
    def add(a: float, b: double) -> double:
        return a + b

    m.compile()
    assert m.add(1.5, 2.5) == 4.0
    assert abs(m.add(0.1, 0.2) - 0.3) < 1e-6


def test_compile_and_call_ptr_float():
    m = PythonModule()
    import array

    @m.compile_fn(pybind=True)
    def element_wise_add(a: list[float], b: list[float], c: list[float], n: int) -> None:
        for i in range(n):
            c[i] = a[i] + b[i]

    m.compile()
    a = array.array("f", [1.0, 2.0, 3.0])
    b = array.array("f", [4.0, 5.0, 6.0])
    c = array.array("f", [0.0, 0.0, 0.0])
    m.element_wise_add(a.buffer_info()[0], b.buffer_info()[0], c.buffer_info()[0], 3)
    assert list(c) == [5.0, 7.0, 9.0]


def test_compile_and_call_ptr_int():
    m = PythonModule()
    import array

    @m.compile_fn(pybind=True)
    def sum_array(data: list[int], n: int) -> int:
        s: int = 0
        for i in range(n):
            s = s + data[i]
        return s

    m.compile()
    arr = array.array("i", [10, 20, 30, 40])
    assert m.sum_array(arr.buffer_info()[0], 4) == 100


def test_compile_and_call_ptr_double():
    m = PythonModule()
    import array

    @m.compile_fn(pybind=True)
    def scale(data: list[double], scalar: double, n: int) -> None:
        for i in range(n):
            data[i] = data[i] * scalar

    m.compile()
    arr = array.array("d", [1.0, 2.0, 3.0])
    m.scale(arr.buffer_info()[0], 10.0, 3)
    assert list(arr) == [10.0, 20.0, 30.0]


def test_compile_and_call_recursion():
    m = PythonModule()

    @m.compile_fn(pybind=True)
    def fibonacci(n: int) -> int:
        if n <= 1:
            return n
        return fibonacci(n - 1) + fibonacci(n - 2)

    m.compile()
    assert m.fibonacci(0) == 0
    assert m.fibonacci(1) == 1
    assert m.fibonacci(10) == 55


def test_compile_multiple_functions():
    m = PythonModule()

    @m.compile_fn(pybind=True)
    def add(a: int, b: int) -> int:
        return a + b

    @m.compile_fn(pybind=True)
    def mul(a: int, b: int) -> int:
        return a * b

    m.compile()
    assert m.add(3, 4) == 7
    assert m.mul(3, 4) == 12


def test_compile_with_cflags():
    m = PythonModule()

    @m.compile_fn(pybind=True)
    def add(a: int, b: int) -> int:
        return a + b

    m.compile(cflags=["-O2"])
    assert m.add(2, 3) == 5


def test_compile_with_ldflags():
    m = PythonModule()

    @m.compile_fn(pybind=True)
    def add(a: int, b: int) -> int:
        return a + b

    m.compile(ldflags=["-O1"])
    assert m.add(2, 3) == 5


def test_compile_fallback_include_path(monkeypatch):
    """Cover the fallback path when INCLUDEPY is not set."""
    orig = sysconfig.get_config_var

    def mock_get_config_var(name: str):
        if name == "INCLUDEPY":
            return None
        return orig(name)

    monkeypatch.setattr(sysconfig, "get_config_var", mock_get_config_var)
    m = PythonModule()

    @m.compile_fn(pybind=True)
    def add(a: int, b: int) -> int:
        return a + b

    m.compile()
    assert m.add(2, 3) == 5


def test_compile_cache_hit(monkeypatch):
    import subprocess

    def add(a: int, b: int) -> int:
        return a + b

    m = PythonModule()
    m.compile_fn(pybind=True)(add)
    m.compile()
    assert m.add(3, 4) == 7

    run_calls: list[object] = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: run_calls.append((a, kw)))
    m2 = PythonModule()
    m2.compile_fn(pybind=True)(add)
    m2.compile()
    assert m2.add(3, 4) == 7
    assert len(run_calls) == 0, "expected cache hit, but subprocess.run was called"


def test_compile_cache_env_var(monkeypatch, tmp_path):
    monkeypatch.setattr(PythonModule, "_cache_dir_override", None)
    monkeypatch.setenv("SNDA_CACHE_HOME", str(tmp_path / "snda_cache"))
    m = PythonModule()

    @m.compile_fn(pybind=True)
    def add(a: int, b: int) -> int:
        return a + b

    m.compile()
    assert m.add(3, 4) == 7
    assert (tmp_path / "snda_cache").exists()


def test_compile_fails_when_spec_is_none(monkeypatch):
    import importlib.util

    m = PythonModule()

    @m.compile_fn(pybind=True)
    def add(a: int, b: int) -> int:
        return a + b

    def mock_spec_from_file_location(name, path):
        return None

    monkeypatch.setattr(importlib.util, "spec_from_file_location", mock_spec_from_file_location)
    with pytest.raises(RuntimeError, match="failed to create module spec"):
        m.compile()


def test_compile_separate_modules_independent():
    m1 = PythonModule()
    m2 = PythonModule()

    @m1.compile_fn(pybind=True)
    def add(a: int, b: int) -> int:
        return a + b

    @m2.compile_fn(pybind=True)
    def mul(a: int, b: int) -> int:
        return a * b

    m1.compile()
    m2.compile()
    assert m1.add(3, 4) == 7
    assert m2.mul(3, 4) == 12


# Module.__getattr__


def test_getattr_before_compile_raises():
    m = PythonModule()

    @m.compile_fn(pybind=True)
    def add(a: int, b: int) -> int:
        return a + b

    with pytest.raises(AttributeError, match="has no attribute 'add'"):
        m.add(1, 2)


def test_getattr_unknown_raises():
    m = PythonModule()
    with pytest.raises(AttributeError, match="has no attribute 'nonexistent'"):
        m.nonexistent


def test_getattr_unknown_after_compile_raises():
    m = PythonModule()

    @m.compile_fn(pybind=True)
    def add(a: int, b: int) -> int:
        return a + b

    m.compile()
    with pytest.raises(AttributeError, match="has no attribute 'nonexistent'"):
        m.nonexistent


# Generic specialization


def test_compile_fn_with_types_generates_specializations():
    m = PythonModule()

    @m.compile_fn([SpecItem({"T": "int"}, "add_int"), SpecItem({"T": "float"}, "add_float")])
    def add[T: DType](a: T, b: T) -> T:
        return a + b

    names = {f.name for f in m._funcs}
    assert "add_int" in names
    assert "add_float" in names


def test_compile_fn_with_types_raises_for_non_function():
    m = PythonModule()

    class NotAFunction:
        pass

    decorator = m.compile_fn([SpecItem({"T": "int"}, "")])
    with pytest.raises(ValueError, match="only function definitions can be compiled"):
        decorator(NotAFunction)


def test_compile_fn_with_types_preserves_original():
    m = PythonModule()

    @m.compile_fn([SpecItem({"T": "int"}, "add_int"), SpecItem({"T": "float"}, "add_float")])
    def add[T: DType](a: T, b: T) -> T:
        return a + b

    assert callable(add)
    assert add(1, 2) == 3


def test_compile_fn_without_types_unchanged():
    m = PythonModule()

    @m.compile_fn(pybind=True)
    def add(a: int, b: int) -> int:
        return a + b

    assert len(m._funcs) == 1
    assert m._funcs[0].name == "add"


def test_compile_and_call_generic_int_float():
    m = PythonModule()

    @m.compile_fn([SpecItem({"T": "int"}, "add_int"), SpecItem({"T": "float"}, "add_float")], pybind=True)
    def add[T: DType](a: T, b: T) -> T:
        return a + b

    m.compile()
    assert m.add_int(3, 4) == 7
    assert abs(m.add_float(1.5, 2.5) - 4.0) < 1e-6


def test_compile_and_call_generic_void():
    m = PythonModule()
    import array

    @m.compile_fn([SpecItem({"T": "float"}, "fill_float")], pybind=True)
    def fill[T](buf: list[T], value: T, n: int) -> None:
        for i in range(n):
            buf[i] = value

    m.compile()
    arr = array.array("f", [0.0, 0.0, 0.0])
    m.fill_float(arr.buffer_info()[0], 42.0, 3)
    assert list(arr) == [42.0, 42.0, 42.0]


# element_wise module coverage


def test_element_wise_binary_compiled_float():
    import array

    from simplendarray.kernels.cpu.element_wise import element_wise_module

    arr_a = array.array("f", [1.0, 2.0, 3.0])
    arr_b = array.array("f", [4.0, 5.0, 6.0])
    arr_c = array.array("f", [0.0, 0.0, 0.0])
    element_wise_module.element_wise_binary_float__add(
        arr_a.buffer_info()[0],
        0,
        1,
        arr_b.buffer_info()[0],
        0,
        1,
        arr_c.buffer_info()[0],
        0,
        1,
        3,
    )
    assert list(arr_c) == [5.0, 7.0, 9.0]


def test_element_wise_binary_compiled_int():
    import array

    from simplendarray.kernels.cpu.element_wise import element_wise_module

    arr_a = array.array("i", [1, 2, 3])
    arr_b = array.array("i", [4, 5, 6])
    arr_c = array.array("i", [0, 0, 0])
    element_wise_module.element_wise_binary_int__add(
        arr_a.buffer_info()[0],
        0,
        1,
        arr_b.buffer_info()[0],
        0,
        1,
        arr_c.buffer_info()[0],
        0,
        1,
        3,
    )
    assert list(arr_c) == [5, 7, 9]


def test_compile_fn_with_multi_dict_types():
    m = PythonModule()

    @m.compile_fn([SpecItem({"A": "int", "B": "float"}, "f_int_float")], pybind=True)
    def f[A: DType, B: DType](x: A, y: B) -> A:
        return x

    names = {e.name for e in m._funcs}
    assert "f_int_float" in names
    assert len(m._funcs) == 1


def test_compile_and_call_multi_dict():
    m = PythonModule()

    @m.compile_fn(
        [
            SpecItem({"T1": "float", "T2": "float"}, "add_mixed_float_float"),
            SpecItem({"T1": "float", "T2": "double"}, "add_mixed_float_double"),
        ],
        pybind=True,
    )
    def add_mixed[T1: DType, T2: DType](a: list[T1], b: list[T2], c: list[T1], n: int) -> None:
        for i in range(n):
            c[i] = a[i] + b[i]  # pyrefly: ignore [unsupported-operation]

    m.compile()
    import array

    arr_a = array.array("f", [1.0, 2.0, 3.0])
    arr_b = array.array("f", [4.0, 5.0, 6.0])
    arr_c = array.array("f", [0.0, 0.0, 0.0])
    m.add_mixed_float_float(arr_a.buffer_info()[0], arr_b.buffer_info()[0], arr_c.buffer_info()[0], 3)
    assert list(arr_c) == [5.0, 7.0, 9.0]

    arr_d = array.array("d", [4.0, 5.0, 6.0])
    arr_c2 = array.array("f", [0.0, 0.0, 0.0])
    m.add_mixed_float_double(arr_a.buffer_info()[0], arr_d.buffer_info()[0], arr_c2.buffer_info()[0], 3)
    assert list(arr_c2) == [5.0, 7.0, 9.0]


def test_register_c_attrs():
    m = PythonModule()

    def add(a: int, b: int) -> int:
        return a + b

    m._register(add, c_attrs=["inline"], pybind=True)
    assert m._funcs[0].c_attrs == ["inline"]


def test_compile_fn_c_attrs():
    m = PythonModule()

    @m.compile_fn(c_attrs=["static", "inline"], pybind=True)
    def add(a: int, b: int) -> int:
        return a + b

    m.compile()
    assert m.add(3, 4) == 7


def test_compile_fn_c_attrs_generic():
    m = PythonModule()

    @m.compile_fn(
        [SpecItem({"T": "int"}, "add_int"), SpecItem({"T": "float"}, "add_float")],
        c_attrs=["static", "inline"],
        pybind=True,
    )
    def add[T: DType](a: T, b: T) -> T:
        return a + b

    m.compile()
    assert m.add_int(3, 4) == 7
    assert abs(m.add_float(1.5, 2.5) - 4.0) < 1e-6


def test_c_attrs_appears_in_c_source():
    m = PythonModule()

    @m.compile_fn([SpecItem({"T": "int"}, "add_int")], c_attrs=["static", "inline"], pybind=True)
    def add[T: DType](a: T, b: T) -> T:
        return a + b

    ext = m._generate_extension("_test_attrs")
    assert "static inline int add_int" in ext


def test_includes_in_extension():
    m = PythonModule(includes=["#include <stdio.h>", '#include "my_util.h"'])

    @m.compile_fn(pybind=True)
    def add(a: int, b: int) -> int:
        return a + b

    ext = m._generate_extension("_test_includes")
    assert "#include <stdio.h>" in ext
    assert '#include "my_util.h"' in ext


def test_includes_compile_and_call():
    m = PythonModule(includes=["#include <stddef.h>"])

    @m.compile_fn(pybind=True)
    def add(a: int, b: int) -> int:
        return a + b

    m.compile()
    assert m.add(3, 4) == 7


def test_useless_module():
    m = PythonModule()

    @m.compile_fn()
    def dummy():
        pass

    with pytest.raises(ValueError, match="Useless module, no python bindings present"):
        m.compile()


def test_generate_forward_decl_single():
    m = PythonModule()
    entry = _func_entry("add", [_param("a", "int"), _param("b", "int")], "int", False, INT_SRC)
    decl = m._generate_forward_decl(entry)
    assert decl == "int add(int, int);"


def test_generate_forward_decl_void():
    m = PythonModule()
    entry = _func_entry("noop", [], "void", False, "void noop() {}")
    decl = m._generate_forward_decl(entry)
    assert decl == "void noop();"


def test_generate_forward_decl_with_attrs():
    m = PythonModule()
    entry = _func_entry("add", [_param("a", "int"), _param("b", "int")], "int", False, INT_SRC)
    entry.c_attrs = ["inline"]
    decl = m._generate_forward_decl(entry)
    assert decl == "inline int add(int, int);"


def test_element_wise_binary_compiled_sub():
    import array

    from simplendarray.kernels.cpu.element_wise import element_wise_module

    arr_a = array.array("i", [10, 20, 30])
    arr_b = array.array("i", [1, 2, 3])
    arr_c = array.array("i", [0, 0, 0])
    element_wise_module.element_wise_binary_int__sub(
        arr_a.buffer_info()[0],
        0,
        1,
        arr_b.buffer_info()[0],
        0,
        1,
        arr_c.buffer_info()[0],
        0,
        1,
        3,
    )
    assert list(arr_c) == [9, 18, 27]


def test_element_wise_unary_compiled_square_float():
    import array

    from simplendarray.kernels.cpu.element_wise import element_wise_module

    arr_a = array.array("f", [1.0, 2.0, 3.0])
    arr_c = array.array("f", [0.0, 0.0, 0.0])
    element_wise_module.element_wise_unary_float__square(
        arr_a.buffer_info()[0],
        0,
        1,
        arr_c.buffer_info()[0],
        0,
        1,
        3,
    )
    assert list(arr_c) == [1.0, 4.0, 9.0]


def test_element_wise_unary_compiled_square_int():
    import array

    from simplendarray.kernels.cpu.element_wise import element_wise_module

    arr_a = array.array("i", [1, 2, 3])
    arr_c = array.array("i", [0, 0, 0])
    element_wise_module.element_wise_unary_int__square(
        arr_a.buffer_info()[0],
        0,
        1,
        arr_c.buffer_info()[0],
        0,
        1,
        3,
    )
    assert list(arr_c) == [1, 4, 9]


def test_element_wise_unary_compiled_relu_float():
    import array

    from simplendarray.kernels.cpu.element_wise import element_wise_module

    arr_a = array.array("f", [-1.0, 0.0, 3.0])
    arr_c = array.array("f", [0.0, 0.0, 0.0])
    element_wise_module.element_wise_unary_float__relu(
        arr_a.buffer_info()[0],
        0,
        1,
        arr_c.buffer_info()[0],
        0,
        1,
        3,
    )
    assert list(arr_c) == [0.0, 0.0, 3.0]


def test_element_wise_unary_compiled_exp_float():
    import array
    import math

    from simplendarray.kernels.cpu.element_wise import element_wise_module

    arr_a = array.array("d", [0.0, 1.0, 2.0])
    arr_c = array.array("d", [0.0, 0.0, 0.0])
    element_wise_module.element_wise_unary_double__exp(
        arr_a.buffer_info()[0],
        0,
        1,
        arr_c.buffer_info()[0],
        0,
        1,
        3,
    )
    expected = [math.exp(0.0), math.exp(1.0), math.exp(2.0)]
    assert all(abs(got - want) < 1e-5 for got, want in zip(arr_c, expected))


def test_element_wise_unary_compiled_log_float():
    import array
    import math

    from simplendarray.kernels.cpu.element_wise import element_wise_module

    arr_a = array.array("d", [1.0, math.e, math.e**2])
    arr_c = array.array("d", [0.0, 0.0, 0.0])
    element_wise_module.element_wise_unary_double__log(
        arr_a.buffer_info()[0],
        0,
        1,
        arr_c.buffer_info()[0],
        0,
        1,
        3,
    )
    expected = [0.0, 1.0, 2.0]
    assert all(abs(got - want) < 1e-5 for got, want in zip(arr_c, expected))


def test_element_wise_unary_compiled_sqrt_float():
    import array

    from simplendarray.kernels.cpu.element_wise import element_wise_module

    arr_a = array.array("f", [0.0, 4.0, 9.0])
    arr_c = array.array("f", [0.0, 0.0, 0.0])
    element_wise_module.element_wise_unary_float__sqrt(
        arr_a.buffer_info()[0],
        0,
        1,
        arr_c.buffer_info()[0],
        0,
        1,
        3,
    )
    expected = [0.0, 2.0, 3.0]
    assert all(abs(got - want) < 1e-5 for got, want in zip(arr_c, expected))


def test_element_wise_binary_non_contiguous():
    import array

    from simplendarray.kernels.cpu.element_wise import element_wise_module

    arr_a = array.array("i", [1, 2, 3, 4, 5, 6])
    arr_b = array.array("i", [10, 20, 30, 40, 50, 60])
    arr_c = array.array("i", [0, 0, 0, 0, 0, 0])
    element_wise_module.element_wise_binary_int__add(
        arr_a.buffer_info()[0],
        0,
        2,
        arr_b.buffer_info()[0],
        0,
        2,
        arr_c.buffer_info()[0],
        0,
        2,
        3,
    )
    assert list(arr_c) == [11, 0, 33, 0, 55, 0]


def test_element_wise_binary_neg_stride():
    import array

    from simplendarray.kernels.cpu.element_wise import element_wise_module

    arr_a = array.array("i", [1, 2, 3, 4, 5])
    arr_b = array.array("i", [10, 10, 10, 10, 10])
    arr_c = array.array("i", [0, 0, 0, 0, 0])
    element_wise_module.element_wise_binary_int__add(
        arr_a.buffer_info()[0],
        4,
        -1,
        arr_b.buffer_info()[0],
        0,
        1,
        arr_c.buffer_info()[0],
        0,
        1,
        5,
    )
    assert list(arr_c) == [15, 14, 13, 12, 11]


def test_element_wise_unary_non_contiguous():
    import array

    from simplendarray.kernels.cpu.element_wise import element_wise_module

    arr_a = array.array("i", [10, 20, 30, 40, 50, 60])
    arr_c = array.array("i", [0, 0, 0, 0, 0, 0])
    element_wise_module.element_wise_unary_int__square(
        arr_a.buffer_info()[0],
        1,
        2,
        arr_c.buffer_info()[0],
        1,
        2,
        3,
    )
    assert list(arr_c) == [0, 400, 0, 1600, 0, 3600]


def test_element_wise_unary_neg_stride():
    import array

    from simplendarray.kernels.cpu.element_wise import element_wise_module

    arr_a = array.array("i", [1, 2, 3, 4, 5])
    arr_c = array.array("i", [0, 0, 0, 0, 0])
    element_wise_module.element_wise_unary_int__square(
        arr_a.buffer_info()[0],
        4,
        -1,
        arr_c.buffer_info()[0],
        0,
        1,
        5,
    )
    assert list(arr_c) == [25, 16, 9, 4, 1]


@pytest.mark.skipif(not cuda_available, reason="CUDA not available")
def test_cuda_kernel_launch():
    m = PythonModule()

    @m.compile_fn(c_attrs=["__global__"])
    def dummy_kernel(val: list[int]):
        val[0] = 5

    @m.compile_fn(pybind=True)
    def call_kernel(val: list[int]):
        dummy_kernel[[[1, 1]]](val)

    buf = BufferCuda.from_iterable([1], dtype="i")
    m.compile("nvcc").call_kernel(buf.address)
    assert buf.copy_to_host()[0] == 5
