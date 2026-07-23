import ast
import copy
import fcntl
import hashlib
import importlib.util
import inspect
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Callable, Iterable, Mapping, Protocol, Self, cast

from .transpiler import T, _c_type, _stmt


@dataclass
class SpecItem:
    mapping: Mapping[str, str]
    fn_name: str


class DType(Protocol):
    def __add__(self, other: Self) -> Self: ...
    def __sub__(self, other: Self) -> Self: ...
    def __mul__(self, other: Self) -> Self: ...
    def __truediv__(self, other: Self) -> Self: ...
    def __lt__(self, other: Self) -> bool: ...
    def __gt__(self, other: Self) -> bool: ...


_POINTER_TYPES = {"int*", "long long*", "float*", "double*", "char*", "unsigned char*"}

_C_PARSE_FMT = {
    "unsigned char": "b",
    "char": "c",
    "unsigned short": "H",
    "short": "h",
    "unsigned int": "I",
    "int": "i",
    "unsigned long long": "K",
    "long long": "L",
    "float": "f",
    "double": "d",
    "unsigned char*": "s",
    "char*": "s",
    "unsigned short*": "K",
    "short*": "K",
    "unsigned int*": "K",
    "int*": "K",
    "unsigned long*": "K",
    "long*": "K",
    "unsigned long long*": "K",
    "long long*": "K",
    "float*": "K",
    "double*": "K",
    "void*": "K",
}

_C_PARSE_TYPE = {
    "int": "int",
    "long long": "long long",
    "unsigned long long": "unsigned long long",
    "float": "float",
    "double": "double",
    "char": "char",
    "char*": "char*",
    "int*": "unsigned long long",
    "long long*": "unsigned long long",
    "float*": "unsigned long long",
    "double*": "unsigned long long",
}

_C_RETURN_BUILD = {
    "int": "PyLong_FromLong",
    "long long": "PyLong_FromLongLong",
    "unsigned long long": "PyLong_FromUnsignedLongLong",
    "float": "PyFloat_FromDouble",
    "double": "PyFloat_FromDouble",
    "char": "PyLong_FromLong",
    "char*": "PyUnicode_FromString",
    "void*": "PyLong_FromVoidPtr",
}

_C_TYPE_TO_PY: dict[str, str] = {
    "int": "int",
    "long long": "int",
    "unsigned long long": "int",
    "float": "float",
    "double": "float",
    "char": "int",
    "char*": "int",
    "void": "None",
    "void*": "int",
    "int*": "int",
    "long long*": "int",
    "float*": "int",
    "double*": "int",
}

_EXTENSION_TEMPLATE = """
#define PY_SSIZE_T_CLEAN
#include <Python.h>
{includes}

/* Forward declarations */
{forward_decls}

/* Transpiled C functions */
{c_funcs}

/* Python wrapper functions */
{wrapper}

/* Method table */
static PyMethodDef {module_name}_methods[] = {{
{method_defs},
    {{NULL, NULL, 0, NULL}}
}};

static struct PyModuleDef {module_name}_module = {{
    PyModuleDef_HEAD_INIT,
    "{module_name}",
    NULL,
    -1,
    {module_name}_methods
}};

PyMODINIT_FUNC PyInit_{module_name}(void) {{
    return PyModule_Create(&{module_name}_module);
}}
"""

_WRAPPER_TEMPLATE = """
static PyObject* {name}_wrapper(PyObject* self, PyObject* args) {{
{var_block}
    {parse_call}
        return NULL;
    }}
{call_line}
{ret_line}
}}
"""


@dataclass
class CFunction:
    name: str
    c_source: str
    params: list
    ret_c_type: str
    ret_is_bool: bool
    c_attrs: list[str]
    pybind: bool
    group: str | None = None
    dispatch_key: str | None = None


class PythonModule:
    def __init__(
        self,
        includes: list[str] | None = None,
        stub_path: str | None = None,
        stub_var: str | None = None,
        module_name: str = "_ndarray_rt_module",
    ):
        self._funcs: list[CFunction] = []
        self._includes = includes or []
        self._module = None
        self._compiled = False
        self._stub_path = stub_path
        self._stub_var = stub_var
        self._module_name = module_name

    def _register(self, func, c_attrs: list[str], pybind: bool, group: str | None = None):
        src = dedent(inspect.getsource(func))
        tree = ast.parse(src)
        fn = tree.body[0]
        if not isinstance(fn, ast.FunctionDef):
            raise ValueError("only function definitions can be compiled")
        name = fn.name

        if fn.returns is None:
            fn.returns = ast.Name("void")

        self._register_ast(name, fn, c_attrs, pybind, group=group)

    def _register_ast(
        self,
        name: str,
        fn: ast.FunctionDef,
        c_attrs: list[str],
        pybind: bool,
        group: str | None = None,
        dispatch_key=None,
    ):
        c_source = _stmt(fn)

        params = []
        for a in fn.args.args:
            annotation = cast(ast.AST, a.annotation)
            is_bool = isinstance(annotation, ast.Name) and annotation.id == "bool"
            is_string = isinstance(annotation, ast.Name) and annotation.id == "str"
            ct = _c_type(annotation)
            params.append({"name": a.arg, "c_type": ct, "is_bool": is_bool, "is_string": is_string})

        ret = cast(ast.AST, fn.returns)
        ret_is_bool = isinstance(ret, ast.Name) and ret.id == "bool"
        ret_c_type = _c_type(ret)

        entry = CFunction(
            name=name,
            c_source=c_source,
            params=params,
            ret_c_type=ret_c_type,
            ret_is_bool=ret_is_bool,
            c_attrs=c_attrs or [],
            pybind=pybind,
            group=group,
            dispatch_key=dispatch_key,
        )
        self._funcs.append(entry)

    _cache_dir_override: Path | None = None

    @staticmethod
    def _cache_dir() -> Path:
        if PythonModule._cache_dir_override is not None:
            return PythonModule._cache_dir_override
        env = os.environ.get("SNDA_CACHE_HOME")
        if env:
            base = Path(env)
        else:
            base = Path.home() / ".cache" / "simplendarray"
        base.mkdir(parents=True, exist_ok=True)
        return base

    @contextmanager
    def _cache_lock(self, cache_dir: Path):
        lock_path = cache_dir / ".lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)

    def _cache_key(self, ext_src: str, compiler: str, cflags: list[str] | None, ldflags: list[str] | None) -> str:
        meta = json.dumps({"compiler": compiler, "cflags": cflags, "ldflags": ldflags}, sort_keys=True)
        raw = ext_src + meta
        return hashlib.sha256(raw.encode()).hexdigest()

    def _load_from_so(self, module_name: str, so_path: Path) -> bool:
        spec = importlib.util.spec_from_file_location(module_name, so_path)
        if spec is None or spec.loader is None:
            return False
        self._module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self._module)
        self._compiled = True
        return True

    def compile(self, compiler: str = "gcc", cflags: list[str] | None = None, ldflags: list[str] | None = None) -> Self:
        if not self._funcs:
            raise ValueError("no functions registered")
        if self._compiled:
            raise RuntimeError("module already compiled")

        self._write_source_stub()

        if compiler == "nvcc" and shutil.which("nvcc") is None:
            return self

        module_name = self._module_name
        ext_src = self._generate_extension(module_name)
        ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
        src_ext = ".cu" if compiler == "nvcc" else ".c"
        cache_dir = self._cache_dir()
        key = self._cache_key(ext_src, compiler, cflags, ldflags)
        build_dir = cache_dir / f"{module_name}_{key}"
        build_dir.mkdir(parents=True, exist_ok=True)
        so_path = build_dir / f"{module_name}_{key}{ext_suffix}"

        if so_path.exists() and self._load_from_so(module_name, so_path):
            self._build_dispatch_dicts()
            return self

        with self._cache_lock(build_dir):
            src_path = build_dir / f"{module_name}_{key}{src_ext}"
            src_path.write_text(ext_src)

            meta = {"compiler": compiler, "cflags": cflags, "ldflags": ldflags}
            meta_path = build_dir / f"{module_name}_{key}.json"
            meta_path.write_text(json.dumps(meta, indent=2))

            py_include = sysconfig.get_config_var("INCLUDEPY")
            if not py_include:
                py_include = sysconfig.get_path("include")

            cmd = [compiler, "-shared"]
            if compiler == "nvcc":  # pragma: no cover
                cmd.extend(["-Xcompiler", "-fPIC"])  # pragma: no cover
            else:
                cmd.append("-fPIC")
            cmd.append(f"-I{py_include}")
            if cflags:
                cmd.extend(cflags)
            cmd.extend(["-o", str(so_path), str(src_path)])
            if ldflags:
                cmd.extend(ldflags)
            if sys.platform == "darwin":
                cmd.append("-undefined")  # pragma: no cover
                cmd.append("dynamic_lookup")  # pragma: no cover

            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)  # noqa: S603
            if result.returncode != 0:
                msg = result.stderr.decode()
                raise RuntimeError(f"compilation failed:\n{msg}")

            if not self._load_from_so(module_name, so_path):
                raise RuntimeError(f"failed to create module spec for {so_path}")
            self._build_dispatch_dicts()

        return self

    def _generate_forward_decl(self, func: CFunction) -> str:
        ret = func.ret_c_type
        name = func.name
        param_strs = [p["c_type"] for p in func.params]
        sig = f"{ret} {name}({', '.join(param_strs)})"
        attrs = func.c_attrs
        if attrs:
            sig = f"{' '.join(attrs)} {sig}"
        return f"{sig};"

    def _generate_extension(self, module_name):
        c_functions = []
        wrapper_functions = []
        method_defs = []
        forward_decls = []

        for func in self._funcs:
            lines = func.c_source.split("\n", 1)
            lines[0] = f"{' '.join(func.c_attrs)} {lines[0]}"
            c_source = "\n".join(lines)
            c_functions.append(c_source)
            if func.pybind:
                wrapper_functions.append(self._generate_wrapper(func))
                method_defs.append(
                    f'    {{"{func.name}", {func.name}_wrapper, METH_VARARGS, "Transpiled function {func.name}"}}'
                )
            forward_decls.append(self._generate_forward_decl(func))

        if not method_defs:
            raise ValueError("Useless module, no python bindings present", module_name)

        c_funcs_str = "\n\n".join(c_functions)
        wrapper_str = "\n\n".join(wrapper_functions)
        method_defs_str = ",\n".join(method_defs)
        forward_decls_str = "\n".join(forward_decls)
        includes_str = "\n".join(self._includes)

        return _EXTENSION_TEMPLATE.format(
            includes=includes_str,
            forward_decls=forward_decls_str,
            c_funcs=c_funcs_str,
            wrapper=wrapper_str,
            method_defs=method_defs_str,
            module_name=module_name,
        )

    def _generate_wrapper(self, func: CFunction):
        name = func.name
        params = func.params
        ret_c_type = func.ret_c_type
        ret_is_bool = func.ret_is_bool

        fmt_chars = []
        for p in params:
            if p["is_bool"]:
                fmt_chars.append("p")
            elif p.get("is_string") and p["c_type"] in ("char*", "unsigned char*"):
                fmt_chars.append("s")
            elif p["c_type"] in ("char*", "unsigned char*"):
                fmt_chars.append("K")
            else:
                f = _C_PARSE_FMT.get(p["c_type"])
                if f is None:
                    raise ValueError(f"unsupported param type for Python wrapper: {p['c_type']}")
                fmt_chars.append(f)

        fmt = "".join(fmt_chars)

        var_lines = []
        parse_args = []
        for p in params:
            if p["is_bool"]:
                vt = "int"
            elif p.get("is_string") and p["c_type"] in ("char*", "unsigned char*"):
                vt = "char*"
            elif p["c_type"] in ("char*", "unsigned char*"):
                vt = "unsigned long long"
            else:
                vt = _C_PARSE_TYPE.get(p["c_type"], p["c_type"])
            var_lines.append(f"{T}{vt} {p['name']};")
            parse_args.append(f"&{p['name']}")

        call_args = []
        for p in params:
            if p.get("is_string") and p["c_type"] in ("char*", "unsigned char*"):
                call_args.append(p["name"])
            elif p["c_type"] in _POINTER_TYPES:
                call_args.append(f"({p['c_type']}){p['name']}")
            else:
                call_args.append(p["name"])
        arg_str = ", ".join(call_args)

        if ret_c_type == "void":
            call_line = f"{T}{name}({arg_str});"
            ret_line = f"{T}Py_RETURN_NONE;"
        else:
            decl_ret_type = "int" if ret_is_bool else ret_c_type
            call_line = f"{T}{decl_ret_type} result = {name}({arg_str});"
            if ret_is_bool:
                ret_line = f"{T}return PyBool_FromLong(result);"
            else:
                builder = _C_RETURN_BUILD.get(ret_c_type)
                if builder is None:
                    raise ValueError(f"unsupported return type for Python wrapper: {ret_c_type}")
                ret_line = f"{T}return {builder}(result);"

        var_block = "\n".join(var_lines) if var_lines else f"{T}(void)self;"

        if parse_args:
            parse_call = f'if (!PyArg_ParseTuple(args, "{fmt}", {", ".join(parse_args)})) {{'
        else:
            parse_call = 'if (!PyArg_ParseTuple(args, "")) {'

        return _WRAPPER_TEMPLATE.format(
            name=name,
            var_block=var_block,
            parse_call=parse_call,
            call_line=call_line,
            ret_line=ret_line,
        )

    def _write_source_stub(self):
        if self._stub_path is not None and self._stub_var is not None:
            stem = Path(self._stub_path).stem
            parent = Path(self._stub_path).parent
            stub_file = parent / f"_{stem}_stubs.py"
            class_name = f"_{self._stub_var.title().replace('_', '')}Class"

            has_dispatch = any(
                func.pybind and func.group is not None and func.dispatch_key is not None for func in self._funcs
            )

            lines = [
                "# Auto-generated stub for compiled functions.",
                "from __future__ import annotations",
                "from typing import TYPE_CHECKING",
            ]
            if has_dispatch:
                lines.append("from typing import Callable, ClassVar")
            lines.append("")
            lines.append("from simplendarray.transpiler.runtime import PythonModule")
            lines.append("")
            lines.append(f"class {class_name}(PythonModule):")
            lines.append(f"{T}if TYPE_CHECKING:")

            seen_groups: set[str] = set()
            for func in self._funcs:
                if (
                    func.pybind
                    and func.group is not None
                    and func.dispatch_key is not None
                    and func.group not in seen_groups
                ):
                    seen_groups.add(func.group)
                    lines.append(
                        f"{T * 2}DISPATCH_DICT_{func.group}: ClassVar[dict[str, Callable[..., None]]]"  # noqa: E501
                    )
            if seen_groups:
                lines.append("")

            for func in self._funcs:
                if func.pybind:
                    params = ", ".join(f"{p['name']}: {_C_TYPE_TO_PY.get(p['c_type'], 'int')}" for p in func.params)
                    return_type = _C_TYPE_TO_PY.get(func.ret_c_type, "None")
                    lines.append(f"{T * 2}def {func.name}(self, {params}) -> {return_type}: ...")
            lines.append(f"{T * 2}pass")
            lines.append("")
            content = "\n".join(lines)
            fd, tmp = tempfile.mkstemp(dir=parent, suffix=".py")
            with os.fdopen(fd, "w") as f:
                f.write(content)
            os.replace(tmp, stub_file)

    def _build_dispatch_dicts(self):
        groups: dict[str, dict[str, str]] = {}
        for func in self._funcs:
            if func.pybind and func.group is not None and func.dispatch_key is not None:
                groups.setdefault(func.group, {})[func.dispatch_key] = func.name
        for group_name, mapping in groups.items():
            dict_name = f"DISPATCH_DICT_{group_name}"
            dispatch_dict = {key: getattr(self, mangled_name) for key, mangled_name in mapping.items()}
            setattr(self, dict_name, dispatch_dict)

    def __getattr__(self, name):
        if self._compiled and self._module is not None and hasattr(self._module, name):
            return getattr(self._module, name)
        raise AttributeError(f"Module has no attribute '{name}'")

    def compile_fn(
        self, types: Iterable[SpecItem] | None = None, c_attrs: list[str] | None = None, pybind: bool = False
    ):
        if c_attrs is None:
            c_attrs = []

        class IndexableFunction[**P, R]:  # pragma: no cover
            # Purely for type hinting
            def __init__(self, fn: Callable[P, R]):
                self.fn = fn

            def __call__(self, *args: P.args, **kwargs: P.kwargs):
                return self.fn(*args, **kwargs)

            def __getitem__(self, _idx):
                return self

        def decorator[R, **P](func: Callable[P, R]) -> IndexableFunction[P, R]:
            if types:
                src = dedent(inspect.getsource(func))
                tree = ast.parse(src)
                fn = tree.body[0]
                if not isinstance(fn, ast.FunctionDef):
                    raise ValueError("only function definitions can be compiled")
                for spec in types:
                    fn_name = spec.fn_name
                    fn_copy = copy.deepcopy(fn)
                    substituter = _TypeSubstituter(spec)
                    specialized = substituter.visit(fn_copy)
                    ast.fix_missing_locations(specialized)
                    group = func.__name__
                    dispatch_key = fn_name
                    self._register_ast(fn_name, specialized, c_attrs, pybind, group=group, dispatch_key=dispatch_key)
            else:
                group = func.__name__
                self._register(func, c_attrs, pybind, group=group)
            return IndexableFunction(func)

        return decorator


class _TypeSubstituter(ast.NodeTransformer):
    def __init__(self, spec: SpecItem):
        self.mapping = {k: (v if isinstance(v, str) else v.ctype) for k, v in spec.mapping.items()}
        self.func_name = spec.fn_name

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self.mapping:
            return ast.Name(id=self.mapping[node.id], lineno=node.lineno, col_offset=node.col_offset)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        visited = cast(ast.FunctionDef, self.generic_visit(node))
        visited.name = self.func_name
        visited.type_params = []
        return visited
