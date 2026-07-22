import ast
import inspect
from textwrap import dedent
from typing import Type

T = " " * 4


def ref[T](x: T) -> list[T]:
    """Equal to doing &x in C. Returning [x] is just for type hinting purposes."""
    return [x]


def sizeof[T](x: T) -> int: ...


def _c_type(node: ast.AST) -> str:
    match node:
        case ast.Name(id="void_ptr"):
            return "void*"
        case ast.Subscript(value=ast.Name(id="list"), slice=inner):
            # Pointer types
            return f"{_c_type(inner)}*"
        case ast.Name(id="bool"):
            return "int"
        case ast.Name(id="void") | ast.Constant(value=None):
            return "void"
        case ast.Name(id="char" | "i8"):
            return "char"
        case ast.Name(id="short" | "i16"):
            return "short"
        case ast.Name(id="int" | "int32" | "i32"):
            return "int"
        case ast.Name(id="int64" | "i64"):
            return "long long"
        case ast.Name(id="float" | "float32" | "f32"):
            return "float"
        case ast.Name(id="float64" | "double" | "f64"):
            return "double"
        case ast.Name(id="unsigned_char" | "u8"):
            return "unsigned char"
        case ast.Name(id="unsigned_short" | "u16"):
            return "unsigned short"
        case ast.Name(id="unsigned_int" | "u32"):
            return "unsigned int"
        case ast.Name(id="unsigned_long_long" | "u64"):
            return "unsigned long long"
        case ast.Name(id="str"):
            return "char*"
        case ast.Name(id="const_char_ptr"):
            return "const char*"
        case ast.Name(id=name):
            return name
        case _:
            raise ValueError(f"Unsupported type: {ast.dump(node)}")


_BINOP: dict[Type[ast.operator], str] = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.FloorDiv: "/",
    ast.Mod: "%",
    ast.BitAnd: "&",
    ast.BitOr: "|",
    ast.BitXor: "^",
    ast.LShift: "<<",
    ast.RShift: ">>",
}

_UNOP: dict[Type[ast.unaryop], str] = {
    ast.UAdd: "+",
    ast.USub: "-",
    ast.Not: "!",
    ast.Invert: "~",
}

_CMP: dict[Type[ast.cmpop], str] = {
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
    ast.Is: "==",
    ast.IsNot: "!=",
}


def _expr(node: ast.AST | None) -> str:
    match node:
        case ast.Constant(value=None) | None:
            return "(NULL)"
        case ast.Constant(value=bool(b)):
            return f"{int(b)}"
        case ast.Constant(value=int(i)):
            return str(i)
        case ast.Constant(value=str(v)):
            return f'"{v}"'

        case ast.Name(id=name):
            return name

        case ast.BinOp(left=l, op=o, right=r):
            c_op = _BINOP.get(type(o))
            if c_op is None:
                raise ValueError(f"unsupported binary op: {type(o).__name__}")
            return f"({_expr(l)} {c_op} {_expr(r)})"

        case ast.UnaryOp(op=o, operand=opnd):
            c_op = _UNOP.get(type(o))
            if c_op is None:
                raise ValueError(f"unsupported unary op: {type(o).__name__}")
            return f"({c_op}{_expr(opnd)})"

        case ast.Subscript(value=v, slice=ast.List(elts=[ast.List(elts=grid)])):
            # Turns kernel[[[3, 4, 5]]] -> kernel<<<3, 4, 5>>> for cuda
            return f"{_expr(v)}<<<{', '.join(map(_expr, grid))}>>>"

        case ast.Subscript(value=v, slice=s):
            return f"({_expr(v)}[{_expr(s)}])"

        case ast.Call(func=f, args=a):
            fn = _expr(f)
            if fn == "ref":
                return f"(&{_expr(a[0])})"
            return f"{fn}({', '.join(_expr(x) for x in a)})"

        case ast.Attribute(value=v, attr=a):
            return f"({_expr(v)}.{a})"

        case ast.Compare(left=l, ops=ops, comparators=comparators):
            parts = []
            left_expr = l
            for op, right_expr in zip(ops, comparators):
                c_op = _CMP.get(type(op))
                if c_op is None:
                    raise ValueError(f"unsupported comparison: {type(op).__name__}")
                parts.append(f"({_expr(left_expr)} {c_op} {_expr(right_expr)})")
                left_expr = right_expr
            if len(parts) == 1:
                return parts[0]
            return "(" + " && ".join(parts) + ")"

        case ast.BoolOp(op=o, values=vs):
            joiner = " && " if isinstance(o, ast.And) else " || "
            return "(" + joiner.join(_expr(v) for v in vs) + ")"

        case ast.List(elts=es):
            return "{" + ", ".join(_expr(e) for e in es) + "}"

        case ast.Tuple(elts=es):
            return ", ".join(_expr(e) for e in es)

        case ast.IfExp(test=t, body=b, orelse=o):
            return f"({_expr(t)} ? {_expr(b)} : {_expr(o)})"

        case _:
            raise ValueError(f"unsupported expression: {ast.dump(node)}")


_AUGOP: dict[Type[ast.operator], str] = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.Mod: "%",
    ast.BitAnd: "&",
    ast.BitOr: "|",
    ast.BitXor: "^",
    ast.LShift: "<<",
    ast.RShift: ">>",
}


def _stmt(node: ast.AST, indent: int = 0) -> str:
    i = T * indent

    match node:
        case ast.FunctionDef(name=name, args=args, body=body, returns=ret):
            if args.vararg or args.kwonlyargs or args.kwarg or args.defaults or args.kw_defaults:
                raise ValueError("keyword / variadic / default args not supported")

            params = []
            for a in args.args:
                match a.annotation:
                    case None | ast.Constant(value=None):
                        raise ValueError(f"parameter '{a.arg}' has no type annotation")
                    case ann:
                        params.append(f"{_c_type(ann)} {a.arg}")

            if ret is None:
                raise ValueError("function has no return type annotation")
            ret_type = _c_type(ret)
            c_body = "\n".join(_stmt(s, indent + 1) for s in body)
            sig = f"{ret_type} {name}({', '.join(params)})"
            if c_body.strip():
                return f"{sig} {{\n{c_body}\n{i}}}"
            return f"{sig} {{}}"

        case ast.Assign(targets=[t], value=v):
            return f"{i}{_expr(t)} = {_expr(v)};"

        case ast.AnnAssign(
            target=t,
            annotation=ast.Subscript(
                value=ast.Name(id="Annotated"),
                slice=ast.Tuple(elts=[ast.Subscript(value=ast.Name(id="list"), slice=ptr_type), *rest]),
            ),
            value=v,
        ):
            ct = _c_type(ptr_type)
            lhs = _expr(t)
            attrs = []
            size = ""
            for r in rest:
                if isinstance(r, ast.Constant) and isinstance(r.value, int):
                    size = str(r.value)
                else:
                    s = r.value if isinstance(r, ast.Constant) and isinstance(r.value, str) else _expr(r)
                    attrs.append(s)
            attr_prefix = " ".join(attrs) + " " if attrs else ""
            if "__shared__" in attrs:
                v = None
            if v is not None:
                return f"{i}{attr_prefix}{ct} {lhs}[{size}] = {_expr(v)};"
            return f"{i}{attr_prefix}{ct} {lhs}[{size}];"

        case ast.AnnAssign(
            target=t,
            annotation=ast.Subscript(
                value=ast.Name(id="Annotated"),
                slice=ast.Tuple(elts=[first_type, *rest]),
            ),
            value=v,
        ):
            ct = _c_type(first_type)
            lhs = _expr(t)
            attrs = []
            for r in rest:
                if isinstance(r, ast.Constant) and isinstance(r.value, str):
                    attrs.append(r.value)
                else:
                    attrs.append(_expr(r))
            attr_prefix = " ".join(attrs) + " " if attrs else ""
            if v is not None:
                return f"{i}{attr_prefix}{ct} {lhs} = {_expr(v)};"
            return f"{i}{attr_prefix}{ct} {lhs};"

        case ast.AnnAssign(target=t, annotation=a, value=v):
            ct = _c_type(a)
            lhs = _expr(t)
            if v is not None:
                return f"{i}{ct} {lhs} = {_expr(v)};"
            return f"{i}{ct} {lhs};"

        case ast.AugAssign(target=t, op=o, value=v):
            c_op = _AUGOP.get(type(o))
            if c_op is None:
                raise ValueError(f"unsupported aug-assign op: {type(o).__name__}")
            return f"{i}{_expr(t)} {c_op}= {_expr(v)};"

        case ast.For(target=t, iter=it, body=body, orelse=orelse):
            if orelse:
                raise ValueError("for-else not supported")
            if not (isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range"):
                raise ValueError("only range() supported as for-loop iter")
            n_args = len(it.args)
            if n_args == 1:
                start, end = "0", _expr(it.args[0])
                step = "1"
            elif n_args == 2:
                start, end = _expr(it.args[0]), _expr(it.args[1])
                step = "1"
            else:
                start, end = _expr(it.args[0]), _expr(it.args[1])
                step = _expr(it.args[2])
            loop = _expr(t)
            step_inc = f" += {step}" if step != "1" else "++"
            cmp = "<"
            if n_args == 3:
                try:
                    step_val = ast.literal_eval(it.args[2])
                    if isinstance(step_val, int) and step_val < 0:
                        cmp = ">"
                except ValueError, TypeError:
                    pass
            c_body = "\n".join(_stmt(s, indent + 1) for s in body)
            return f"{i}for (int {loop} = {start}; {loop} {cmp} {end}; {loop}{step_inc}) {{\n{c_body}\n{i}}}"

        case ast.While(test=t, body=body, orelse=orelse):
            if orelse:
                raise ValueError("while-else not supported")
            c_body = "\n".join(_stmt(s, indent + 1) for s in body)
            return f"{i}while ({_expr(t)}) {{\n{c_body}\n{i}}}"

        case ast.If(test=t, body=body, orelse=orelse):
            c_body = "\n".join(_stmt(s, indent + 1) for s in body)
            out = f"{i}if ({_expr(t)}) {{\n{c_body}\n{i}}}"
            if orelse:
                out += _else_or_elif(orelse, indent)
            return out

        case ast.Return(value=v):
            return f"{i}return {_expr(v)};"

        case ast.Break():
            return f"{i}break;"

        case ast.Continue():
            return f"{i}continue;"

        case ast.Expr(value=v):
            return f"{i}{_expr(v)};"

        case ast.Pass():
            return ""

        case _:
            raise ValueError(f"unsupported statement: {ast.dump(node)}")


def _else_or_elif(orelse: list[ast.stmt], indent: int) -> str:
    """Format else / elif continuation cleanly."""
    i = T * indent

    if len(orelse) == 1 and isinstance(orelse[0], ast.If):
        elif_node = orelse[0]
        el_body = "\n".join(_stmt(s, indent + 1) for s in elif_node.body)
        out = f" else if ({_expr(elif_node.test)}) {{\n{el_body}\n{i}}}"
        if elif_node.orelse:
            out += _else_or_elif(elif_node.orelse, indent)
        return out

    c_else = "\n".join(_stmt(s, indent + 1) for s in orelse)
    return f" else {{\n{c_else}\n{i}}}"


def compile_str(func) -> str:
    """Transpile a python function into a C source code string."""
    src = dedent(inspect.getsource(func))
    tree = ast.parse(src)
    fn = tree.body[0]
    if not isinstance(fn, ast.FunctionDef):
        raise ValueError("only function definitions can be compiled")
    c_str = _stmt(fn)
    return c_str
