import ast
import functools
import inspect
import os
import sqlite3
import tempfile
from pathlib import Path
from subprocess import run
from textwrap import dedent
from typing import Callable


def benchmark(env: dict[str, str] | None = None):
    def _fn(f: Callable[[], None]):
        return _benchmark(f, env=env)

    return _fn


def _benchmark(fn: Callable[[], None], *, env: dict[str, str] | None) -> Callable[[], list[dict]]:
    source = dedent(inspect.getsource(fn))
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fn.__name__:
            lines = source.splitlines()
            source = "\n".join(lines[node.lineno - 1 : node.end_lineno])
            break

    proc_env = {**os.environ, **env} if env else None

    @functools.wraps(fn)
    def wrapper() -> list[dict]:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(source)
            f.write(f"\n{fn.__name__}()\n")
            tmp = Path(f.name)

        report = tmp.with_suffix(".nsys-rep")
        sqlite = tmp.with_suffix(".sqlite")
        report.unlink(missing_ok=True)
        sqlite.unlink(missing_ok=True)

        out = run(["uv", "run", str(tmp)], capture_output=True, env=proc_env)  # noqa: S603, S607
        if len(out.stderr) > 0:
            print(out.stderr.decode())
        run(  # noqa: S603
            [  # noqa: S607
                "nsys",
                "profile",
                "--force-overwrite",
                "true",
                "-o",
                str(report),
                "uv",
                "run",
                str(tmp),
            ],
            check=True,
            capture_output=True,
            env=proc_env,
        )
        out = run(  # noqa: S603
            ["nsys", "stats", "--force-export=true", "--report", "cuda_gpu_trace", str(report)],  # noqa: S607
            check=True,
            capture_output=True,
            env=proc_env,
        )
        if len(out.stderr) > 0:
            print(out.stderr.decode())
        with sqlite3.connect(sqlite) as con:
            rows = con.execute("""
                SELECT
                    s.value AS name,
                    k.start,
                    k.end - k.start AS duration_ns,
                    k.gridX, k.gridY, k.gridZ,
                    k.blockX, k.blockY, k.blockZ,
                    k.registersPerThread
                FROM CUPTI_ACTIVITY_KIND_KERNEL k
                JOIN StringIds s ON s.id = k.demangledName
                ORDER BY k.start
            """).fetchall()

        tmp.unlink(missing_ok=True)
        report.unlink(missing_ok=True)
        sqlite.unlink(missing_ok=True)

        return [
            {
                "name": r[0],
                "start_ns": r[1],
                "duration_ns": r[2],
                "duration_us": r[2] / 1000.0,
                "grid": (r[3], r[4], r[5]),
                "block": (r[6], r[7], r[8]),
                "registers_per_thread": r[9],
            }
            for r in rows
        ]

    return wrapper


def mem_bandwidth():
    from cuda.bindings import runtime

    err, device = runtime.cudaGetDevice()
    err, props = runtime.cudaGetDeviceProperties(device)
    name = props.name.decode()
    return {
        "NVIDIA A40": 696 * 10**9,
        "Tesla V100-PCIE-32GB": 900 * 10**9,
    }[name]


def fp32_flops():
    from cuda.bindings import runtime

    err, device = runtime.cudaGetDevice()
    err, props = runtime.cudaGetDeviceProperties(device)
    name = props.name.decode()
    return {
        "NVIDIA A40": 37.4 * 10**12,
    }[name]
