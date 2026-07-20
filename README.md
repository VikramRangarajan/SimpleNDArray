# SimpleNDArray
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

An NDArray python library. Supports CPU/CUDA operations. Mostly educational, for my own learning.

# Installation

```bash
uv add git+https://github.com/VikramRangarajan/SimpleNDArray.git
```

Requires Python >= 3.14.

# Development / Testing

Clone the repository. Run `uv sync`. Run `uv run prek install` to install the pre-commit hooks. The pre-commit hook runs `ruff format`, `ruff check --fix`, and `pyrefly check` for code quality. You can use `uv run prek run --all-files` to run all pre-commit hooks on all files.

To test, run `uv run pytest --cov --cov-report=term-missing -n auto`.

# AI Usage
AI was used only to generate tests, and do other menial tasks that came with the development of this library. However, the core was done by hand.

# TODO
- Unify Buffer and BufferCuda
- Figure out proper memory management for the above 2
- Pivot away from array.array for Buffer? Can get rid of typecode shenanigans
- Add array broadcasting
- Transpiler: Add macro support for forced inlining
- Implement more proper testing suite?
- Benchmarking vs. numpy, pytorch/cublas, etc.
- Start with reduction ops, then go to linear ops (matmul, conv), then flash attention, moe, etc.
- Add Cuda Graphs
- Add documentation
- Better dispatch implementation (more robust)?