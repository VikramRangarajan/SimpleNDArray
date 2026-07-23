from __future__ import annotations

from simplendarray.dtypes import NULL, const_char_ptr, u64, void_ptr
from simplendarray.transpiler import ref

from ._buffer_cuda_stubs import _BufferCudaModuleClass as _BufferCudaModule
from .helpers import cudaError_t, cudaMemcpyDeviceToHost, cudaMemcpyHostToDevice, cudaSuccess

void = None

buffer_cuda_module = _BufferCudaModule(
    includes=["#include <cuda_runtime.h>", "#include <stdio.h>", "#include <stdlib.h>"],
    stub_path=__file__,
    stub_var="buffer_cuda_module",
    module_name="buffer_cuda_module",
)

fprintf = print
stderr = 1


def cudaMalloc(ptr: list[void_ptr], size: u64) -> cudaError_t: ...
def cudaFree(ptr: void_ptr) -> cudaError_t: ...
def cudaMemcpy(dst: void_ptr, src: void_ptr, size: u64, mode: int) -> cudaError_t: ...
def cudaGetErrorName(error: cudaError_t) -> const_char_ptr: ...
def cudaGetErrorString(error: cudaError_t) -> const_char_ptr: ...


@buffer_cuda_module.compile_fn()
def snda_cuda_malloc(size: u64) -> void_ptr:
    p: void_ptr = NULL
    status: cudaError_t = cudaMalloc(ref(p), size)
    cuda_check(status)
    return p


@buffer_cuda_module.compile_fn()
def snda_cuda_free(ptr: void_ptr) -> void:
    status: cudaError_t = cudaFree(ptr)
    cuda_check(status)


@buffer_cuda_module.compile_fn()
def snda_cuda_memcpy_h2d(dst: void_ptr, src: void_ptr, size: u64):
    status: cudaError_t = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice)
    cuda_check(status)


@buffer_cuda_module.compile_fn()
def snda_cuda_memcpy_d2h(dst: void_ptr, src: void_ptr, size: u64):
    status: cudaError_t = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost)
    cuda_check(status)


@buffer_cuda_module.compile_fn()
def cuda_check(status: cudaError_t):
    if status != cudaSuccess:
        err_name: const_char_ptr = cudaGetErrorName(status)
        err_str: const_char_ptr = cudaGetErrorString(status)
        fprintf(stderr, "CUDA memcpy D2H error: %s (%s)\\n", err_str, err_name)
        exit(status)


@buffer_cuda_module.compile_fn(pybind=True)
def cuda_malloc(size: u64) -> void_ptr:
    return snda_cuda_malloc(size)


@buffer_cuda_module.compile_fn(pybind=True)
def cuda_free(ptr: void_ptr):
    snda_cuda_free(ptr)


@buffer_cuda_module.compile_fn(pybind=True)
def cuda_memcpy_h2d(dst: void_ptr, src: void_ptr, size: u64):
    snda_cuda_memcpy_h2d(dst, src, size)


@buffer_cuda_module.compile_fn(pybind=True)
def cuda_memcpy_d2h(dst: void_ptr, src: void_ptr, size: u64):
    snda_cuda_memcpy_d2h(dst, src, size)


buffer_cuda_module = buffer_cuda_module.compile(compiler="nvcc", ldflags=["-lcudart"])
