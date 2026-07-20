from __future__ import annotations

import array
from typing import TYPE_CHECKING, Iterable, Protocol

from .kernels.cuda import buffer_cuda_module


class HasBufferInfo(Protocol):
    def buffer_info(self) -> tuple[int, int]: ...


class HasDataAndAddress(Protocol):
    data: array.array
    address: int


if TYPE_CHECKING:
    from .dtypes import Device


class BufferCuda:
    def __init__(self, size: int, dtype: str):
        self.dtype = dtype
        self.typecode = dtype
        self.size = size
        self.num_bytes = size * array.array(dtype).itemsize
        self.device: Device = "gpu"

        # Allocate device memory
        ptr = buffer_cuda_module.cuda_malloc(self.num_bytes)
        self.address = ptr
        self._owns_memory = True

    @classmethod
    def empty(cls, size: int, dtype: str) -> BufferCuda:
        return cls(size, dtype)

    @classmethod
    def from_iterable(cls, data: array.array | Iterable[int | float | bool], dtype: str) -> BufferCuda:
        if isinstance(data, array.array):
            cpu_buf = data
        else:
            cpu_buf = array.array(dtype, data)
        gpu_buf = cls(len(cpu_buf), dtype)
        gpu_buf.copy_from_host(cpu_buf)
        return gpu_buf

    def copy_from_host(self, cpu_buffer: array.array | HasDataAndAddress) -> None:
        """Copy data from a CPU buffer or array.array to this GPU buffer."""
        if isinstance(cpu_buffer, array.array):
            src_data = cpu_buffer
            src_addr = cpu_buffer.buffer_info()[0]
        else:
            src_data = cpu_buffer.data
            src_addr = cpu_buffer.address
        src_bytes = len(src_data) * src_data.itemsize
        if src_bytes != self.num_bytes:
            raise ValueError("Buffer size mismatch")
        buffer_cuda_module.cuda_memcpy_h2d(self.address, src_addr, self.num_bytes)

    def copy_to_host(self) -> array.array:
        """Copy data from this GPU buffer to a new CPU array.array."""
        cpu_data = array.array(self.typecode, [0]) * self.size
        buffer_cuda_module.cuda_memcpy_d2h(cpu_data.buffer_info()[0], self.address, self.num_bytes)
        return cpu_data

    def __repr__(self) -> str:
        cpu_data = self.copy_to_host()
        return f"BufferCuda({repr(cpu_data)})"

    def __del__(self):  # pragma: no cover
        owns = getattr(self, "_owns_memory", False)
        if owns and hasattr(self, "address") and self.address:
            buffer_cuda_module.cuda_free(self.address)
            self._owns_memory = False

    @property
    def data(self) -> array.array:
        """Return a CPU copy of the GPU data for compatibility."""
        return self.copy_to_host()  # pragma: no cover
