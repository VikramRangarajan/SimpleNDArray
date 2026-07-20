import array
import shutil

import pytest

from simplendarray.buffer import Buffer
from simplendarray.buffer_cuda import BufferCuda

cuda_available = shutil.which("nvcc") is not None


def test_buffer_init_cpu():
    a = array.array("i", [1, 2, 3])
    b = Buffer(a)
    assert b.data is a
    assert b.address == a.buffer_info()[0]
    assert b.num_bytes == a.buffer_info()[1] * a.itemsize
    assert b.device == "cpu"


def test_buffer_from_iterable_int():
    b = Buffer.from_iterable([10, 20, 30], "i")
    assert isinstance(b, Buffer)
    assert not isinstance(b, BufferCuda)
    assert isinstance(b.data, array.array)
    assert b.data.typecode == "i"
    assert list(b.data) == [10, 20, 30]
    assert b.device == "cpu"


def test_buffer_from_iterable_float():
    b = Buffer.from_iterable([1.5, 2.5], "d")
    assert isinstance(b, Buffer)
    assert not isinstance(b, BufferCuda)
    assert b.data.typecode == "d"
    assert list(b.data) == [1.5, 2.5]


def test_buffer_from_iterable_bool():
    b = Buffer.from_iterable([True, False, True], "b")
    assert isinstance(b, Buffer)
    assert not isinstance(b, BufferCuda)
    assert b.data.typecode == "b"
    assert list(b.data) == [True, False, True]


def test_buffer_from_iterable_empty():
    b = Buffer.from_iterable([], "i")
    assert isinstance(b, Buffer)
    assert not isinstance(b, BufferCuda)
    assert list(b.data) == []


def test_buffer_from_iterable_generator():
    b = Buffer.from_iterable(range(3), "i")
    assert isinstance(b, Buffer)
    assert not isinstance(b, BufferCuda)
    assert list(b.data) == [0, 1, 2]


def test_buffer_repr_cpu():
    b = Buffer(array.array("i", [1, 2, 3]))
    assert repr(b) == "array('i', [1, 2, 3])"


@pytest.mark.skipif(not cuda_available, reason="CUDA not available")
def test_buffer_empty_gpu():
    b = BufferCuda.empty(5, "i")
    assert isinstance(b, BufferCuda)
    assert b.size == 5
    assert b.device == "gpu"
    assert b.num_bytes == 5 * 4  # 5 ints * 4 bytes


@pytest.mark.skipif(not cuda_available, reason="CUDA not available")
def test_buffer_from_iterable_gpu():
    b = BufferCuda.from_iterable(array.array("i", [10, 20, 30]), "i")
    assert isinstance(b, BufferCuda)
    assert b.size == 3
    assert list(b.copy_to_host()) == [10, 20, 30]


@pytest.mark.skipif(not cuda_available, reason="CUDA not available")
def test_buffer_copy_to_host():
    b = BufferCuda.from_iterable([1.5, 2.5, 3.5], "d")
    assert isinstance(b, BufferCuda)
    cpu_data = b.copy_to_host()
    assert list(cpu_data) == [1.5, 2.5, 3.5]
    assert isinstance(cpu_data, array.array)
    assert cpu_data.typecode == "d"


@pytest.mark.skipif(not cuda_available, reason="CUDA not available")
def test_buffer_repr_gpu_cuda():
    b = BufferCuda.from_iterable([1, 2, 3], "i")
    assert isinstance(b, BufferCuda)
    r = repr(b)
    assert "BufferCuda" in r
    assert "[1, 2, 3]" in r


@pytest.mark.skipif(not cuda_available, reason="CUDA not available")
def test_buffer_cuda_from_iterable_with_list():
    """Test from_iterable with a list (not array.array)."""
    b = BufferCuda.from_iterable([1, 2, 3], "i")
    assert isinstance(b, BufferCuda)
    assert b.size == 3
    assert list(b.copy_to_host()) == [1, 2, 3]


@pytest.mark.skipif(not cuda_available, reason="CUDA not available")
def test_buffer_cuda_copy_from_host_with_buffer():
    """Test copy_from_host with a CPU Buffer object."""
    gpu_buf = BufferCuda.empty(3, "i")
    cpu_buf = Buffer.from_iterable([10, 20, 30], "i")
    assert isinstance(cpu_buf, Buffer)
    gpu_buf.copy_from_host(cpu_buf)
    assert list(gpu_buf.copy_to_host()) == [10, 20, 30]


@pytest.mark.skipif(not cuda_available, reason="CUDA not available")
def test_buffer_cuda_copy_from_host_size_mismatch():
    """Test copy_from_host raises on size mismatch."""
    gpu_buf = BufferCuda.empty(3, "i")
    cpu_buf = Buffer.from_iterable([10, 20], "i")  # Only 2 elements
    assert isinstance(cpu_buf, Buffer)
    with pytest.raises(ValueError, match="Buffer size mismatch"):
        gpu_buf.copy_from_host(cpu_buf)


@pytest.mark.skipif(not cuda_available, reason="CUDA not available")
def test_buffer_cuda_del():
    """Test that __del__ doesn't crash."""
    import gc

    # Just create and let it be garbage collected
    b = BufferCuda.empty(3, "i")
    address = b.address
    b.__del__()  # Directly call to ensure coverage
    gc.collect()  # Force garbage collection
    # If we get here without error, __del__ worked
    assert address != 0
