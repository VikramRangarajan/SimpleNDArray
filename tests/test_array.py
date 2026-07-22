import array
import shutil
from unittest.mock import MagicMock

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from simplendarray import kernels as _kernels
from simplendarray.array import Array, flatten_and_get_shape, reshape_strides
from simplendarray.buffer import Buffer
from simplendarray.buffer_cuda import BufferCuda
from simplendarray.kernels import (
    dispatch_arange,
    dispatch_element_wise_binary,
    dispatch_reduction,
    dispatch_reshape_copy,
)
from simplendarray.utils import contiguous_strides, product

cuda_available = shutil.which("nvcc") is not None


class TestFlattenAndGetShape:
    def test_scalar_int(self):
        assert flatten_and_get_shape(42) == ([42], ())

    def test_scalar_float(self):
        assert flatten_and_get_shape(3.14) == ([3.14], ())

    def test_scalar_bool(self):
        assert flatten_and_get_shape(True) == ([True], ())

    def test_empty_list(self):
        assert flatten_and_get_shape([]) == ([], (0,))

    def test_1d_list(self):
        flat, shape = flatten_and_get_shape([1, 2, 3])
        assert flat == [1, 2, 3]
        assert shape == (3,)

    def test_2d_list(self):
        flat, shape = flatten_and_get_shape([[1, 2], [3, 4]])
        assert flat == [1, 2, 3, 4]
        assert shape == (2, 2)

    def test_3d_list(self):
        flat, shape = flatten_and_get_shape([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        assert flat == [1, 2, 3, 4, 5, 6, 7, 8]
        assert shape == (2, 2, 2)

    def test_uneven_rows(self):
        data = [[1, 2], [3, 4, 5]]
        with pytest.raises(ValueError, match="Jagged array"):
            flatten_and_get_shape(data)

    def test_uneven_nested_shape(self):
        data = [[[1, 2], [3, 4]], [[5, 6], [7]]]
        with pytest.raises(ValueError, match="Jagged array"):
            flatten_and_get_shape(data)

    def test_single_row(self):
        flat, shape = flatten_and_get_shape([[10, 20, 30]])
        assert flat == [10, 20, 30]
        assert shape == (1, 3)


class TestArrayConstruction:
    def test_array_1d(self):
        t = Array.from_iterable([1, 2, 3], "i")
        assert t.device == "cpu"
        assert t.shape == (3,)
        assert list(t.data.data) == [1, 2, 3]

    def test_array_2d(self):
        t = Array.from_iterable([[1.0, 2.0], [3.0, 4.0]], "d")
        assert t.shape == (2, 2)
        assert list(t.data.data) == [1.0, 2.0, 3.0, 4.0]

    def test_array_scalar(self):
        t = Array.from_iterable(42, "i")
        assert t.shape == ()
        assert list(t.data.data) == [42]

    def test_array_empty(self):
        t = Array.from_iterable([], "i")
        assert t.shape == (0,)
        assert list(t.data.data) == []

    def test_array_jagged_raises(self):
        with pytest.raises(ValueError, match="Jagged array"):
            Array.from_iterable([[1, 2], [3]], "i")

    def test_array_repr_1d(self):
        t = Array.from_iterable([1, 2, 3], "i")
        r = repr(t)
        assert "shape=(3,)" in r
        assert "strides=(1,)" in r

    def test_array_repr_2d(self):
        t = Array.from_iterable([[1, 2], [3, 4]], "i")
        r = repr(t)
        assert "shape=(2, 2)" in r
        assert "strides=(2, 1)" in r

    def test_array_repr_scalar(self):
        t = Array.from_iterable(42, "i")
        r = repr(t)
        assert "shape=()" in r
        assert "strides=" in r

    def test_from_iterable_device_cpu_explicit(self):
        t = Array.from_iterable([1, 2, 3], "i", device="cpu")
        assert t.device == "cpu"
        assert t.shape == (3,)
        assert list(t.data.data) == [1, 2, 3]

    def test_from_iterable_device_gpu_mocked(self, monkeypatch):
        mock_buffer = MagicMock()
        mock_buffer.device = "gpu"
        mock_buffer.typecode = "i"
        monkeypatch.setattr(BufferCuda, "from_iterable", lambda data, dtype: mock_buffer)

        a = Array.from_iterable([1, 2, 3], "i", device="gpu")
        assert a.device == "gpu"
        assert a.shape == (3,)

    def test_from_iterable_invalid_device(self):
        with pytest.raises(KeyError):
            Array.from_iterable([1, 2, 3], "i", device="invalid")  # type: ignore[bad-argument-type]


def test_contiguous_strides_1d():
    assert contiguous_strides((3,)) == (1,)


def test_contiguous_strides_2d():
    assert contiguous_strides((2, 3)) == (3, 1)


def test_contiguous_strides_3d():
    assert contiguous_strides((2, 3, 4)) == (12, 4, 1)


def test_contiguous_strides_scalar():
    assert contiguous_strides(()) == ()


class TestSlicing:
    # Helper: create a flat 1D array [0, 1, 2, ..., n-1]
    @staticmethod
    def _1d(n: int, dtype: str = "i") -> Array:
        buf = Buffer(array.array(dtype, list(range(n))))
        return Array(buf, (n,), (1,), 0)

    @staticmethod
    def _2d(rows: int, cols: int, dtype: str = "i") -> Array:
        data = list(range(rows * cols))
        buf = Buffer(array.array(dtype, data))
        return Array(buf, (rows, cols), (cols, 1), 0)

    # --- 1D slicing ---

    def test_getitem_1d_int(self):
        a = self._1d(5)
        r = a[2]
        assert r.to_python() == [2]

    def test_getitem_1d_neg_int(self):
        a = self._1d(5)
        # a[-1] selects index 4, with shape (1,) stride (0,), squeeze to scalar
        r = a[-1].squeeze(0)
        assert r.is_contiguous
        assert r.data.data[r.offset] == 4

    def test_getitem_1d_full_slice(self):
        a = self._1d(5)
        r = a[:]
        assert r.shape == (5,)
        assert r.to_python() == [0, 1, 2, 3, 4]

    def test_getitem_1d_step(self):
        a = self._1d(10)
        r = a[::2]
        assert r.shape == (5,)
        assert r.strides == (2,)
        assert r.to_python() == [0, 2, 4, 6, 8]

    def test_getitem_1d_neg_step(self):
        a = self._1d(5)
        r = a[::-1]
        assert r.shape == (5,)
        assert r.strides == (-1,)
        assert r.offset == 4
        assert r.to_python() == [4, 3, 2, 1, 0]

    def test_getitem_1d_range_rev(self):
        a = self._1d(10)
        r = a[7:2:-2]
        assert r.shape == (3,)
        assert r.strides == (-2,)
        assert r.offset == 7
        assert r.to_python() == [7, 5, 3]

    def test_getitem_1d_start_stop_neg_step(self):
        a = self._1d(10)
        r = a[5:0:-1]
        assert r.shape == (5,)
        assert r.to_python() == [5, 4, 3, 2, 1]

    def test_getitem_1d_empty_slice(self):
        a = self._1d(5)
        r = a[10:20]
        assert r.shape == (0,)

    def test_getitem_1d_empty_neg_step(self):
        a = self._1d(5)
        r = a[0:10:-1]
        assert r.shape == (0,)

    def test_getitem_1d_zero_step_raises(self):
        a = self._1d(5)
        with pytest.raises(ValueError):
            a[::0]

    # --- 2D slicing ---

    def test_getitem_2d_int_row(self):
        a = self._2d(3, 4)
        r = a[1, :]
        assert r.shape == (1, 4)
        assert r.to_python() == [[4, 5, 6, 7]]

    def test_getitem_2d_int_col(self):
        a = self._2d(3, 4)
        r = a[:, 2]
        assert r.shape == (3, 1)
        assert r.to_python() == [[2], [6], [10]]

    def test_getitem_2d_submatrix(self):
        a = self._2d(4, 4)
        r = a[1:3, 1:3]
        assert r.shape == (2, 2)
        assert r.to_python() == [[5, 6], [9, 10]]

    def test_getitem_2d_neg_step_rows(self):
        a = self._2d(3, 4)
        r = a[::-1, :]
        assert r.shape == (3, 4)
        assert r.strides == (-4, 1)
        assert r.to_python() == [[8, 9, 10, 11], [4, 5, 6, 7], [0, 1, 2, 3]]

    def test_getitem_2d_neg_step_cols(self):
        a = self._2d(3, 4)
        r = a[:, ::-1]
        assert r.shape == (3, 4)
        assert r.strides == (4, -1)
        assert r.to_python() == [[3, 2, 1, 0], [7, 6, 5, 4], [11, 10, 9, 8]]

    def test_getitem_2d_both_neg_step(self):
        a = self._2d(3, 4)
        r = a[::-1, ::-1]
        assert r.shape == (3, 4)
        assert r.to_python() == [[11, 10, 9, 8], [7, 6, 5, 4], [3, 2, 1, 0]]

    def test_getitem_2d_odd_step(self):
        a = self._2d(4, 6)
        r = a[::2, ::3]
        assert r.shape == (2, 2)
        assert r.to_python() == [[0, 3], [12, 15]]

    def test_getitem_2d_neg_step_rows_odd(self):
        a = self._2d(4, 6)
        r = a[3:0:-2, :]
        assert r.shape == (2, 6)
        assert r.to_python() == [[18, 19, 20, 21, 22, 23], [6, 7, 8, 9, 10, 11]]

    # --- TypeError / ValueError ---

    def test_getitem_wrong_ndim(self):
        a = self._1d(5)
        with pytest.raises(ValueError, match="Must index the same number"):
            a[:, :]

    def test_getitem_typeerror(self):
        a = self._1d(5)
        with pytest.raises(TypeError):
            a["bad"]  # type: ignore[bad-index]

    # --- to_python ---

    def test_to_python_1d(self):
        a = self._1d(5)
        assert a.to_python() == [0, 1, 2, 3, 4]

    def test_to_python_2d(self):
        a = self._2d(2, 3)
        assert a.to_python() == [[0, 1, 2], [3, 4, 5]]

    def test_to_python_scalar(self):
        a = Array(Buffer(array.array("i", [42])), (), (), 0)
        assert a.to_python() == 42

    def test_to_python_rev_1d(self):
        a = self._1d(5)
        r = a[::-1]
        assert r.to_python() == [4, 3, 2, 1, 0]

    def test_to_python_rev_2d(self):
        a = self._2d(3, 4)
        r = a[::-1, ::-1]
        assert r.to_python() == [[11, 10, 9, 8], [7, 6, 5, 4], [3, 2, 1, 0]]

    def test_to_python_submatrix(self):
        a = self._2d(4, 4)
        r = a[1:3, 1:3]
        assert r.to_python() == [[5, 6], [9, 10]]

    def test_to_python_submatrix_rev(self):
        a = self._2d(4, 4)
        r = a[3:0:-1, 2::-1]
        expected = [[14, 13, 12], [10, 9, 8], [6, 5, 4]]
        assert r.to_python() == expected

    # --- squeeze ---

    def test_squeeze_1d(self):
        a = Array(Buffer(array.array("i", [10])), (1,), (0,), 0)
        s = a.squeeze(0)
        assert s.shape == ()
        assert s.strides == ()
        assert s.offset == 0

    def test_squeeze_raises_on_non_one(self):
        a = self._1d(5)
        with pytest.raises(ValueError, match="Can only squeeze"):
            a.squeeze(0)

    # --- transpose ---

    def test_transpose_swap_2d(self):
        a = self._2d(3, 4)
        t = a.transpose((1, 0))
        assert t.shape == (4, 3)
        assert t.strides == (1, 4)
        assert t.offset == 0
        assert t.to_python() == [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]]

    def test_transpose_identity(self):
        a = self._2d(3, 4)
        t = a.transpose((0, 1))
        assert t.shape == (3, 4)
        assert t.strides == (4, 1)
        assert t.to_python() == a.to_python()

    def test_transpose_3d(self):
        a = self._2d(2, 3)
        t = a.transpose((1, 0))
        assert t.shape == (3, 2)
        assert t.to_python() == [[0, 3], [1, 4], [2, 5]]

    def test_transpose_1d(self):
        a = self._1d(5)
        t = a.transpose((0,))
        assert t.shape == (5,)
        assert t.strides == (1,)
        assert t.to_python() == [0, 1, 2, 3, 4]

    # --- T property ---

    def test_T_2d(self):
        a = self._2d(2, 3)
        assert a.T.shape == (3, 2)
        assert a.T.strides == (1, 3)
        assert a.T.to_python() == [[0, 3], [1, 4], [2, 5]]

    def test_T_3d(self):
        import array

        from simplendarray.array import Array as Arr
        from simplendarray.buffer import Buffer

        buf = Buffer(array.array("i", list(range(24))))
        a = Arr(buf, (2, 3, 4), (12, 4, 1), 0)
        assert a.T.shape == (4, 3, 2)
        expected = [
            [[0, 12], [4, 16], [8, 20]],
            [[1, 13], [5, 17], [9, 21]],
            [[2, 14], [6, 18], [10, 22]],
            [[3, 15], [7, 19], [11, 23]],
        ]
        assert a.T.to_python() == expected

    def test_T_1d(self):
        a = self._1d(5)
        assert a.T.shape == (5,)
        assert a.T.to_python() == [0, 1, 2, 3, 4]

    # --- mT property ---

    def test_mT_2d(self):
        a = self._2d(2, 3)
        assert a.mT.shape == (3, 2)
        assert a.mT.strides == (1, 3)
        assert a.mT.to_python() == [[0, 3], [1, 4], [2, 5]]

    def test_mT_3d(self):
        import array

        from simplendarray.array import Array as Arr
        from simplendarray.buffer import Buffer

        buf = Buffer(array.array("i", list(range(24))))
        a = Arr(buf, (2, 3, 4), (12, 4, 1), 0)
        assert a.mT.shape == (2, 4, 3)
        expected = [
            [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]],
            [[12, 16, 20], [13, 17, 21], [14, 18, 22], [15, 19, 23]],
        ]
        assert a.mT.to_python() == expected

    def test_mT_1d_raises(self):
        a = self._1d(5)
        with pytest.raises(ValueError, match="matrix transpose with ndim < 2"):
            a.mT


def _make_array(shape: tuple[int, ...]) -> Array:
    numel = 1
    for d in shape:
        numel *= d
    flat = list(range(numel))
    buf = Buffer(array.array("i", flat))
    strides = contiguous_strides(shape)
    arr = Array(buf, shape, strides, 0)
    assert arr.is_contiguous
    return arr


def _factorizations(num: int) -> list[list[int]]:
    if num <= 0:
        return []
    results: list[list[int]] = []

    def _dfs(remaining: int, start: int, current: list[int]) -> None:
        if remaining == 1:
            if current:
                results.append(current[:])
            return
        for i in range(start, remaining + 1):
            if remaining % i == 0:
                current.append(i)
                _dfs(remaining // i, i, current)
                current.pop()

    if num >= 2:
        _dfs(num, 2, [])
    return results


@st.composite
def shape_and_slices(draw, max_ndim: int = 3, max_dim: int = 6):
    ndim = draw(st.integers(min_value=1, max_value=max_ndim))
    shape_list = draw(st.lists(st.integers(min_value=1, max_value=max_dim), min_size=ndim, max_size=ndim))
    shape = tuple(shape_list)
    slices = tuple(draw(st.slices(d)) for d in shape)
    return shape, slices


@st.composite
def shape_and_perm(draw, max_ndim: int = 4, max_dim: int = 6):
    ndim = draw(st.integers(min_value=1, max_value=max_ndim))
    shape_list = draw(st.lists(st.integers(min_value=1, max_value=max_dim), min_size=ndim, max_size=ndim))
    shape = tuple(shape_list)
    perm = draw(st.permutations(range(ndim)))
    return shape, perm


@st.composite
def shape_and_reshape_target(draw, max_ndim: int = 4, max_dim: int = 6):
    ndim = draw(st.integers(min_value=1, max_value=max_ndim))
    shape_list = draw(st.lists(st.integers(min_value=1, max_value=max_dim), min_size=ndim, max_size=ndim))
    shape = tuple(shape_list)
    numel = 1
    for d in shape:
        numel *= d
    if numel <= 1:
        return shape, shape
    factors = _factorizations(numel)
    if not factors:
        return shape, shape
    target = draw(st.sampled_from(factors))
    return shape, tuple(target)


@given(shape_and_slices())
def test_slicing_hypothesis(args):
    shape, slices = args
    a = _make_array(shape)
    na = np.arange(1, dtype=int).reshape(tuple())
    try:
        na = np.arange(1).reshape(shape)
    except ValueError:
        na = np.arange(len(a.data.data)).reshape(shape).astype(int)
    try:
        sliced = a[slices]
    except ValueError, NotImplementedError:
        return
    nsliced = na[slices]
    assert sliced.to_python() == nsliced.tolist()


@given(shape_and_perm())
def test_transpose_hypothesis(args):
    shape, perm = args
    a = _make_array(shape)
    na = np.arange(len(a.data.data)).reshape(shape).astype(int)
    t = a.transpose(tuple(perm))
    nt = na.transpose(tuple(perm))
    assert t.shape == nt.shape
    assert t.to_python() == nt.tolist()


@st.composite
def shape_and_reshape_target_with_neg1(draw, max_ndim: int = 4, max_dim: int = 6):
    """Like shape_and_reshape_target but also generates -1 in the target."""
    use_neg1 = draw(st.booleans())
    if not use_neg1:
        return draw(shape_and_reshape_target(max_ndim, max_dim))
    ndim = draw(st.integers(min_value=1, max_value=max_ndim))
    shape_list = draw(st.lists(st.integers(min_value=1, max_value=max_dim), min_size=ndim, max_size=ndim))
    shape = tuple(shape_list)
    numel = 1
    for d in shape:
        numel *= d
    if numel <= 1:
        return shape, shape
    factors = _factorizations(numel)
    if not factors:
        return shape, shape
    target = list(draw(st.sampled_from(factors)))
    # Replace one factor with -1
    idx = draw(st.integers(min_value=0, max_value=len(target) - 1))
    neg1_target = target[:]
    neg1_target[idx] = -1
    return shape, tuple(neg1_target)


@st.composite
def shape_and_reshape_int(draw, max_ndim: int = 4, max_dim: int = 6):
    """Generates (shape, int_target) where int_target == numel (i.e., flatten to 1d)."""
    ndim = draw(st.integers(min_value=1, max_value=max_ndim))
    shape_list = draw(st.lists(st.integers(min_value=1, max_value=max_dim), min_size=ndim, max_size=ndim))
    shape = tuple(shape_list)
    numel = 1
    for d in shape:
        numel *= d
    return shape, numel


@st.composite
def shape_and_reshape_neg1_scalar(draw, max_ndim: int = 4, max_dim: int = 6):
    """Generates (shape, -1) to test scalar -1 reshape (flatten)."""
    ndim = draw(st.integers(min_value=1, max_value=max_ndim))
    shape_list = draw(st.lists(st.integers(min_value=1, max_value=max_dim), min_size=ndim, max_size=ndim))
    shape = tuple(shape_list)
    return shape, -1


@given(shape_and_reshape_target())
def test_reshape_hypothesis(args):
    shape, new_shape = args
    a = _make_array(shape)
    na = np.arange(len(a.data.data)).reshape(shape).astype(int)
    r = a.reshape(new_shape)
    nr = na.reshape(new_shape)
    assert r.shape == nr.shape
    assert r.to_python() == nr.tolist()


@given(shape_and_reshape_target_with_neg1())
def test_reshape_hypothesis_with_neg1(args):
    shape, new_shape = args
    a = _make_array(shape)
    na = np.arange(len(a.data.data)).reshape(shape).astype(int)
    r = a.reshape(new_shape)
    nr = na.reshape(new_shape)
    assert r.shape == nr.shape
    assert r.to_python() == nr.tolist()


@given(shape_and_reshape_int())
def test_reshape_hypothesis_int(args):
    shape, int_target = args
    a = _make_array(shape)
    na = np.arange(len(a.data.data)).reshape(shape).astype(int)
    r = a.reshape(int_target)
    nr = na.reshape(int_target)
    assert r.shape == nr.shape
    assert r.to_python() == nr.tolist()


@given(shape_and_reshape_neg1_scalar())
def test_reshape_hypothesis_neg1_scalar(args):
    shape, int_target = args
    a = _make_array(shape)
    na = np.arange(len(a.data.data)).reshape(shape).astype(int)
    r = a.reshape(int_target)
    nr = na.reshape(int_target)
    assert r.shape == nr.shape
    assert r.to_python() == nr.tolist()


@given(shape_and_slices())
def test_T_hypothesis(args):
    shape, _ = args
    a = _make_array(shape)
    na = np.arange(len(a.data.data)).reshape(shape).astype(int)
    t = a.T
    nt = na.T
    assert t.shape == nt.shape
    assert t.to_python() == nt.tolist()


@given(shape_and_slices())
def test_mT_hypothesis(args):
    shape, _ = args
    if len(shape) < 2:
        return
    a = _make_array(shape)
    na = np.arange(len(a.data.data)).reshape(shape).astype(int)
    t = a.mT
    nt = na.mT
    assert t.shape == nt.shape
    assert t.to_python() == nt.tolist()


@given(shape_and_slices())
def test_T_equals_mT_2d_hypothesis(args):
    shape, _ = args
    if len(shape) != 2:
        return
    a = _make_array(shape)
    assert a.T.to_python() == a.mT.to_python()


def test_reshape_multiple_neg1_raises():
    a = _make_array((2, 3))
    with pytest.raises(ValueError, match="only one dimension can be -1"):
        a.reshape((-1, -1))


def test_reshape_dim_less_than_neg1_raises():
    a = _make_array((2, 3))
    with pytest.raises(ValueError, match="Dimension must be non-negative, or with a single -1"):
        a.reshape((-2,))


def test_reshape_shape_mismatch_raises():
    a = _make_array((2, 3))
    with pytest.raises(ValueError, match="do not share the same number of elements"):
        a.reshape((5,))


def test_reshape_neg1_non_divisible_raises():
    a = _make_array((4, 4))
    with pytest.raises(ValueError, match="Cannot reshape array"):
        a.reshape((-1, 5))


def test_reshape_non_contiguous_copy():
    a = _make_array((3, 4))[::2, :]
    assert a.strides == (8, 1)
    r = a.reshape((8,))
    assert r.to_python() == np.arange(12).reshape(3, 4)[::2, :].reshape(8).tolist()

    # from transpose (positive strides, non-contiguous)
    a = Array.from_iterable([1, 2, 3, 4, 5, 6], "f").reshape((2, 3)).T
    r = a.reshape((6,))
    assert r.to_python() == np.arange(1, 7).reshape(2, 3).T.reshape(6).tolist()

    # from negative stride on non-last axis
    a = _make_array((2, 3))[:, ::-1]
    r = a.reshape((6,))
    assert r.to_python() == np.arange(6).reshape(2, 3)[:, ::-1].reshape(6).tolist()

    # from reversed view (all negative strides)
    a = _make_array((2, 3))[::-1, ::-1]
    r = a.reshape((6,))
    assert r.to_python() == np.arange(6).reshape(2, 3)[::-1, ::-1].reshape(6).tolist()

    # 3d transpose + reshape
    a = _make_array((2, 3, 4)).transpose((2, 0, 1))
    r = a.reshape((24,))
    assert r.to_python() == np.arange(24).reshape(2, 3, 4).transpose((2, 0, 1)).reshape(24).tolist()

    # reshape to different dimensions (non-contiguous, copy)
    a = _make_array((4, 3))[::2, :]
    r = a.reshape((3, 2))
    assert r.to_python() == np.arange(12).reshape(4, 3)[::2, :].reshape(3, 2).tolist()

    # fully reversed 1-d view
    a = _make_array((5,))[::-1]
    r = a.reshape((5,))
    assert r.to_python() == np.arange(5)[::-1].reshape(5).tolist()


def test_product():
    assert product((2, 3)) == 6
    assert product(()) == 1
    assert product((5,)) == 5
    assert product((2, 3, 4)) == 24


def test_product_with_one():
    assert product((2, 1, 3)) == 6


def test_arange() -> None:
    a = Array.arange(5, "f")
    assert a.shape == (5,)
    assert a.strides == (1,)
    assert a.offset == 0
    assert a.to_python() == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_arange_negative_numel() -> None:
    with pytest.raises(ValueError, match="numel must be >= 0"):
        Array.arange(-1, "f")


def test_arange_int() -> None:
    a = Array.arange(5, "i")
    assert a.to_python() == [0, 1, 2, 3, 4]


def test_arange_double() -> None:
    a = Array.arange(3, "d")
    assert a.to_python() == [0.0, 1.0, 2.0]


def test_arange_device_cpu_explicit() -> None:
    a = Array.arange(5, "i", device="cpu")
    assert a.device == "cpu"
    assert a.shape == (5,)
    assert a.to_python() == [0, 1, 2, 3, 4]


def test_arange_device_gpu_mocked(monkeypatch) -> None:
    mock_buffer = MagicMock()
    mock_buffer.device = "gpu"
    mock_buffer.typecode = "i"
    monkeypatch.setattr(BufferCuda, "empty", lambda size, dtype: mock_buffer)

    mock_dispatch = MagicMock()
    monkeypatch.setattr("simplendarray.array.dispatch_arange", mock_dispatch)

    a = Array.arange(5, "i", device="gpu")
    assert a.device == "gpu"
    assert a.shape == (5,)
    mock_dispatch.assert_called_once_with(mock_buffer, 0, 1, 5)


class TestEmpty:
    def test_empty_basic(self):
        a = Array.empty(5, "i")
        assert a.shape == (5,)
        assert a.strides == (1,)
        assert list(a.data.data) == [0, 0, 0, 0, 0]

    def test_empty_with_float_dtype(self):
        a = Array.empty(3, "d")
        assert a.shape == (3,)
        assert list(a.data.data) == [0.0, 0.0, 0.0]

    def test_empty_negative_raises(self):
        with pytest.raises(ValueError, match="numel must be >= 0"):
            Array.empty(-1, "i")


def test_arange_invalid_device() -> None:
    with pytest.raises(KeyError):
        Array.arange(5, "i", device="invalid")  # type: ignore[bad-argument-type]


class TestReshapeStrides:
    def test_flatten_contiguous_2d(self):
        s = reshape_strides((2, 3), (6,), (3, 1))
        assert s == (1,)

    def test_expand_1d_to_2d(self):
        s = reshape_strides((6,), (2, 3), (1,))
        assert s == (3, 1)

    def test_identity(self):
        s = reshape_strides((2, 3), (2, 3), (3, 1))
        assert s == (3, 1)

    def test_combine_last_two(self):
        s = reshape_strides((2, 3, 4), (2, 12), (12, 4, 1))
        assert s == (12, 1)

    def test_split_first(self):
        s = reshape_strides((12,), (4, 3), (1,))
        assert s == (3, 1)

    def test_with_old_ones_squeezed(self):
        s = reshape_strides((1, 6), (2, 3), (3, 1))
        assert s == (3, 1)

    def test_with_new_ones(self):
        s = reshape_strides((6,), (1, 2, 1, 3), (1,))
        assert s == (6, 3, 3, 1)

    def test_into_single_element(self):
        s = reshape_strides((1, 6), (6,), (6, 1))
        assert s == (1,)

    def test_into_scalar(self):
        s = reshape_strides((1,), (), (0,))
        assert s == ()

    def test_non_contiguous_returns_none(self):
        s = reshape_strides((3, 4), (12,), (4, 2))
        assert s is None

    def test_strided_2d_combine(self):
        s = reshape_strides((2, 6), (4, 3), (12, 2))
        assert s == (6, 2)

    def test_empty_array_returns_zeros(self):
        s = reshape_strides((0,), (0, 5), (1,))
        assert s == (0, 0)

    def test_empty_array_noop(self):
        s = reshape_strides((0,), (0,), (1,))
        assert s == (0,)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="do not share the same number of elements"):
            reshape_strides((2, 3), (5,), (3, 1))

    def test_non_contiguous_within_combine(self):
        s = reshape_strides((2, 3, 4), (6, 4), (12, 4, 2))
        assert s == (4, 2)

    def test_ones_squeeze_to_contiguous(self):
        s = reshape_strides((3, 1, 4), (3, 4), (4, 4, 1))
        assert s == (4, 1)

    def test_old_strides_with_ones(self):
        s = reshape_strides((2, 1, 3), (6,), (3, 1, 1))
        assert s == (1,)

    def test_multiple_combines(self):
        s = reshape_strides((2, 3, 4), (24,), (12, 4, 1))
        assert s == (1,)

    def test_contiguous_strided(self):
        s = reshape_strides((6,), (3, 2), (2,))
        assert s == (4, 2)

    def test_triple_expand(self):
        s = reshape_strides((12,), (3, 2, 2), (1,))
        assert s == (4, 2, 1)

    def test_triple_flatten(self):
        s = reshape_strides((3, 2, 2), (12,), (4, 2, 1))
        assert s == (1,)

    def test_contiguous_strided_with_neg_stride(self):
        s = reshape_strides((6,), (2, 3), (-1,))
        assert s == (-3, -1)

    def test_squeeze_strided_reshape(self):
        s = reshape_strides((2, 1, 4), (2, 4), (8, 4, 2))
        assert s == (8, 2)

    def test_trailing_ones_in_new_shape(self):
        s = reshape_strides((6,), (2, 3, 1), (1,))
        assert s == (3, 1, 1)

    def test_all_ones_old_shape(self):
        s = reshape_strides((1, 1), (1, 1), (0, 0))
        assert s == (1, 1)


class TestRelu:
    def test_relu_1d(self):
        a = Array.from_iterable([-3, -1, 0, 2, 4], "i")
        r = a.relu()
        assert r.to_python() == [0, 0, 0, 2, 4]

    def test_relu_float(self):
        a = Array.from_iterable([-1.5, -0.0, 2.5], "f")
        r = a.relu()
        assert r.to_python() == [0.0, 0.0, 2.5]

    def test_relu_2d(self):
        a = Array.from_iterable([[-1, 2], [3, -4]], "i")
        r = a.relu()
        assert r.to_python() == [[0, 2], [3, 0]]


class TestAdd:
    def test_add_1d(self):
        a = Array.from_iterable([1, 2, 3], "i")
        b = Array.from_iterable([4, 5, 6], "i")
        c = a + b
        assert c.to_python() == [5, 7, 9]

    def test_add_2d(self):
        a = Array.from_iterable([[1, 2], [3, 4]], "i")
        b = Array.from_iterable([[5, 6], [7, 8]], "i")
        c = a + b
        assert c.to_python() == [[6, 8], [10, 12]]

    def test_add_float_1d(self):
        a = Array.from_iterable([1.0, 2.0, 3.0], "d")
        b = Array.from_iterable([0.5, 0.5, 0.5], "d")
        c = a + b
        assert c.to_python() == [1.5, 2.5, 3.5]


@pytest.mark.skipif(not cuda_available, reason="CUDA not available")
def test_from_iterable_device_gpu_integration() -> None:
    a = Array.from_iterable([1, 2, 3], "i", device="gpu")
    assert a.device == "gpu"
    assert a.shape == (3,)
    assert a.to_python() == [1, 2, 3]


@pytest.mark.skipif(not cuda_available, reason="CUDA not available")
def test_arange_device_gpu_integration() -> None:
    a = Array.arange(5, "i", device="gpu")
    assert a.device == "gpu"
    assert a.shape == (5,)
    assert a.strides == (1,)
    assert a.to_python() == [0, 1, 2, 3, 4]


class TestDispatchErrorPaths:
    def test_dispatch_element_wise_binary_gpu(self, monkeypatch):
        mock_dispatch = MagicMock()
        mock_gpu_module = MagicMock()
        mock_gpu_module.DISPATCH_DICT_element_wise_binary = {
            (("T", "int"), ("Op", "_add_int")): mock_dispatch,
        }
        monkeypatch.setitem(_kernels.elem_wise_modules, "gpu", mock_gpu_module)

        mock_buf = MagicMock()
        mock_buf.device = "gpu"
        mock_buf.typecode = "i"
        mock_buf.size = 3

        class FakeArray(Array):
            def __init__(self):
                self.data = mock_buf
                self.shape = (3,)
                self.strides = (1,)
                self.offset = 0

            def reshape(self, new_shape):
                return self

            @property
            def is_contiguous(self):
                return True

            @property
            def size(self):
                return 3

        a = FakeArray()
        b = FakeArray()
        c = FakeArray()
        dispatch_element_wise_binary(a, b, c, "add")
        mock_dispatch.assert_called_once()

    def test_element_wise_binary_size_mismatch(self):
        a = _make_array((3,))
        b = _make_array((4,))
        out = _make_array((3,))
        with pytest.raises(ValueError, match="same size, dtype, and device"):
            dispatch_element_wise_binary(a, b, out, "add")

    def test_element_wise_binary_typecode_mismatch(self):
        a = Array.from_iterable([1, 2, 3], "i")
        b = Array.from_iterable([1, 2, 3], "i")
        out = Array.from_iterable([0, 0, 0], "f")
        with pytest.raises(ValueError, match="same size, dtype, and device"):
            dispatch_element_wise_binary(a, b, out, "add")

    def test_element_wise_binary_non_contiguous_output(self):
        a = _make_array((6,))
        b = _make_array((6,))
        out_buf = Buffer(array.array("i", list(range(20))))
        out = Array(out_buf, (6,), (2,), 0)
        assert not out.is_contiguous
        with pytest.raises(ValueError, match="Output not contiguous"):
            dispatch_element_wise_binary(a, b, out, "add")

    def test_dispatch_reshape_copy_gpu(self, monkeypatch):
        mock_dispatch = MagicMock()
        mock_gpu_module = MagicMock()
        mock_gpu_module.DISPATCH_DICT_reshape_copy = {
            (("T", "int"), ("Kernel", "reshape_copy_kernel_int")): mock_dispatch,
        }
        monkeypatch.setitem(_kernels.elem_wise_modules, "gpu", mock_gpu_module)

        mock_buf = MagicMock()
        mock_buf.device = "gpu"
        mock_buf.typecode = "i"
        mock_buf.size = 6

        class FakeArray(Array):
            def __init__(self):
                self.data = mock_buf
                self.shape = (6,)
                self.strides = (1,)
                self.offset = 0

            @property
            def size(self):
                return 6

            @property
            def ndim(self):
                return len(self.shape)

        arr = FakeArray()
        new_buffer = MagicMock()
        inp_shape = MagicMock()
        inp_strides = MagicMock()
        out_shape = MagicMock()
        out_strides = MagicMock()
        inp_work = MagicMock()
        out_work = MagicMock()

        dispatch_reshape_copy(
            arr,
            (2, 3),
            new_buffer,
            inp_shape,
            inp_strides,
            inp_work,
            out_shape,
            out_strides,
            out_work,
        )
        mock_dispatch.assert_called_once()

    def test_dispatch_arange_gpu(self, monkeypatch):
        mock_dispatch = MagicMock()
        mock_gpu_module = MagicMock()
        mock_gpu_module.DISPATCH_DICT_arange = {
            (("T", "int"), ("Kernel", "arange_kernel_int")): mock_dispatch,
        }
        monkeypatch.setitem(_kernels.elem_wise_modules, "gpu", mock_gpu_module)

        mock_buf = MagicMock()
        mock_buf.device = "gpu"
        mock_buf.typecode = "i"
        mock_buf.address = 12345
        dispatch_arange(mock_buf, 0, 1, 5)
        mock_dispatch.assert_called_once_with(12345, 0, 1, 5)


class TestBinaryOpBroadcasting:
    def test_add_with_scalar_rev(self):
        a = Array.from_iterable([1, 2, 3], "i")
        r = 5 + a
        assert r.to_python() == [6, 7, 8]

    def test_add_with_scalar(self):
        a = Array.from_iterable([1, 2, 3], "i")
        r = a + 5
        assert r.to_python() == [6, 7, 8]

    def test_sub_with_scalar(self):
        a = Array.from_iterable([10, 20, 30], "i")
        r = a - 5
        assert r.to_python() == [5, 15, 25]

    def test_mul_with_scalar(self):
        a = Array.from_iterable([1, 2, 3], "i")
        r = a * 3
        assert r.to_python() == [3, 6, 9]

    def test_div_with_scalar(self):
        a = Array.from_iterable([10.0, 20.0, 30.0], "d")
        r = a / 2.0
        assert r.to_python() == [5.0, 10.0, 15.0]

    def test_broadcast_1d_to_2d(self):
        a = Array.from_iterable([[1, 2, 3], [4, 5, 6]], "i")
        b = Array.from_iterable([10, 20, 30], "i")
        r = a + b
        assert r.to_python() == [[11, 22, 33], [14, 25, 36]]

    def test_broadcast_2d_to_1d(self):
        a = Array.from_iterable([10, 20, 30], "i")
        b = Array.from_iterable([[1, 2, 3]], "i")
        r = a + b
        assert r.to_python() == [[11, 22, 33]]

    def test_broadcast_dim1_expand(self):
        a = Array.from_iterable([[1, 2, 3]], "i")
        b = Array.from_iterable([[10, 20, 30], [40, 50, 60]], "i")
        r = a + b
        assert r.to_python() == [[11, 22, 33], [41, 52, 63]]

    def test_broadcast_non_compatible_raises(self):
        a = Array.from_iterable([1, 2], "i")
        b = Array.from_iterable([1, 2, 3], "i")
        with pytest.raises(ValueError, match="Not Broadcastable"):
            a + b


class TestTransposeErrors:
    def test_transpose_wrong_dims_raises(self):
        a = _make_array((2, 3))
        with pytest.raises(ValueError, match="Transpose dims needs to be the same length"):
            a.transpose((0,))


class TestReduction:
    def test_sum_2d_axis0(self):
        a = Array.from_iterable([[1, 2], [3, 4]], "i")
        r = a.sum((0,))
        assert r.to_python() == [4, 6]

    def test_sum_2d_axis1(self):
        a = Array.from_iterable([[1, 2], [3, 4]], "i")
        r = a.sum((1,))
        assert r.to_python() == [3, 7]

    def test_sum_2d_float(self):
        a = Array.from_iterable([[1.5, 2.5], [3.5, 4.5]], "d")
        r = a.sum((0,))
        assert r.to_python() == [5.0, 7.0]

    def test_max_2d_axis1(self):
        a = Array.from_iterable([[1, 5], [3, 2]], "i")
        r = a.max((1,))
        assert r.to_python() == [5, 3]

    def test_min_2d_axis0(self):
        a = Array.from_iterable([[1, 5], [3, 2]], "i")
        r = a.min((0,))
        assert r.to_python() == [1, 2]

    def test_prod_2d_axis0(self):
        a = Array.from_iterable([[2, 3], [4, 5]], "i")
        r = a.prod((0,))
        assert r.to_python() == [8, 15]

    def test_prod_2d_axis1(self):
        a = Array.from_iterable([[2, 3], [4, 5]], "i")
        r = a.prod((1,))
        assert r.to_python() == [6, 20]


class TestReductionErrorPaths:
    def test_typecode_mismatch(self):
        a = _make_array((2, 3))
        out_buf = Buffer(array.array("f", [0.0, 0.0, 0.0]))
        out = Array(out_buf, (3,), (1,), 0)
        with pytest.raises(ValueError, match="same size, dtype, and device"):
            dispatch_reduction(a, out, "add", (0,))

    def test_device_mismatch(self):
        mock_buf_cpu = MagicMock()
        mock_buf_cpu.typecode = "i"
        mock_buf_cpu.device = "cpu"
        mock_buf_gpu = MagicMock()
        mock_buf_gpu.typecode = "i"
        mock_buf_gpu.device = "gpu"

        a = MagicMock()
        a.data = mock_buf_cpu
        a.shape = (2, 3)
        a.strides = (3, 1)
        a.offset = 0
        a.device = "cpu"
        a.ndim = 2
        a.size = 6

        out = MagicMock()
        out.data = mock_buf_gpu
        out.shape = (3,)
        out.strides = (1,)
        out.offset = 0
        out.device = "gpu"

        with pytest.raises(ValueError, match="same size, dtype, and device"):
            dispatch_reduction(a, out, "add", (0,))

    def test_ndim_lt_2_raises(self):
        a = _make_array((4,))
        out = _make_array((4,))
        with pytest.raises(ValueError, match="Unsqueeze not implemented yet"):
            dispatch_reduction(a, out, "add", (0,))

    def test_output_shape_mismatch(self):
        a = _make_array((3, 4))
        out_buf = Buffer(array.array("i", [0] * 5))
        out = Array(out_buf, (5,), (1,), 0)
        with pytest.raises(ValueError, match="Output array is invalid shape"):
            dispatch_reduction(a, out, "add", (0,))

    def test_reduction_op_ndim_lt_2(self):
        a = Array.from_iterable([1, 2, 3, 4], "i")
        with pytest.raises(ValueError, match="Unsqueeze not implemented yet"):
            a.sum((0,))

    def test_gpu_mocked(self, monkeypatch):
        mock_dispatch = MagicMock()
        mock_gpu_module = MagicMock()
        mock_gpu_module.DISPATCH_DICT_reduction = {
            (("T", "int"), ("Op", "reduction_kernel_add_int")): mock_dispatch,
        }
        monkeypatch.setitem(_kernels.reduction_modules, "gpu", mock_gpu_module)

        mock_buf = MagicMock()
        mock_buf.device = "gpu"
        mock_buf.typecode = "i"
        mock_buf.address = 9999

        class FakeArray:
            data = mock_buf
            shape = (2, 3)
            strides = (3, 1)
            offset = 0
            device = "gpu"
            ndim = 2
            size = 6

            def reshape(self, new_shape):
                r = MagicMock()
                r.data = mock_buf
                r.shape = (3, 2)
                r.strides = (1, 3)
                r.offset = 0
                r.device = "gpu"
                return r

            def transpose(self, dims):
                r = MagicMock()
                r.data = mock_buf
                r.shape = (3, 2)
                r.strides = (1, 3)
                r.offset = 0
                r.device = "gpu"

                r2 = MagicMock()
                r2.data = mock_buf
                r2.shape = (3, 2)
                r2.strides = (1, 3)
                r2.offset = 0
                r2.device = "gpu"
                r.reshape = lambda new_shape: r2
                return r

        out = MagicMock()
        out.data = mock_buf
        out.shape = (3,)
        out.strides = (1,)
        out.offset = 0
        out.device = "gpu"

        a = FakeArray()
        dispatch_reduction(a, out, "add", (0,))  # pyrefly: ignore [bad-argument-type]
        mock_dispatch.assert_called_once()
