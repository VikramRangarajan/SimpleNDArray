import pytest

from simplendarray.transpiler.runtime import PythonModule


@pytest.fixture(autouse=True)
def _runtime_cache_dir(tmp_path):
    original = PythonModule._cache_dir_override
    PythonModule._cache_dir_override = tmp_path / "simplendarray_cache"
    yield
    PythonModule._cache_dir_override = original
