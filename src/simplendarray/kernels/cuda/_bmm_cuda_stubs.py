# Auto-generated stub for compiled functions.
from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Callable, ClassVar

from simplendarray.transpiler.runtime import PythonModule

class _BmmModuleClass(PythonModule):
    if TYPE_CHECKING:
        DISPATCH_DICT_bmm: ClassVar[dict[str, Callable[..., None]]]

        def bmm_float(self, a_ptr: int, a_off: int, a_stride_b: int, a_stride_m: int, a_stride_k: int, b_ptr: int, b_off: int, b_stride_b: int, b_stride_k: int, b_stride_n: int, c_ptr: int, c_off: int, c_stride_b: int, c_stride_m: int, c_stride_n: int, B: int, M: int, K: int, N: int, alpha: float, beta: float) -> None: ...
        pass
