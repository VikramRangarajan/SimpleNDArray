type cudaError_t = int
cudaMemcpyHostToHost: int = 0
cudaMemcpyHostToDevice: int = 1
cudaMemcpyDeviceToHost: int = 2
cudaMemcpyDeviceToDevice: int = 3

cudaSuccess: cudaError_t = 0


class dim3(list):
    x: int
    y: int
    z: int


class threadIdx(dim3): ...


class blockIdx(dim3): ...


class blockDim(dim3): ...


class gridDim(dim3): ...


warpSize: int
printf = print


def __syncthreads(): ...
def atomicAdd[T](ptr: list[T], elem: T): ...
def atomicMin[T](ptr: list[T], elem: T): ...
def atomicMax[T](ptr: list[T], elem: T): ...
