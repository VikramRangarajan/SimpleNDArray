import torch

torch.set_float32_matmul_precision("high")

n = 16384
a = torch.randn((n, n), device="cuda")
b = torch.randn((n, n), device="cuda")

# warmup
for _ in range(20):
    torch.mm(a, b)

torch.cuda.synchronize()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

iters = 20

start.record()
for _ in range(iters):
    torch.mm(a, b)
end.record()

torch.cuda.synchronize()

ms = start.elapsed_time(end) / iters
tflops = (2 * n**3) / (ms * 1e-3) / 1e12

print(ms, tflops)
