from itertools import batched
from statistics import mean

import altair as alt
import polars as pl
from utils import benchmark, fp32_flops, mem_bandwidth


def get_stats(x: list[int]):
    return mean(x), max(x), min(x)


def run():
    pl.Config.set_tbl_rows(10**5)
    res = []
    for log_n in [9, 10, 11, 12, 13]:  # range(1, 14, 3):
        n = 1 << log_n
        print(f"Benchmarking m = n = k = {n} (2^{log_n})")
        REPEATS = 2

        @benchmark({"N": str(n), "REPEATS": str(REPEATS)})
        def snda():
            import os

            from simplendarray import Array

            n = int(os.environ["N"])

            for i in range(int(os.environ["REPEATS"])):
                a = Array.arange(n * n, "f", "gpu").reshape((1, n, n))
                b = Array.arange(n * n, "f", "gpu").reshape((1, n, n)).mT
                _trash = Array.empty(2**26, "f", "gpu").sin()  # Clear L2 Cache
                _c = a @ b
                _trash = Array.empty(2**26, "f", "gpu").sin()  # Clear L2 Cache

        @benchmark({"N": str(n), "REPEATS": str(REPEATS)})
        def torch():
            import os

            import torch

            torch.set_float32_matmul_precision("highest")

            n = int(os.environ["N"])

            for i in range(int(os.environ["REPEATS"])):
                a = torch.arange(n * n, dtype=torch.float, device="cuda").reshape((n, n))
                b = torch.arange(n * n, dtype=torch.float, device="cuda").reshape((n, n)).mT
                _trash = torch.empty(2**26, dtype=torch.float, device="cuda").sin()  # Clear L2 Cache
                _c = a @ b
                _trash = torch.empty(2**26, dtype=torch.float, device="cuda").sin()  # Clear L2 Cache

        num_bytes = 4 * n * n * 4  # read A, B, C and write C: 4 matrices × n² elements × 4 bytes
        FLOPS = (
            2 * n * n * (n - 1) + 3 * n * n
        )  # 2n^2(n-1) for the matmul, and an extra 3n^2 because we do C=alpha*AC+beta*C
        AI = FLOPS / num_bytes  # flops per byte
        MAX_FLOPS_PER_SEC = min(
            fp32_flops(), AI * mem_bandwidth()
        )  # max flops / sec for this problem size and hardware

        our_times = snda()
        our_bmm = [x for x in our_times if "bmm" in x["name"]]
        our_bmm_durations_ns = [sum(x["duration_ns"] for x in batch) for batch in batched(our_bmm, 1)]
        mean_our_bmm, max_our_bmm, min_our_bmm = get_stats(our_bmm_durations_ns)

        mean_our_bmm_gbps = (num_bytes / mean_our_bmm) * 10**9
        min_our_bmm_gbps = (num_bytes / max_our_bmm) * 10**9  # max duration → min bandwidth
        max_our_bmm_gbps = (num_bytes / min_our_bmm) * 10**9  # min duration → max bandwidth

        res.append(
            {
                "kernel": "bmm",
                "SoL % Mean": mean_our_bmm_gbps * AI / MAX_FLOPS_PER_SEC * 100,
                "SoL % Min": min_our_bmm_gbps * AI / MAX_FLOPS_PER_SEC * 100,
                "SoL % Max": max_our_bmm_gbps * AI / MAX_FLOPS_PER_SEC * 100,
                "lib": "snda",
                "n": n,
            }
        )

        torch_times = torch()
        torch_bmm = [x for x in torch_times if "gemm" in x["name"]]
        num_kernels_per_bmm = 1  # len(torch_bmm) // REPEATS
        torch_bmm_durations_ns = [
            sum(x["duration_ns"] for x in batch) for batch in batched(torch_bmm, num_kernels_per_bmm)
        ]
        mean_bmm_torch, max_bmm_torch, min_bmm_torch = get_stats(torch_bmm_durations_ns)
        mean_torch_bmm_gbps = (num_bytes / mean_bmm_torch) * 10**9
        min_torch_bmm_gbps = (num_bytes / max_bmm_torch) * 10**9  # max duration → min bandwidth
        max_torch_bmm_gbps = (num_bytes / min_bmm_torch) * 10**9  # min duration → max bandwidth

        res.append(
            {
                "kernel": "bmm",
                "SoL % Mean": mean_torch_bmm_gbps * AI / MAX_FLOPS_PER_SEC * 100,
                "SoL % Min": min_torch_bmm_gbps * AI / MAX_FLOPS_PER_SEC * 100,
                "SoL % Max": max_torch_bmm_gbps * AI / MAX_FLOPS_PER_SEC * 100,
                "lib": "torch",
                "n": n,
            }
        )
    df = pl.DataFrame(res)
    df.write_csv("bench.csv")
    df = df.with_columns(
        actual_max=pl.max_horizontal("SoL % Mean", "SoL % Min", "SoL % Max"),
        actual_min=pl.min_horizontal("SoL % Mean", "SoL % Min", "SoL % Max"),
    )

    for kernel in df["kernel"].unique():
        sub = df.filter(pl.col("kernel") == kernel)

        line = (
            alt.Chart(sub)
            .mark_line()
            .encode(
                x=alt.X("n", scale=alt.Scale(type="log", base=2)),
                y=alt.Y("SoL % Mean").title("SoL %"),
                color="lib",
            )
        )

        band = (
            alt.Chart(sub)
            .mark_errorband(opacity=0.2)
            .encode(
                x=alt.X("n", scale=alt.Scale(type="log", base=2)),
                y=alt.Y("actual_min").title("SoL %"),
                y2="actual_max",
                color="lib",
            )
        )

        (line + band).save(f"plot_{kernel}.png", ppi=350)
        print(f"Saved plot_{kernel}.png")


run()
