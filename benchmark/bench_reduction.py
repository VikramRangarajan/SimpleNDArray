from itertools import batched
from statistics import mean

import altair as alt
import polars as pl
from utils import benchmark, mem_bandwidth


def get_stats(x: list[int]):
    return mean(x), max(x), min(x)


def run():
    pl.Config.set_tbl_rows(10**5)
    res = []
    for log_n in range(0, 31, 3):
        n = 1 << log_n
        print(f"Benchmarking n = {n} (2^{log_n})")
        REPEATS = 10

        @benchmark({"N": str(n), "REPEATS": str(REPEATS)})
        def snda():
            import os

            from simplendarray import Array

            for i in range(int(os.environ["REPEATS"])):
                a = Array.arange(int(os.environ["N"]), "f", "gpu").reshape((1, -1))
                _trash = Array.empty(2**26, "f", "gpu").sin()  # Clear L2 Cache
                _b = a.sum((1,))
                _trash = Array.empty(2**26, "f", "gpu").sin()  # Clear L2 Cache

        @benchmark({"N": str(n), "REPEATS": str(REPEATS)})
        def torch():
            import os

            import torch

            for i in range(int(os.environ["REPEATS"])):
                a = torch.arange(int(os.environ["N"]), dtype=torch.float, device="cuda").reshape((1, -1))
                _trash = torch.empty(2**26, dtype=torch.float, device="cuda").sin()  # Clear L2 Cache
                _b = a.sum((1,))
                _trash = torch.empty(2**26, dtype=torch.float, device="cuda").sin()  # Clear L2 Cache

        num_bytes = n * 4
        our_times = snda()
        our_sum = [x for x in our_times if "reduction_kernel" in x["name"]]
        num_kernels_per_sum = len(our_sum) // REPEATS
        our_sum_durations_ns = [sum(x["duration_ns"] for x in batch) for batch in batched(our_sum, num_kernels_per_sum)]
        mean_our_sum, min_our_sum, max_our_sum = get_stats(our_sum_durations_ns)

        mean_our_sum_gbps = (num_bytes / mean_our_sum) * 10**9
        min_our_sum_gbps = (num_bytes / min_our_sum) * 10**9
        max_our_sum_gbps = (num_bytes / max_our_sum) * 10**9

        res.append(
            {
                "kernel": "sum",
                "SoL % Mean": mean_our_sum_gbps / mem_bandwidth() * 100,  # 1 write
                "SoL % Min": min_our_sum_gbps / mem_bandwidth() * 100,  # 1 write
                "SoL % Max": max_our_sum_gbps / mem_bandwidth() * 100,  # 1 write
                "lib": "snda",
                "n": n,
            }
        )

        torch_times = torch()
        torch_sum = [x for x in torch_times if "sum" in x["name"]]
        num_kernels_per_sum = len(torch_sum) // REPEATS
        torch_sum_durations_ns = [
            sum(x["duration_ns"] for x in batch) for batch in batched(torch_sum, num_kernels_per_sum)
        ]
        mean_sum_torch, max_sum_torch, min_sum_torch = get_stats(torch_sum_durations_ns)
        mean_torch_sum_gbps = (num_bytes / mean_sum_torch) * 10**9
        min_torch_sum_gbps = (num_bytes / min_sum_torch) * 10**9
        max_torch_sum_gbps = (num_bytes / max_sum_torch) * 10**9

        res.append(
            {
                "kernel": "sum",
                "SoL % Mean": mean_torch_sum_gbps / mem_bandwidth() * 100,  # 1 read
                "SoL % Min": min_torch_sum_gbps / mem_bandwidth() * 100,  # 1 read
                "SoL % Max": max_torch_sum_gbps / mem_bandwidth() * 100,  # 1 read
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
