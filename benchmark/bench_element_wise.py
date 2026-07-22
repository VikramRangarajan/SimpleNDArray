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
                a = Array.arange(int(os.environ["N"]), "f", "gpu")
                _trash = Array.empty(2**26, "f", "gpu").sin()  # Clear L2 Cache
                _b = a.tanh()
                _trash = Array.empty(2**26, "f", "gpu").sin()  # Clear L2 Cache

        @benchmark({"N": str(n), "REPEATS": str(REPEATS)})
        def torch():
            import os

            import torch

            for i in range(int(os.environ["REPEATS"])):
                a = torch.arange(int(os.environ["N"]), dtype=torch.float, device="cuda")
                _trash = torch.empty(2**26, dtype=torch.float, device="cuda").sin()  # Clear L2 Cache
                _b = a.tanh()
                _trash = torch.empty(2**26, dtype=torch.float, device="cuda").sin()  # Clear L2 Cache

        num_bytes = n * 4
        our_times = snda()
        our_arange = [x for x in our_times if "arange" in x["name"]]
        our_arange_durations_ns = [x["duration_ns"] for x in our_arange]
        mean_our_arange, min_our_arange, max_our_arange = get_stats(our_arange_durations_ns)

        mean_our_arange_gbps = (num_bytes / mean_our_arange) * 10**9
        min_our_arange_gbps = (num_bytes / min_our_arange) * 10**9
        max_our_arange_gbps = (num_bytes / max_our_arange) * 10**9

        our_tanh = [x for x in our_times if "tanh" in x["name"]]
        our_tanh_durations_ns = [x["duration_ns"] for x in our_tanh]
        mean_our_tanh, min_our_tanh, max_our_tanh = get_stats(our_tanh_durations_ns)

        mean_our_tanh_gbps = (num_bytes / mean_our_tanh) * 10**9
        min_our_tanh_gbps = (num_bytes / min_our_tanh) * 10**9
        max_our_tanh_gbps = (num_bytes / max_our_tanh) * 10**9
        res.append(
            {
                "kernel": "arange",
                "SoL % Mean": mean_our_arange_gbps / mem_bandwidth() * 100,  # 1 write
                "SoL % Min": min_our_arange_gbps / mem_bandwidth() * 100,  # 1 write
                "SoL % Max": max_our_arange_gbps / mem_bandwidth() * 100,  # 1 write
                "lib": "snda",
                "n": n,
            }
        )
        res.append(
            {
                "kernel": "tanh",
                "SoL % Mean": 2 * mean_our_tanh_gbps / mem_bandwidth() * 100,  # 1 read + 1 write, 2x
                "SoL % Min": 2 * min_our_tanh_gbps / mem_bandwidth() * 100,  # 1 read + 1 write, 2x
                "SoL % Max": 2 * max_our_tanh_gbps / mem_bandwidth() * 100,  # 1 read + 1 write, 2x
                "lib": "snda",
                "n": n,
            }
        )
        torch_times = torch()
        torch_arange = [x for x in torch_times if "arange" in x["name"]]
        num_kernels_per_arange = len(torch_arange) // REPEATS
        torch_arange_durations_ns = [
            sum(x["duration_ns"] for x in batch) for batch in batched(torch_arange, num_kernels_per_arange)
        ]
        mean_arange_torch, max_arange_torch, min_arange_torch = get_stats(torch_arange_durations_ns)
        mean_torch_arange_gbps = (num_bytes / mean_arange_torch) * 10**9
        min_torch_arange_gbps = (num_bytes / min_arange_torch) * 10**9
        max_torch_arange_gbps = (num_bytes / max_arange_torch) * 10**9
        torch_tanh = [x for x in torch_times if "tanh" in x["name"]]
        num_kernels_per_tanh = len(torch_tanh) // REPEATS
        torch_tanh_durations_ns = [
            sum(x["duration_ns"] for x in batch) for batch in batched(torch_tanh, num_kernels_per_tanh)
        ]
        mean_tanh_torch, max_tanh_torch, min_tanh_torch = get_stats(torch_tanh_durations_ns)
        mean_torch_tanh_gbps = (num_bytes / mean_tanh_torch) * 10**9
        min_torch_tanh_gbps = (num_bytes / min_tanh_torch) * 10**9
        max_torch_tanh_gbps = (num_bytes / max_tanh_torch) * 10**9
        res.append(
            {
                "kernel": "arange",
                "SoL % Mean": mean_torch_arange_gbps / mem_bandwidth() * 100,  # 1 write
                "SoL % Min": min_torch_arange_gbps / mem_bandwidth() * 100,  # 1 write
                "SoL % Max": max_torch_arange_gbps / mem_bandwidth() * 100,  # 1 write
                "lib": "torch",
                "n": n,
            }
        )
        res.append(
            {
                "kernel": "tanh",
                "SoL % Mean": 2 * mean_torch_tanh_gbps / mem_bandwidth() * 100,  # 1 read + 1 write, 2x
                "SoL % Min": 2 * min_torch_tanh_gbps / mem_bandwidth() * 100,  # 1 read + 1 write, 2x
                "SoL % Max": 2 * max_torch_tanh_gbps / mem_bandwidth() * 100,  # 1 read + 1 write, 2x
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
