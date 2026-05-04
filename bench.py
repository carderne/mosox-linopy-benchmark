import argparse
import re
import subprocess
import sys
import time
from typing import Callable, TypeVar

EPS = 0.1
T = TypeVar("T")


def run(cmd: list[str]) -> str:
    return subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    ).stdout


def timed(label: str, fn: Callable[[], T]) -> tuple[T, float]:
    print(f"Running {label}...", file=sys.stderr)
    start = time.time()
    value = fn()
    elapsed = time.time() - start
    return value, elapsed


def parse_gmpl_objective(output: str) -> float:
    value = None
    for line in output.splitlines():
        parts = line.split()
        for i in range(len(parts) - 2):
            if parts[i] == "obj" and parts[i + 1] == "=":
                value = parts[i + 2]
    if value is None:
        raise RuntimeError("Could not parse GMPL objective")
    return float(value)


def parse_mosox_objective(output: str) -> float:
    match = re.findall(r"Objective value:\s*([^\s]+)", output)
    if not match:
        raise RuntimeError("Could not parse Mosox objective")
    return float(match[-1])


SIZES = {
    "sm": "gmpl/data_sm.dat",
    "lg": "gmpl/data_lg.dat",
}


def gmpl_generate(size: str) -> None:
    dat = SIZES[size]
    run(
        [
            "glpsol",
            "-m",
            "gmpl/model.mod",
            "-d",
            dat,
            "--check",
            "--wfreemps",
            f"/tmp/gmpl_{size}.mps",
        ]
    )


def gmpl_solve(size: str) -> float:
    # Also writes the LP file for lino.py to consume.
    dat = SIZES[size]
    return parse_gmpl_objective(run(["glpsol", "-m", "gmpl/model.mod", "-d", dat]))


def lino_generate(size: str) -> None:
    run(["uv", "run", "python", "lino/main.py", "--size", size, "--generate-only"])


def lino_solve(size: str) -> float:
    return float(run(["uv", "run", "python", "lino/main.py", "--size", size]).strip())


def mosox_generate(size: str) -> None:
    dat = SIZES[size]
    run(["mosox", "compile", "gmpl/model.mod", dat, "-o", f"/tmp/mosox_{size}.mps"])


def mosox_solve(size: str) -> float:
    dat = SIZES[size]
    return parse_mosox_objective(run(["mosox", "solve", "gmpl/model.mod", dat]))


def bench_pair(
    name: str, generate_fn: Callable[[], None], solve_fn: Callable[[], float]
) -> tuple[float, float, float, float]:
    _, gen = timed(f"{name} matrix", generate_fn)
    value, full = timed(f"{name} full", solve_fn)
    solve_only = full - gen
    return value, full, gen, solve_only


def compare(name: str, value: float, gmpl: float) -> bool:
    diff = abs(value - gmpl)
    if diff <= EPS:
        return True

    print(
        f"ERROR: {name} DOES NOT MATCH GMPL within eps={EPS} (diff={diff:.12g})",
        file=sys.stderr,
    )
    print(f"ERROR: {name}: {value:.12g}", file=sys.stderr)
    print(f"ERROR: GMPL: {gmpl:.12g}", file=sys.stderr)
    return False


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("runs must be an integer") from None
    if parsed <= 0:
        raise argparse.ArgumentTypeError("runs must be greater than 0")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark GMPL, Linopy, and Mosox on OSeMOSYS Atlantis"
    )
    parser.add_argument(
        "runs",
        nargs="?",
        default=1,
        type=positive_int,
        help="number of runs to average (default: 1)",
    )
    parser.add_argument(
        "--size", choices=SIZES, default="sm", help="model size: sm or lg (default: sm)"
    )
    args = parser.parse_args()

    print("Benchmarking OSeMOSYS Atlantis model", file=sys.stderr)
    print(f"Objective tolerance: eps={EPS}", file=sys.stderr)
    print(f"Runs: {args.runs}", file=sys.stderr)
    print(f"Size: {args.size}", file=sys.stderr)

    timings = {"GMPL": [], "Linopy": [], "Mosox": []}
    ok = True

    for i in range(args.runs):
        if args.runs > 1:
            print(f"Iteration {i + 1}/{args.runs}", file=sys.stderr)

        gmpl, *gmpl_times = bench_pair(
            "GMPL", lambda: gmpl_generate(args.size), lambda: gmpl_solve(args.size)
        )
        lino, *lino_times = bench_pair(
            "Linopy", lambda: lino_generate(args.size), lambda: lino_solve(args.size)
        )
        mosox, *mosox_times = bench_pair(
            "Mosox", lambda: mosox_generate(args.size), lambda: mosox_solve(args.size)
        )

        print("Checking objectives...", file=sys.stderr)
        ok &= compare("Linopy", lino, gmpl)
        ok &= compare("Mosox", mosox, gmpl)

        timings["GMPL"].append(gmpl_times)
        timings["Linopy"].append(lino_times)
        timings["Mosox"].append(mosox_times)

    print("tool,full_s,matrix_s,solve_s")
    for name, rows in timings.items():
        avg = [sum(row[i] for row in rows) / len(rows) for i in range(3)]
        print(f"{name},{avg[0]:.3f},{avg[1]:.3f},{avg[2]:.3f}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
