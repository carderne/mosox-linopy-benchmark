import argparse
from pathlib import Path

from model import build_model


def _data_for_size(size: str):
    if size == "lg":
        from data_lg import DATA, PARAM_DIMS
    else:
        from data_sm import DATA, PARAM_DIMS
    return DATA, PARAM_DIMS


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Solve the Atlantis OSeMOSYS Linopy model"
    )
    parser.add_argument("--size", choices=["sm", "lg"], default="sm", help="model size")
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="build the Linopy model without solving",
    )
    args = parser.parse_args()

    data, param_dims = _data_for_size(args.size)
    model = build_model(data, param_dims)
    if args.generate_only:
        model.to_file(
            Path("/tmp") / f"lino_{args.size}.mps", io_api="mps", progress=False
        )
        return 0

    status, condition = model.solve(
        solver_name="highs", log_to_console=False, progress=False
    )
    if condition != "optimal":
        raise RuntimeError(f"Solve failed: {status}/{condition}")
    print(model.objective.value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
