"""Run the non-downloading Stage A environment audit."""

import argparse
from pathlib import Path

from safecompress_aya.preflight import audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("results/raw/preflight"))
    args = parser.parse_args()
    print(audit(args.config, args.output_root))


if __name__ == "__main__":
    main()
