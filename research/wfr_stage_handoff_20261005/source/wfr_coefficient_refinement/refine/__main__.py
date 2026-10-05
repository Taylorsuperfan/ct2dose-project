"""Command-line entry point for the paired coefficient study."""
from pathlib import Path
import argparse
from cwfr import common as io
from .config import RefinementConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("register")
    p.add_argument("--cache", type=Path, required=True); p.add_argument("--parent-work", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True); p.add_argument("--config", type=Path, required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--cache", type=Path, required=True); p.add_argument("--parent-work", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True); p.add_argument("--device", default="cpu")
    p.add_argument("--max-records", type=int, default=32); p.add_argument("--allow-cache-environment-change", action="store_true")
    p = sub.add_parser("train")
    p.add_argument("--work", type=Path, required=True); p.add_argument("--arm", choices=("composition", "profile"), required=True)
    p.add_argument("--device", default="cpu"); p.add_argument("--max-updates", type=int, default=128)
    p.add_argument("--run", type=Path)
    p = sub.add_parser("fork")
    p.add_argument("--work", type=Path, required=True); p.add_argument("--parent-run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True); p.add_argument("--device", required=True)
    p = sub.add_parser("finalize")
    p.add_argument("--work", type=Path, required=True); p.add_argument("--composition-run", type=Path)
    p.add_argument("--profile-run", type=Path)
    p = sub.add_parser("evaluate")
    p.add_argument("--cache", type=Path, required=True); p.add_argument("--parent-work", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True); p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path); p.add_argument("--max-records", type=int, default=100)
    p = sub.add_parser("compare")
    p.add_argument("--cache", type=Path, required=True); p.add_argument("--parent-work", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True); p.add_argument("--hjd-evaluation", type=Path, required=True)
    p.add_argument("--direct-evaluation", type=Path, required=True); p.add_argument("--evaluation", type=Path)
    p.add_argument("--out", type=Path)
    args = parser.parse_args()
    if hasattr(args, "max_updates") and args.max_updates < 0 or hasattr(args, "max_records") and args.max_records < 0:
        parser.error("Use a positive per-call limit, or 0 for no per-call limit.")
    if args.command == "register":
        from .cache import register
        register(args.cache, args.parent_work, args.work, RefinementConfig(**io.read(args.config)).validate())
    elif args.command == "prepare":
        from .cache import prepare
        prepare(args.cache, args.parent_work, args.work, args.device, args.max_records or None, args.allow_cache_environment_change)
    elif args.command == "train":
        from .train import train
        train(args.work, args.arm, args.device, args.max_updates or None, args.run)
    elif args.command == "fork":
        from .train import fork
        fork(args.work, args.parent_run, args.out, args.device)
    elif args.command == "finalize":
        from .evaluation import finalize
        finalize(args.work, args.composition_run, args.profile_run)
    elif args.command == "evaluate":
        from .evaluation import evaluate
        evaluate(args.cache, args.parent_work, args.work, args.device, args.out, args.max_records or None)
    elif args.command == "compare":
        from .compare import compare
        compare(args.cache, args.parent_work, args.work, args.hjd_evaluation, args.direct_evaluation, args.evaluation, args.out)

if __name__ == "__main__":
    main()
