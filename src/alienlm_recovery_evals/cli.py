from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from importlib.resources import files
from pathlib import Path
from typing import Iterable


TARGET_HELP = """\
Targets:
  list                         Show this target list
  smoke                        Syntax/entrypoint smoke checks
  outputs [OUT_ROOT]           List checkpoints under an output root

  o1-frequency                 O1 passive frequency/ngram recovery
  o2-known-ngram               O2 known-plaintext ngram single run
  o2-known-sweep               O2 known-plaintext ngram sweep
  o2-known-summary             O2 known-plaintext result summaries
  o2-llm-single                O2 LLM decoding single run
  o2-llm-sweep                 O2 LLM decoding progressive sweep
  o2-mt-train                  O2 MT/NLLB translator training
  o3-weight                    O3 weight-space mapping recovery

  data-volume-build            Build 50k/150k Magpie subsets
  data-volume-train <50k|150k> Train a data-volume ablation model
  data-volume-eval [MODEL]     Evaluate a data-volume model
  data-volume-summary          Summarize data-volume eval runs
  token-length                 Original-tokenizer length summary
  token-length-original-alien  Original vs Alien tokenizer length summary
  vocab-overlap                Tokenizer vocab overlap summary
  failure-analysis             GSM8K/MBPP failure analysis
  long-truthfulqa <model> [gpu] TruthfulQA generation run
  long-longbench <model> [gpu] LongBench core run
  long-longbench-all [gpu]     LongBench core for all configured models
  long-summarize               LongBench result summary
"""

RESOURCE_NAMES = ("README.md", "scripts", "recovery-attacks", "robustness")
IGNORED_NAMES = {
    ".cache",
    "__pycache__",
    "data",
    "data-prepared",
    "dist",
    "logs",
    "outputs",
    "results",
    "wandb",
}
IGNORED_SUFFIXES = (".log", ".pyc", ".pyo", ".jsonl")


def is_workspace_root(path: Path) -> bool:
    return (
        (path / "scripts" / "run_eval.sh").is_file()
        and (path / "recovery-attacks").is_dir()
        and (path / "robustness").is_dir()
    )


def find_workspace_root(start: Path) -> Path | None:
    start = start.resolve()
    candidates = [start, *start.parents]
    for candidate in candidates:
        if is_workspace_root(candidate):
            return candidate
    return None


def source_checkout_root() -> Path | None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if is_workspace_root(parent):
            return parent
    return None


def packaged_resources():
    try:
        root = files("alienlm_recovery_evals").joinpath("resources")
    except ModuleNotFoundError:
        return None
    return root if root.is_dir() else None


def resolve_workspace_root(explicit_root: str | None) -> Path | None:
    if explicit_root:
        root = Path(explicit_root).expanduser().resolve()
        if not is_workspace_root(root):
            raise SystemExit(f"Not an AlienLM recovery-evals workspace: {root}")
        return root

    env_root = os.environ.get("ALIENLM_RECOVERY_EVALS_ROOT")
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if not is_workspace_root(root):
            raise SystemExit(f"ALIENLM_RECOVERY_EVALS_ROOT is not a workspace: {root}")
        return root

    cwd_root = find_workspace_root(Path.cwd())
    if cwd_root:
        return cwd_root

    return source_checkout_root()


def should_skip(name: str) -> bool:
    return name in IGNORED_NAMES or name.endswith(IGNORED_SUFFIXES)


def copy_traversable(src, dst: Path, force: bool) -> None:
    if should_skip(src.name):
        return

    if src.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
        for child in src.iterdir():
            copy_traversable(child, dst / child.name, force=force)
        return

    if dst.exists() and not force:
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as in_handle, dst.open("wb") as out_handle:
        shutil.copyfileobj(in_handle, out_handle)
    if dst.suffix == ".sh":
        dst.chmod(dst.stat().st_mode | 0o111)


def copy_path(src: Path, dst: Path, force: bool) -> None:
    if should_skip(src.name):
        return

    if src.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
        for child in src.iterdir():
            copy_path(child, dst / child.name, force=force)
        return

    if dst.exists() and not force:
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    if dst.suffix == ".sh":
        dst.chmod(dst.stat().st_mode | 0o111)


def init_workspace(destination: Path, force: bool) -> int:
    destination = destination.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()) and not force:
        print(f"Destination is not empty: {destination}", file=sys.stderr)
        print("Use --force to merge/overwrite packaged files.", file=sys.stderr)
        return 2
    destination.mkdir(parents=True, exist_ok=True)

    source_root = source_checkout_root()
    if source_root:
        for name in RESOURCE_NAMES:
            copy_path(source_root / name, destination / name, force=force)
    else:
        resource_root = packaged_resources()
        if resource_root is None:
            print("Packaged resources are not available in this installation.", file=sys.stderr)
            return 2
        for name in RESOURCE_NAMES:
            copy_traversable(resource_root.joinpath(name), destination / name, force=force)

    print(f"Initialized AlienLM recovery-evals workspace: {destination}")
    print(f"Run: alienlm-recovery-evals --root {destination} smoke")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alienlm-recovery-evals",
        description="Run or scaffold AlienLM recovery and robustness evaluations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=TARGET_HELP,
    )
    parser.add_argument(
        "--root",
        help="Existing recovery-evals workspace. Defaults to ALIENLM_RECOVERY_EVALS_ROOT, cwd parents, or source checkout.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Set DRY_RUN=1 before dispatching to scripts/run_eval.sh.",
    )
    parser.add_argument("target", nargs="?", default="list")
    parser.add_argument("target_args", nargs=argparse.REMAINDER)
    return parser


def build_init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alienlm-recovery-evals init",
        description="Create a writable AlienLM recovery-evals workspace from packaged templates.",
    )
    parser.add_argument("destination", nargs="?", default="alienlm-recovery-evals-work")
    parser.add_argument("--force", action="store_true", help="Overwrite files that already exist.")
    return parser


def dispatch(root: Path, target: str, target_args: Iterable[str], dry_run: bool) -> int:
    script = root / "scripts" / "run_eval.sh"
    env = os.environ.copy()
    env["ALIENLM_RECOVERY_EVALS_ROOT"] = str(root)
    if dry_run:
        env["DRY_RUN"] = "1"
    cmd = [str(script), target, *target_args]
    return subprocess.run(cmd, cwd=root, env=env).returncode


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if argv and argv[0] == "init":
        args = build_init_parser().parse_args(argv[1:])
        return init_workspace(Path(args.destination), force=args.force)

    parser = build_parser()
    args = parser.parse_args(argv)
    target_args = list(args.target_args)
    if target_args and target_args[0] == "--":
        target_args = target_args[1:]

    if args.target in {"list", "help", "-h", "--help"}:
        root = resolve_workspace_root(args.root)
        if root:
            return dispatch(root, "list", [], dry_run=args.dry_run)
        print(TARGET_HELP.rstrip())
        print()
        print("No workspace found. Create one with:")
        print("  alienlm-recovery-evals init ./alienlm-recovery-evals-work")
        return 0

    root = resolve_workspace_root(args.root)
    if root is None:
        print("No AlienLM recovery-evals workspace found.", file=sys.stderr)
        print("Run `alienlm-recovery-evals init ./alienlm-recovery-evals-work` first,", file=sys.stderr)
        print("or pass `--root /path/to/AlienLM-recovery-evals`.", file=sys.stderr)
        return 2

    return dispatch(root, args.target, target_args, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
