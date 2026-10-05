"""Copy this source release to an existing repository, after a non-overwriting dry run."""
from pathlib import Path
import argparse
import hashlib
import shutil
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()
    repo = args.repo.resolve(strict=True)
    top = Path(subprocess.check_output(["git", "-C", str(repo), "rev-parse", "--show-toplevel"], text=True).strip()).resolve()
    if repo != top:
        raise ValueError("Pass the repository root, not a nested or copied folder.")
    source = Path(__file__).resolve().parents[1]
    destination = repo / "research" / "wfr_coefficient_refinement"
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Source and destination overlap.")
    files = [f for f in sorted(source.rglob("*")) if f.is_file()
             and "__pycache__" not in f.parts and f.suffix != ".pyc"]
    added = []; identical = []
    for f in files:
        if f.is_symlink():
            raise ValueError("Do not install symlinked files.")
        target = destination / f.relative_to(source)
        if not target.resolve().is_relative_to(repo):
            raise ValueError("Destination escapes the repository through a symlink.")
        if target.exists():
            if not target.is_file() or hashlib.sha256(target.read_bytes()).digest() != hashlib.sha256(f.read_bytes()).digest():
                raise FileExistsError("Different file already exists: " + str(target))
            identical.append(target)
        else:
            added.append((f, target))
    print("Repository:", repo)
    print("Destination:", destination)
    print("New files:", len(added), "Identical files retained:", len(identical))
    for _, target in added:
        print(target.relative_to(repo))
    if not args.apply:
        print("CHECK ONLY. No repository files were changed. Review, then add --apply.")
        return
    for f, target in added:
        target.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation closes the common overwrite gap after the dry check.
        with target.open("xb") as stream:
            stream.write(f.read_bytes())
    print("Source installed. Nothing was staged, committed or pushed.")

if __name__ == "__main__":
    main()
