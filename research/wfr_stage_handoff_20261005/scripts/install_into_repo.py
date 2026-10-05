"""Add this handoff to an existing Git checkout. Dry-run by default; never push."""
from pathlib import Path
import argparse
import importlib.util
import json
import shutil
import subprocess
import sys

NAME = 'wfr_stage_handoff_20261005'

def install(repo, apply=False, source=None):
    repo = Path(repo).expanduser().resolve(strict=True)
    actual = Path(subprocess.check_output(['git','-C',str(repo),'rev-parse','--show-toplevel'],text=True).strip()).resolve()
    if actual != repo:
        raise RuntimeError('The destination must be the existing repository root.')
    if source is None:
        base = Path(__file__).resolve().parents[1]
        source = base if (base/'RELEASE_MANIFEST.json').is_file() else base/'repo'/'research'/NAME
    source = Path(source).resolve(strict=True)
    spec = importlib.util.spec_from_file_location('release_check',source/'scripts/check_release.py')
    checker = importlib.util.module_from_spec(spec); spec.loader.exec_module(checker)
    checker.verify(source)
    destination = repo / 'research' / NAME
    for p in [repo/'research', destination]:
        if p.is_symlink() or (p.exists() and not p.is_dir()):
            raise RuntimeError('Unsafe destination: ' + str(p))
    files = checker.relative_files(source)
    if destination.exists():
        extras = set(checker.relative_files(destination)) - set(files)
        if extras:
            raise RuntimeError('Unexpected files already in destination: ' + repr(sorted(extras)))
    new, same = [], []
    for rel, p in files.items():
        out = destination / rel
        for ancestor in [out, *out.parents]:
            if ancestor == repo: break
            if ancestor.is_symlink():
                raise RuntimeError('Destination symlink: ' + str(ancestor))
            if ancestor != out and ancestor.exists() and not ancestor.is_dir():
                raise RuntimeError('A destination parent is not a directory: ' + str(ancestor))
        if out.exists():
            if not out.is_file() or out.read_bytes() != p.read_bytes():
                raise RuntimeError('Existing content differs; nothing copied: ' + rel)
            same.append(rel)
        else:
            new.append(rel)
    print('Repository:', repo)
    print('Destination:', destination)
    print('New files:', len(new), 'Identical files retained:', len(same))
    for name in sorted(new): print('ADD', name)
    if not apply:
        print('CHECK ONLY. Re-run with --apply after reviewing the list.')
        return len(new), len(same)
    for rel in sorted(new):
        out = destination/rel; out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('xb') as stream:
            stream.write(files[rel].read_bytes())
    checker.verify(destination)
    print('Imported. No files were staged, committed, or pushed.')
    return len(new), len(same)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    a = parser.parse_args()
    install(a.repo, a.apply)
