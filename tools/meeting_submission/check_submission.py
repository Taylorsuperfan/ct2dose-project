"""Read-only submission check: preserved sources, figures, and exact staged blobs.

Uses Python standard library only. Does not train, execute discovered notebooks,
change Git, or certify privacy/permission/scientific conclusions.
"""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys

OWN = 'tools/meeting_submission/RELEASE_FILES.json'
NEW_SCOPES = ('docs/thesis/meeting_20260918',
              'docs/thesis/submission_20260918', 'tools/meeting_submission')
IGNORED = {'__pycache__', '.pytest_cache', '.DS_Store'}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def safe(root, relative):
    p = PurePosixPath(relative)
    if not relative or p.is_absolute() or '..' in p.parts or '\\' in relative:
        raise ValueError('Unsafe path: ' + relative)
    target = root.joinpath(*p.parts)
    for q in (target, *target.parents):
        if q == root.parent:
            break
        if q.is_symlink():
            raise ValueError('Symlink: ' + str(q))
    return target


def import_checked_helper(path):
    spec = importlib.util.spec_from_file_location('unchanged_visual_check', path)
    module = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old
    return module


def verify_tree(root, manifest):
    errors = []
    expected = manifest['files']
    # Validate every expected byte before importing any existing helper.
    for relative, digest in expected.items():
        try:
            p = safe(root, relative)
            if not p.is_file():
                raise ValueError('Missing')
            if sha(p.read_bytes()) != digest:
                raise ValueError('Different bytes from supplied snapshot')
        except (ValueError, OSError) as exc:
            errors.append(relative + ': ' + str(exc))
    if errors:
        return errors
    old = import_checked_helper(root/'tools/pilot_v1_visual_release/check_release.py')
    old_manifest = json.loads((root/'tools/pilot_v1_visual_release/RELEASE_FILES.json').read_text())
    errors.extend(old.verify_tree(root, old_manifest))
    content_check = old._legacy(root).obvious_content_problem
    # The reader-facing source snapshot has its own original manifest.
    mroot = root/'docs/thesis/meeting_20260918'
    mfiles = json.loads((mroot/'FILES.json').read_text())
    for relative, digest in mfiles.items():
        if sha(safe(mroot, relative).read_bytes()) != digest:
            errors.append('Reader-facing archive mismatch: ' + relative)
    for scope in NEW_SCOPES:
        for p in (root/scope).rglob('*'):
            relative = p.relative_to(root).as_posix()
            if any(x in IGNORED for x in p.relative_to(root).parts):
                continue
            if p.is_symlink():
                errors.append(relative + ': unreviewed symlink')
            elif p.is_file():
                if relative not in expected and relative != OWN:
                    errors.append(relative + ': extra file; not automatically deleted')
                    continue
                # PNG files are accepted only through the above exact hash list.
                if p.suffix == '.png':
                    continue
                try:
                    issue = content_check(relative, p.read_bytes())
                    if issue:
                        errors.append(relative + ': ' + issue)
                except (UnicodeError, ValueError) as exc:
                    errors.append(relative + ': ' + str(exc))
    return errors


def verify_staged(root, manifest):
    expected = dict(manifest['files'])
    expected[OWN] = sha((root/OWN).read_bytes())
    # --no-renames makes old paths/deletions visible instead of hiding them.
    changed = subprocess.check_output([
        'git', '-C', str(root), 'diff', '--cached', '--no-renames',
        '--name-only', '-z'], text=False).decode().split('\0')
    errors = []
    for name in filter(None, changed):
        if name not in expected:
            errors.append(name + ': staged outside this submission; review separately')
    # All planned files must exist in the index, including already committed ones.
    # This catches files omitted by ignore rules or an incomplete git add.
    entries = subprocess.check_output([
        'git', '-C', str(root), 'ls-files', '--stage', '-z'], text=False)
    modes = {}
    for entry in entries.decode().split('\0'):
        if not entry:
            continue
        meta, name = entry.split('\t', 1)
        mode, _oid, stage = meta.split()
        if stage == '0':
            modes[name] = mode
    for name, digest in expected.items():
        if modes.get(name) not in {'100644', '100755'}:
            errors.append(name + ': not a normal indexed file (missing, deletion or conflict)')
            continue
        try:
            raw = subprocess.check_output(
                ['git', '-C', str(root), 'show', ':'+name],
                stderr=subprocess.DEVNULL)
            if sha(raw) != digest:
                errors.append(name + ': indexed bytes differ from supplied snapshot')
        except subprocess.CalledProcessError:
            errors.append(name + ': unreadable index entry')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--staged', action='store_true')
    args = parser.parse_args()
    root = args.repo.resolve(strict=True)
    manifest = json.loads((root/OWN).read_text(encoding='utf-8'))
    errors = verify_tree(root, manifest)
    if args.staged and not errors:
        actual_root = Path(subprocess.check_output(
            ['git', '-C', str(root), 'rev-parse', '--show-toplevel'], text=True).strip()).resolve()
        if actual_root != root:
            errors.append('Use the existing repository root, not a subdirectory.')
        else:
            errors.extend(verify_staged(root, manifest))
    if errors:
        raise SystemExit('STOP: no changes were made.\n' + '\n'.join(errors))
    print(f"PASS: {len(manifest['files'])+1} submission files match their listed bytes.")
    print('PASS: 196 earlier source/presentation files preserved; 7 latest PNG/SVG pairs included.')
    print('PASS: original training and visualization notebook payload checks passed.')
    if args.staged:
        print('PASS: all intended files exist in the Git index; no unrelated staged paths.')
    print('LIMIT: no publication permission, full privacy/Git-history audit, model validation or external writes.')


if __name__ == '__main__':
    main()
