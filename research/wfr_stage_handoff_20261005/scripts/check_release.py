"""Check this source/result handoff; no models, private arrays, or network access."""
from pathlib import Path
import argparse
import ast
import base64
import hashlib
import io
import json
import re
import subprocess
import zipfile

IGNORED_DIRS = {'__pycache__', '.pytest_cache', '.git', '.venv', 'venv'}
FORBIDDEN_SUFFIXES = {'.pt', '.pth', '.ckpt', '.npy', '.npz', '.dcm', '.nii', '.gz', '.pkl', '.pickle'}
SECRET = re.compile(r'gh[pousr]_[A-Za-z0-9]{24,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9]{28,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')
LONG_DECIMAL = re.compile(r'(?<![0-9A-Za-z])[0-9]{28,}(?![0-9A-Za-z])')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def relative_files(root):
    files = {}
    for p in root.rglob('*'):
        rel = p.relative_to(root)
        if any(x in IGNORED_DIRS for x in rel.parts):
            continue
        if p.is_symlink():
            raise RuntimeError('Symlinks are not allowed: ' + rel.as_posix())
        if p.is_file():
            files[rel.as_posix()] = p
    return files


def review_content(name, data):
    p = Path(name)
    if p.suffix.lower() in FORBIDDEN_SUFFIXES or '.private' in name or p.name == '.env':
        raise RuntimeError('Unexpected runtime/private file: ' + name)
    if any(part.endswith('.local') for part in p.parts):
        raise RuntimeError('Runtime directory found: ' + name)
    if p.suffix.lower() in {'.py', '.md', '.txt', '.csv', '.json', '.ipynb', '.sh'}:
        text = data.decode('utf-8')
        if SECRET.search(text) or LONG_DECIMAL.search(text):
            raise RuntimeError('Potential credential or long decimal identifier: ' + name)
    if p.suffix.lower() == '.ipynb':
        nb = json.loads(data)
        for cell in nb.get('cells', []):
            if cell.get('attachments'):
                raise RuntimeError('Notebook attachments require separate review: ' + name)
            if cell.get('cell_type') != 'code':
                continue
            if cell.get('outputs') or cell.get('execution_count') is not None:
                raise RuntimeError('Executed notebook output found: ' + name)
            source = ''.join(cell.get('source', []))
            tree = ast.parse(source)
            literals = {}
            for node in tree.body:
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            literals[target.id] = node.value.value
            if 'PAYLOAD_BASE64' in literals:
                raw = base64.b64decode(literals['PAYLOAD_BASE64'], validate=True)
                if sha(raw) != literals.get('PAYLOAD_SHA256'):
                    raise RuntimeError('Embedded payload hash mismatch: ' + name)
                with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                    for info in archive.infolist():
                        rel = Path(info.filename)
                        if rel.is_absolute() or '..' in rel.parts or info.file_size > 10_000_000:
                            raise RuntimeError('Unsafe payload entry: ' + info.filename)
                        if not info.is_dir():
                            review_content(info.filename, archive.read(info))


def verify(root):
    root = Path(root).resolve()
    manifest_path = root / 'RELEASE_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    expected = {item['path']: item for item in manifest['files']}
    actual = relative_files(root)
    if set(actual) != set(expected) | {'RELEASE_MANIFEST.json'}:
        missing = set(expected) - set(actual)
        extra = set(actual) - set(expected) - {'RELEASE_MANIFEST.json'}
        raise RuntimeError(f'File-set mismatch; missing={sorted(missing)}, extra={sorted(extra)}')
    for name, item in expected.items():
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts:
            raise RuntimeError('Unsafe manifest path: ' + name)
        data = actual[name].read_bytes()
        if len(data) != item['bytes'] or sha(data) != item['sha256']:
            raise RuntimeError('Changed file: ' + name)
        review_content(name, data)
    print(f'PASS: {len(expected)} listed files match their recorded bytes.')
    print('PASS: no prohibited array/weight files or executed notebook outputs detected.')
    print('LIMIT: heuristic content checks do not establish publication permission or audit Git history.')
    return manifest


def verify_staged(root, repo):
    root, repo = Path(root).resolve(), Path(repo).resolve()
    top = Path(subprocess.check_output(['git','-C',str(repo),'rev-parse','--show-toplevel'], text=True).strip()).resolve()
    if top != repo:
        raise RuntimeError('Provide the repository root, not a subdirectory.')
    prefix = root.relative_to(repo).as_posix()
    changed = subprocess.check_output(['git','-C',str(repo),'diff','--cached','--name-only','-z']).split(b'\0')
    names = [s.decode('utf-8') for s in changed if s]
    if not names:
        raise RuntimeError('No staged changes. Nothing to commit.')
    outside = [s for s in names if not s.startswith(prefix + '/')]
    if outside:
        raise RuntimeError('Unrelated staged paths; review before committing: ' + repr(outside))
    # Inspect actual index bytes, not only the current working tree.
    for name, p in relative_files(root).items():
        index_name = prefix + '/' + name
        data = subprocess.check_output(['git','-C',str(repo),'show', ':'+index_name])
        if data != p.read_bytes():
            raise RuntimeError('Index and checked working copy differ: ' + index_name)
    print('PASS: all staged changes are inside this handoff and index bytes match the checked copy.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--staged', action='store_true')
    p.add_argument('--repo', type=Path)
    args = p.parse_args()
    verify(args.root)
    if args.staged:
        if args.repo is None:
            p.error('--repo is required with --staged')
        verify_staged(args.root, args.repo)

if __name__ == '__main__':
    main()
