"""Non-destructive records and resumable local research runs."""
from __future__ import annotations
from pathlib import Path
from contextlib import contextmanager
import datetime, hashlib, json, os, platform, random, tempfile, uuid
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, obj, *, replace=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        if read(path) != obj: raise RuntimeError(f'Existing record differs: {path}. Preserve it and use a new run.')
        return
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    raw = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    try:
        with tmp.open('x', encoding='utf-8') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists(): tmp.unlink()


def safe(root, rel):
    root = Path(root).resolve(); rel = Path(rel)
    if rel.is_absolute() or '..' in rel.parts: raise ValueError('Unsafe relative path.')
    p = (root/rel).resolve()
    if not p.is_relative_to(root): raise ValueError('Path escapes its declared root.')
    return p


def separate(out, *inputs):
    out = Path(out).resolve()
    for src in inputs:
        src = Path(src).resolve()
        if out.is_relative_to(src) or src.is_relative_to(out):
            raise ValueError(f'Output and input trees must be separate: {out}, {src}')


def source_identity():
    root = Path(__file__).parent
    return {p.name: sha(p) for p in sorted(root.glob('*.py'))}


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


@contextmanager
def single_writer(out):
    import fcntl
    name = 'cwfr_' + hashlib.sha256(str(Path(out).resolve()).encode()).hexdigest() + '.lock'
    with (Path(tempfile.gettempdir())/name).open('a') as f:
        try: fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Another local process is writing this run.')
        try: yield
        finally: fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def environment(device='cpu'):
    import torch
    return {'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'device': str(device),
            'gpu': torch.cuda.get_device_name(device) if str(device).startswith('cuda') else None,
            'threads': torch.get_num_threads()}


def seed_all(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(int(os.environ.get('CWFR_CPU_THREADS','2')))
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state():
    import torch
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    import torch
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state['cuda']:
        if not torch.cuda.is_available(): raise RuntimeError('Cannot restore CUDA RNG state on CPU without an explicit fork.')
        torch.cuda.set_rng_state_all(state['cuda'])


def save_checkpoint(out, state):
    import torch
    out = Path(out); folder = out/'checkpoints.local'; folder.mkdir(exist_ok=True)
    path = folder/f"step_{state['step']:07d}_{uuid.uuid4().hex[:8]}.pt"
    tmp = path.with_suffix('.tmp')
    with tmp.open('xb') as f: torch.save(state, f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)
    return {'path': str(path.relative_to(out)), 'sha256': sha(path), 'step': state['step']}


def load_checkpoint(out, pointer):
    """Load only a hash-verified checkpoint created by this package, never arbitrary files."""
    import torch
    path = safe(out,pointer['path'])
    if sha(path) != pointer['sha256']: raise ValueError('Checkpoint byte hash mismatch.')
    # Full trusted local state contains Python/NumPy RNG objects.
    return torch.load(path, map_location='cpu', weights_only=False)


def finish(out, status, **extra):
    out = Path(out)
    files = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*'))
             if p.is_file() and p.name not in ('FILES.json','COMPLETE.json')}
    write(out/'FILES.json',files,replace=True)
    write(out/'COMPLETE.json',{'status':status,'files_sha256':sha(out/'FILES.json'),**extra},replace=True)


def verify_finish(out, status):
    out = Path(out); c=read(out/'COMPLETE.json')
    if c['status']!=status or c['files_sha256']!=sha(out/'FILES.json'): raise ValueError('Completion receipt differs.')
    for rel,h in read(out/'FILES.json').items():
        if sha(safe(out,rel))!=h: raise ValueError(f'Changed output: {rel}')
    return c
