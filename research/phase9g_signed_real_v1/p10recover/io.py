"""Local evidence I/O. All writes belong to a new comparison output directory."""
from pathlib import Path
import hashlib, json, os, platform, sys, tempfile
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def canonical(obj):
    return json.dumps(obj,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':'))


def digest_object(obj):return hashlib.sha256(canonical(obj).encode()).hexdigest()


def read_json(path):
    with Path(path).open(encoding='utf-8') as f:return json.load(f)


def write_json(path,obj,replace=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload=(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    if path.exists() and not replace:
        if path.read_bytes()==payload:return
        raise FileExistsError(f'Refuse to overwrite existing evidence: {path}')
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='._pending_')
    try:
        with os.fdopen(fd,'wb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    if path.read_bytes()!=payload:raise IOError(f'Readback mismatch: {path}')


def save_arrays(path,arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='._pending_',suffix='.npz')
    os.close(fd)
    try:
        np.savez_compressed(tmp,**arrays)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    # Verify readable archive and finite numeric contents.
    with np.load(path,allow_pickle=False) as a:
        for k,v in arrays.items():
            if k not in a or not np.array_equal(a[k],v):raise IOError('Array readback mismatch')


def environment():
    import torch
    return dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,torch=str(torch.__version__),
                cuda_runtime=torch.version.cuda,cuda_available=torch.cuda.is_available(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                num_threads=torch.get_num_threads())


def code_hashes():
    return {p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
