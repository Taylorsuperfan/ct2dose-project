"""Immutable contracts, atomic progress snapshots, and explicit single-writer scope."""
from pathlib import Path
from contextlib import contextmanager
import hashlib, json, os, tempfile, uuid, random, platform, sys
import numpy as np
import torch
from p10recover.io import sha, digest_object, read_json, write_json, save_arrays

SCOPE = 'REAL_PHASE9G_ERROR_CORRECTION_PILOT_NOT_WATER_NOT_BLIND_FINAL_TEST'

def code_hashes():
    root=Path(__file__).resolve().parent.parent
    return {str(p.relative_to(root)):sha(p) for pkg in ['pg9learn','p10recover']
            for p in sorted((root/pkg).glob('*.py'))}

def device_identity(device):
    d=torch.device(device)
    return dict(python=sys.version.split()[0],torch=str(torch.__version__),numpy=np.__version__,
        cuda=torch.version.cuda,device_type=d.type,
        gpu=torch.cuda.get_device_name(d) if d.type=='cuda' else None,
        cpu_threads=torch.get_num_threads(),deterministic=torch.are_deterministic_algorithms_enabled(),
        warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32)

def runtime_settings(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True,warn_only=True)

def rng_state():
    # Restriction-compatible state: tensors and Python primitives only.
    n=np.random.get_state()
    return dict(python=random.getstate(),numpy=(n[0],n[1].tolist(),n[2],n[3],n[4]),
                torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])

def restore_rng(s):
    random.setstate(s['python'])
    n=s['numpy'];np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(s['torch'].cpu())
    if s['cuda']:torch.cuda.set_rng_state_all([v.cpu() for v in s['cuda']])

@contextmanager
def single_writer(folder):
    """Advisory protection in ONE runtime. Not a distributed Drive lock."""
    import fcntl
    p=Path(folder).resolve()
    key=hashlib.sha256(str(p).encode()).hexdigest()
    lock=Path(tempfile.gettempdir())/('pg9learn_'+key+'.lock')
    with lock.open('a') as h:
        try:fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('This output has an active writer in this runtime.')
        try:yield
        finally:fcntl.flock(h,fcntl.LOCK_UN)

def checkpoint_write(run,state,label):
    folder=Path(run)/'checkpoints';folder.mkdir(parents=True,exist_ok=True)
    name=f'{label}_{uuid.uuid4().hex[:10]}.pt';final=folder/name
    fd,tmp=tempfile.mkstemp(dir=folder,prefix='._checkpoint_',suffix='.pt');os.close(fd)
    try:
        torch.save(state,tmp)
        with open(tmp,'rb') as f:os.fsync(f.fileno())
        expected=sha(tmp);os.replace(tmp,final)
        if sha(final)!=expected:raise IOError('Checkpoint readback hash mismatch.')
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    return dict(path=str(final.relative_to(run)),sha256=expected,global_step=int(state['step']))

def checkpoint_read(run,pointer):
    run=Path(run).resolve();p=(run/pointer['path']).resolve()
    if not p.is_relative_to(run) or sha(p)!=pointer['sha256']:raise ValueError('Checkpoint pointer/hash mismatch.')
    return torch.load(p,map_location='cpu',weights_only=True)

def status(run,name,**kwargs):
    write_json(Path(run)/'run_status.json',dict(status=name,scope=SCOPE,scientific_task_status='PENDING_HUMAN_REVIEW',**kwargs),replace=True)

def log(run,text):
    print(text,flush=True)
    with (Path(run)/'training.log').open('a',encoding='utf-8') as h:h.write(text+'\n');h.flush()
