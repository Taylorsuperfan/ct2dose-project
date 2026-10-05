"""Small, auditable file receipts. Completed results are never silently overwritten."""
from pathlib import Path
import hashlib
import os
import uuid
import numpy as np
import torch
from cwfr import common as io


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def source_identity():
    root = Path(__file__).parent
    return {p.name: io.sha(p) for p in sorted(root.glob("*.py"))}


def state_digest(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        a = tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def save_npz(path, arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("Refusing to replace an array archive: " + str(path))
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with tmp.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)


def save_weights(path, state):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("Refusing to replace saved weights: " + str(path))
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with tmp.open("xb") as stream:
        torch.save({k: v.detach().cpu().clone() for k, v in state.items()}, stream)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)


class Receipt:
    """Verify the ledger and only the files actually read, not unrelated medical arrays."""
    def __init__(self, folder, status):
        self.root = Path(folder).resolve()
        self.complete = io.read(self.root / "COMPLETE.json")
        require(self.complete["status"] == status, "Unexpected completion status: " + str(folder))
        require(io.sha(self.root / "FILES.json") == self.complete["files_sha256"], "Changed file ledger.")
        self.files = io.read(self.root / "FILES.json")
        self.used = {
            str(self.root / "COMPLETE.json"): io.sha(self.root / "COMPLETE.json"),
            str(self.root / "FILES.json"): io.sha(self.root / "FILES.json"),
        }

    def file(self, rel):
        path = io.safe(self.root, rel)
        require(rel in self.files and io.sha(path) == self.files[rel], "Changed or unlisted input: " + str(path))
        self.used[str(path)] = self.files[rel]
        return path

    def json(self, rel):
        return io.read(self.file(rel))

    def recheck(self):
        for path, digest in self.used.items():
            require(io.sha(path) == digest, "An input changed while it was being used: " + path)


def assert_outputs_separate(work, parent, data):
    io.separate(work, parent, data.path, data.old)
