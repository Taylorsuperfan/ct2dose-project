"""Read-only release/index preflight. Not a full privacy or permission audit."""
from pathlib import Path, PurePosixPath
import argparse, ast, base64, hashlib, io, json, re, subprocess, sys, zipfile

SCOPES = (
    "research/phase10d_recovered_comparison",
    "research/phase9g_signed_real_v1",
    "docs/thesis/updates/20260917_real_pilot_v1",
    "tools/pilot_v1_release",
)
IGNORED = {"__pycache__", ".pytest_cache", ".DS_Store"}
CREDENTIAL = re.compile(r"\b(?:ghp_|github_pat_|sk-proj-)[A-Za-z0-9_-]{15,}")
LONG_ID = re.compile(r"(?<![A-Za-z0-9])\d{30,45}(?![A-Za-z0-9])")
BANNED = (".pt", ".pth", ".ckpt", ".safetensors", ".npy", ".npz", ".dcm", ".nii", ".nii.gz", ".png", ".jpg", ".jpeg", ".pdf", ".html")

def sha(raw): return hashlib.sha256(raw).hexdigest()

def safe_relative(text):
    p = PurePosixPath(text)
    if not text or p.is_absolute() or ".." in p.parts or "\\" in text:
        raise ValueError("Unsafe release path: " + text)
    return p

def notebook_payload(raw):
    nb = json.loads(raw)
    constants = {}
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code": continue
        if cell.get("outputs") or cell.get("execution_count") is not None:
            raise ValueError("Notebook contains executed outputs/counts")
        source = cell.get("source", "")
        if isinstance(source, list): source = "".join(source)
        try: tree = ast.parse(source)
        except SyntaxError: continue
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                for target in node.targets:
                    if isinstance(target, ast.Name): constants[target.id] = node.value.value
    if "PAYLOAD_B64" in constants:
        payload = base64.b64decode(constants["PAYLOAD_B64"], validate=True)
    elif "PAYLOAD" in constants:
        payload = base64.b85decode(constants["PAYLOAD"])
    else: return nb, []
    members = []
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        if sum(x.file_size for x in z.infolist()) > 5_000_000:
            raise ValueError("Over-limit embedded archive")
        for info in z.infolist():
            safe_relative(info.filename)
            if not info.is_dir(): members.append((info.filename, z.read(info)))
    return nb, members

def obvious_content_problem(path, raw):
    text = raw.decode("utf-8", errors="strict")
    if CREDENTIAL.search(text): return "possible credential"
    if LONG_ID.search(text): return "possible long record/case identifier"
    if re.search(r"(?m)^-----BEGIN (?:[A-Z0-9]+ )?PRIVATE KEY-----\s*$", text):
        return "private key"
    return None

def verify_tree(repo, manifest):
    problems = []
    expected = manifest["files"]
    for rel, digest in expected.items():
        safe_relative(rel)
        p = repo / rel
        if p.is_symlink() or any(x.is_symlink() for x in p.parents if x != repo.parent):
            problems.append(rel + ": symlink"); continue
        if not p.is_file(): problems.append(rel + ": missing"); continue
        raw = p.read_bytes()
        if sha(raw) != digest: problems.append(rel + ": bytes differ from handoff snapshot"); continue
        problem = obvious_content_problem(rel, raw)
        if problem: problems.append(rel + ": " + problem)
        if p.suffix == ".ipynb":
            try:
                _, members = notebook_payload(raw)
                package = PurePosixPath(rel).parts[1]
                for name, data in members:
                    candidates = ["research/"+name, "research/"+package+"/"+name]
                    if not any(x in expected and sha(data) == expected[x] for x in candidates):
                        problems.append(rel + ": embedded file does not match preserved source: " + name)
                    issue = obvious_content_problem(name, data)
                    if issue: problems.append(rel + " embedded " + name + ": " + issue)
            except (ValueError, TypeError, KeyError, zipfile.BadZipFile) as exc:
                problems.append(rel + ": " + str(exc))
    own_manifest = "tools/pilot_v1_release/RELEASE_FILES.json"
    for scope in SCOPES:
        for p in (repo/scope).rglob("*"):
            rel = p.relative_to(repo).as_posix()
            if any(part in IGNORED for part in p.relative_to(repo).parts): continue
            if p.is_symlink(): problems.append(rel+": unreviewed symlink"); continue
            if p.is_file() and rel not in expected and rel != own_manifest:
                problems.append(rel+": extra unreviewed file (not automatically deleted)")
    return problems

def verify_staged(repo, manifest):
    """Check all staged file blobs, not only the worktree version."""
    names = subprocess.check_output(["git","-C",str(repo),"diff","--cached","--name-only","-z","--diff-filter=ACDMRTUXB"]).decode().split("\0")
    names = [n for n in names if n]
    if not names: return ["Nothing is staged; no publication commit has been prepared."]
    expected = manifest["files"]
    issues=[]
    own="tools/pilot_v1_release/RELEASE_FILES.json"
    for name in names:
        safe_relative(name)
        if name not in expected and name != own:
            issues.append(name+": staged outside the reviewed handoff; inspect/unstage separately"); continue
        try: blob=subprocess.check_output(["git","-C",str(repo),"show",":"+name],stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            issues.append(name+": staged deletion or unreadable index entry");continue
        wanted = sha((repo/own).read_bytes()) if name==own else expected[name]
        if sha(blob)!=wanted: issues.append(name+": staged bytes do not match reviewed snapshot")
    return issues

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",type=Path,default=Path.cwd())
    parser.add_argument("--staged",action="store_true")
    args=parser.parse_args();repo=args.repo.resolve()
    mf=repo/"tools/pilot_v1_release/RELEASE_FILES.json"
    manifest=json.loads(mf.read_text(encoding="utf-8"))
    errors=verify_tree(repo,manifest)
    if args.staged: errors+=verify_staged(repo,manifest)
    if errors: raise SystemExit("STOP (no files changed):\n"+"\n".join(errors))
    print(f"PASS: {len(manifest['files'])} planned files match the handoff snapshot.")
    print("PASS: two clean notebook payloads match the preserved source packages.")
    if args.staged: print("PASS: staged file blobs are within the reviewed handoff.")
    print("LIMIT: automated checks do not certify privacy, publication permission, scientific accuracy, or Git history cleanliness.")
if __name__=="__main__": main()
