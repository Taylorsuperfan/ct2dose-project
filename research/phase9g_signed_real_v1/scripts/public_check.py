"""Conservative source-package preflight; not proof of absence of private content."""
from pathlib import Path
import sys,json
root=Path(sys.argv[1]).resolve();bad=[]
for p in root.rglob('*'):
    if not p.is_file() or '__pycache__' in p.parts or '.pytest_cache' in p.parts:continue
    n=p.name.lower()
    if p.suffix.lower() in {'.pt','.pth','.ckpt','.npy','.npz','.dcm'} or '.local.' in n or n in {'.env'}:bad.append(str(p))
    if p.suffix=='.ipynb':
        nb=json.loads(p.read_text())
        if any(c.get('outputs') or c.get('execution_count') is not None for c in nb.get('cells',[]) if c.get('cell_type')=='code'):bad.append(str(p)+' has execution outputs')
if bad:raise SystemExit('STOP:\n'+'\n'.join(bad))
print('No obvious runtime/private file or executed-notebook output detected. Manually review source for credentials and identifiers.')
