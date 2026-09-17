"""Basic public-tree check, not a complete privacy audit."""
from pathlib import Path
import sys,json,re
root=Path(sys.argv[1]);bad=[]
for p in root.rglob('*'):
 if not p.is_file() or any(x in p.parts for x in ['__pycache__','.pytest_cache','.git']):continue
 if p.suffix.lower() in {'.pt','.pth','.ckpt','.npy','.npz','.dcm'} or '.local' in p.name:bad.append(str(p))
 if p.suffix=='.ipynb':
  n=json.loads(p.read_text())
  if any(c.get('outputs') or c.get('execution_count') is not None for c in n['cells'] if c['cell_type']=='code'):bad.append(str(p)+' has executed outputs')
 if p.suffix in {'.py','.md','.json','.csv','.txt'}:
  s=p.read_text(errors='replace')
  if re.search(r'\b(?:ghp_|github_pat_|sk-proj-)[A-Za-z0-9_]{15,}',s):bad.append(str(p)+' possible credential')
if bad:raise SystemExit('STOP:\n'+'\n'.join(bad))
print('PASS: no obvious raw arrays/checkpoints/executed notebooks/credential patterns. Review manually before publishing.')
