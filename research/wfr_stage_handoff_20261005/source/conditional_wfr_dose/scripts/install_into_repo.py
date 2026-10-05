"""Import only this new code package; never replace a differing repository file."""
from pathlib import Path
import argparse,subprocess,shutil,hashlib
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--apply',action='store_true');a=p.parse_args()
root=a.repo.resolve(strict=True);gitroot=Path(subprocess.check_output(['git','-C',str(root),'rev-parse','--show-toplevel'],text=True).strip()).resolve()
if root!=gitroot:raise SystemExit('Use the actual repository root, not a same-name copy.')
source=Path(__file__).resolve().parents[1];dest=root/'research'/'conditional_wfr_dose'
files=[f for f in sorted(source.rglob('*')) if f.is_file() and '__pycache__' not in f.parts and f.suffix!='.pyc']
new=[];same=[];conflicts=[]
for f in files:
 d=dest/f.relative_to(source)
 if not d.exists():new.append((f,d))
 elif f.read_bytes()==d.read_bytes():same.append(d)
 else:conflicts.append(d)
print('Repository:',root);print('New files:',len(new));print('Identical files retained:',len(same))
if conflicts:raise SystemExit('Conflicting existing files; no files copied:\n'+'\n'.join(map(str,conflicts)))
if not a.apply:print('CHECK ONLY: re-run with --apply after review.')
else:
 for f,d in new:d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,d)
 print('Imported into:',dest)
