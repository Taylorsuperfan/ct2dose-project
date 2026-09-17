"""Check the combined frozen-code + aggregate-figure handoff; no Git writes.

This is an allowlisted snapshot check, NOT permission, scientific or full privacy certification.
"""
from pathlib import Path, PurePosixPath
import argparse,ast,base64,hashlib,importlib.util,io,json,subprocess,zipfile,sys
OWN='tools/pilot_v1_visual_release/RELEASE_FILES.json'
SCOPES=(
 'research/phase10d_recovered_comparison','research/phase9g_signed_real_v1',
 'research/phase9g_v1_visuals','docs/thesis/updates/20260917_real_pilot_v1',
 'docs/thesis/updates/20260917_real_pilot_v1_visuals','tools/pilot_v1_release',
 'tools/pilot_v1_visual_release')
IGNORED={'__pycache__','.pytest_cache','.DS_Store'}
def sha(b):return hashlib.sha256(b).hexdigest()
def safe(root,rel):
 p=PurePosixPath(rel)
 if not rel or p.is_absolute() or '..' in p.parts or '\\' in rel:raise ValueError('Unsafe path: '+rel)
 target=root.joinpath(*p.parts)
 for q in (target,*target.parents):
  if q==root.parent:break
  if q.is_symlink():raise ValueError('Symlink: '+str(q))
 return target

def _legacy(repo):
 spec=importlib.util.spec_from_file_location('preserved_v1_check',repo/'tools/pilot_v1_release/check_release.py')
 mod=importlib.util.module_from_spec(spec)
 old=sys.dont_write_bytecode;sys.dont_write_bytecode=True
 try:spec.loader.exec_module(mod)
 finally:sys.dont_write_bytecode=old
 return mod

def visual_notebook(raw,expected):
 nb=json.loads(raw);constants={}
 for cell in nb.get('cells',[]):
  if cell.get('cell_type')!='code':continue
  if cell.get('outputs') or cell.get('execution_count') is not None:raise ValueError('Executed visualization notebook')
  s=cell.get('source','');s=''.join(s) if isinstance(s,list) else s
  tree=ast.parse(s)
  for node in tree.body:
   if isinstance(node,ast.Assign) and isinstance(node.value,ast.Constant):
    for t in node.targets:
     if isinstance(t,ast.Name):constants[t.id]=node.value.value
 rawzip=base64.b64decode(constants['VIZ_PAYLOAD_B64'],validate=True)
 if sha(rawzip)!=constants['VIZ_PAYLOAD_SHA256']:raise ValueError('Embedded visual ZIP hash mismatch')
 with zipfile.ZipFile(io.BytesIO(rawzip)) as z:
  if sum(i.file_size for i in z.infolist())>500_000:raise ValueError('Over-limit visualization archive')
  m=json.loads(z.read('payload_manifest.json'))
  if set(z.namelist())!=set(m['files'])|{'payload_manifest.json'}:raise ValueError('Unexpected visualization embedded file')
  for name,h in m['files'].items():
   p=PurePosixPath(name)
   if p.is_absolute() or '..' in p.parts or '\\' in name:raise ValueError('Unsafe notebook file')
   wanted=expected.get('research/phase9g_v1_visuals/'+name)
   if wanted!=h or sha(z.read(name))!=h:raise ValueError('Visual notebook/source mismatch: '+name)

def verify_tree(repo,manifest):
 errors=[];expected=manifest['files']
 # Hash all files before importing the preserved read-only checker.
 for rel,h in expected.items():
  try:
   p=safe(repo,rel)
   if not p.is_file():raise ValueError('Missing file')
   if sha(p.read_bytes())!=h:raise ValueError('Bytes differ')
  except (ValueError,OSError) as e:errors.append(rel+': '+str(e))
 if errors:return errors
 legacy=_legacy(repo)
 old=json.loads((repo/'tools/pilot_v1_release/RELEASE_FILES.json').read_text())
 errors.extend(legacy.verify_tree(repo,old))
 for rel in expected:
  if not (rel.startswith('research/phase9g_v1_visuals/') or rel.startswith('docs/thesis/updates/20260917_real_pilot_v1_visuals/') or rel.startswith('tools/pilot_v1_visual_release/')):continue
  raw=(repo/rel).read_bytes()
  if Path(rel).suffix=='.png':continue # Only explicitly listed, hashed aggregate PNGs are admitted.
  try:
   problem=legacy.obvious_content_problem(rel,raw)
   if problem:errors.append(rel+': '+problem)
   if Path(rel).suffix=='.ipynb':visual_notebook(raw,expected)
  except (ValueError,KeyError,TypeError,SyntaxError,UnicodeError) as e:errors.append(rel+': '+str(e))
 for scope in SCOPES:
  for p in (repo/scope).rglob('*'):
   rel=p.relative_to(repo).as_posix()
   if any(x in IGNORED for x in p.parts):continue
   if p.is_symlink():errors.append(rel+': unreviewed symlink')
   elif p.is_file() and rel not in expected and rel!=OWN:errors.append(rel+': extra file, not deleted automatically')
 return errors

def verify_staged(repo,manifest):
 names=subprocess.check_output(['git','-C',str(repo),'diff','--cached','--name-only','-z']).decode().split('\0')
 names=[n for n in names if n]
 if not names:return ['Nothing is staged.']
 expected=dict(manifest['files']);expected[OWN]=sha((repo/OWN).read_bytes());errors=[]
 for rel in names:
  if rel not in expected:errors.append(rel+': outside combined reviewed handoff');continue
  try:blob=subprocess.check_output(['git','-C',str(repo),'show',':'+rel],stderr=subprocess.DEVNULL)
  except subprocess.CalledProcessError:errors.append(rel+': index deletion/unreadable');continue
  if sha(blob)!=expected[rel]:errors.append(rel+': staged bytes differ from reviewed snapshot')
 return errors

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--repo',type=Path,default=Path.cwd());ap.add_argument('--staged',action='store_true');a=ap.parse_args()
 root=a.repo.resolve();manifest=json.loads((root/OWN).read_text());errors=verify_tree(root,manifest)
 if a.staged and not errors:errors.extend(verify_staged(root,manifest))
 if errors:raise SystemExit('STOP: no files changed.\n'+'\n'.join(errors))
 print(f"PASS: {len(manifest['files'])} reviewed files verified; preserved training/recovery packages unchanged.")
 print('PASS: three clean notebooks have matching embedded source; seven aggregate PNG/SVG pairs are allowlisted.')
 if a.staged:print('PASS: staged blobs belong to the combined reviewed handoff.')
 print('LIMIT: no publication permission, full privacy audit, Git-history cleanup, scientific validation or external writes.')
if __name__=='__main__':main()
