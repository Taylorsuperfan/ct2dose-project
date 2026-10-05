"""Packaging checks on temporary files. No clinical arrays or external service."""
from pathlib import Path
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
def module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
CHECK=module('release_check_test',ROOT/'scripts/check_release.py')
INSTALL=module('release_install_test',ROOT/'scripts/install_into_repo.py')

def manifest(root):
    items=[]
    for p in sorted(root.rglob('*')):
        if p.is_file() and p.name!='RELEASE_MANIFEST.json' and '__pycache__' not in p.parts:
            b=p.read_bytes();items.append({'path':p.relative_to(root).as_posix(),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
    (root/'RELEASE_MANIFEST.json').write_text(json.dumps({'files':items}))

def git(repo,*args):
    return subprocess.check_output(['git','-C',str(repo),*args],stderr=subprocess.STDOUT)

class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name);self.src=self.base/'source';self.src.mkdir()
        (self.src/'README.md').write_text('Temporary packaging fixture.\n')
        (self.src/'scripts').mkdir()
        shutil.copyfile(ROOT/'scripts/check_release.py',self.src/'scripts/check_release.py')
        manifest(self.src)
        self.repo=self.base/'checkout';self.repo.mkdir()
        git(self.repo,'init','-q');git(self.repo,'config','user.name','Packaging Test')
        git(self.repo,'config','user.email','test@example.invalid')
        (self.repo/'initial.txt').write_text('initial\n');git(self.repo,'add','initial.txt');git(self.repo,'commit','-qm','initial')
    def quiet(self,fn,*args,**kwargs):
        with contextlib.redirect_stdout(io.StringIO()):return fn(*args,**kwargs)
    def test_valid_manifest(self):self.quiet(CHECK.verify,self.src)
    def test_tamper_stops(self):
        (self.src/'README.md').write_text('changed')
        with self.assertRaises(RuntimeError):self.quiet(CHECK.verify,self.src)
    def test_extra_file_stops(self):
        (self.src/'extra.txt').write_text('extra')
        with self.assertRaises(RuntimeError):self.quiet(CHECK.verify,self.src)
    def test_weight_file_stops_even_if_listed(self):
        (self.src/'weights.pt').write_bytes(b'not an actual model');manifest(self.src)
        with self.assertRaises(RuntimeError):self.quiet(CHECK.verify,self.src)
    def test_executed_notebook_stops(self):
        (self.src/'run.ipynb').write_text(json.dumps({'cells':[{'cell_type':'code','source':['x=1'],'outputs':[{'text':'private'}],'execution_count':1}]}));manifest(self.src)
        with self.assertRaises(RuntimeError):self.quiet(CHECK.verify,self.src)
    def test_credential_pattern_stops(self):
        (self.src/'note.txt').write_text('ghp_'+'A'*40);manifest(self.src)
        with self.assertRaises(RuntimeError):self.quiet(CHECK.verify,self.src)
    def test_symlink_stops(self):
        (self.src/'link').symlink_to(self.src/'README.md')
        with self.assertRaises(RuntimeError):self.quiet(CHECK.verify,self.src)
    def test_dryrun_no_copy(self):
        self.quiet(INSTALL.install,self.repo,False,self.src)
        self.assertFalse((self.repo/'research').exists())
    def test_install_and_repeat(self):
        n,_=self.quiet(INSTALL.install,self.repo,True,self.src);self.assertGreater(n,0)
        n,same=self.quiet(INSTALL.install,self.repo,True,self.src);self.assertEqual(n,0);self.assertGreater(same,0)
    def test_conflict_no_partial_copy(self):
        dst=self.repo/'research'/INSTALL.NAME;dst.mkdir(parents=True)
        (dst/'README.md').write_text('user work')
        with self.assertRaises(RuntimeError):self.quiet(INSTALL.install,self.repo,True,self.src)
        self.assertFalse((dst/'scripts').exists());self.assertEqual((dst/'README.md').read_text(),'user work')
    def test_non_repository_rejected(self):
        d=self.base/'notgit';d.mkdir()
        with self.assertRaises(subprocess.CalledProcessError):self.quiet(INSTALL.install,d,True,self.src)
    def test_destination_symlink_rejected(self):
        d=self.base/'elsewhere';d.mkdir();(self.repo/'research').symlink_to(d)
        with self.assertRaises(RuntimeError):self.quiet(INSTALL.install,self.repo,True,self.src)
    def test_staged_index_matches(self):
        self.quiet(INSTALL.install,self.repo,True,self.src);dst=self.repo/'research'/INSTALL.NAME
        git(self.repo,'add','research');self.quiet(CHECK.verify_staged,dst,self.repo)
    def test_unrelated_staged_file_rejected(self):
        self.quiet(INSTALL.install,self.repo,True,self.src);dst=self.repo/'research'/INSTALL.NAME
        (self.repo/'unrelated.txt').write_text('not part of handoff');git(self.repo,'add','research','unrelated.txt')
        with self.assertRaises(RuntimeError):self.quiet(CHECK.verify_staged,dst,self.repo)
    def test_stale_index_rejected(self):
        self.quiet(INSTALL.install,self.repo,True,self.src);dst=self.repo/'research'/INSTALL.NAME
        git(self.repo,'add','research');(dst/'README.md').write_text('unstaged edit')
        with self.assertRaises(RuntimeError):self.quiet(CHECK.verify_staged,dst,self.repo)
    def test_report_table_has_seven_primary_methods(self):
        with (ROOT/'results/comparison_reported.csv').open() as f:r=list(csv.DictReader(f))
        self.assertEqual(len(r),7);self.assertEqual(len({x['method'] for x in r}),7)
        self.assertTrue(all(x['n_records']=='600' and x['n_cases']=='2' for x in r))
    def test_parent_source_exact_match(self):
        for p in (ROOT/'source/wfr_coefficient_refinement/parent_source/cwfr').glob('*.py'):
            self.assertEqual(p.read_bytes(),(ROOT/'source/conditional_wfr_dose/cwfr'/p.name).read_bytes())

if __name__=='__main__':unittest.main()
