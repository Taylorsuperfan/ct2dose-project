"""Tests use aggregate fixtures only; never user arrays/checkpoints."""
import csv
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('pg9viz',ROOT/'pg9viz.py')
viz=importlib.util.module_from_spec(spec);spec.loader.exec_module(viz)
SOURCE=ROOT/'data/comparison_reported.csv'

class VisualTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.csv=self.root/'comparison.csv'
        self.csv.write_bytes(SOURCE.read_bytes())
    def mutate(self,func):
        rows=list(csv.DictReader(io.StringIO(self.csv.read_text())))
        rows=func(rows)
        with self.csv.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    def saved_report(self):
        status={'scope':viz.SCOPE,'new_method_comparison_completed':True,'n_records':600,'n_validation_cases':2}
        (self.root/'REPORT_READY.json').write_text(json.dumps(status))
        self.rehash()
    def rehash(self):
        (self.root/'FILES.json').write_text(json.dumps({n:viz.sha((self.root/n).read_bytes()) for n in ('comparison.csv','REPORT_READY.json')}))
    def test_exact_six_and_approved_columns(self):
        rows,_=viz.read_comparison(self.csv)
        self.assertEqual(tuple(rows),viz.ORDER)
        self.assertEqual(set(rows['new_hjd_rf']),set(viz.COLS))
    def test_missing_method_rejected(self):
        self.mutate(lambda rs:rs[:-1])
        with self.assertRaises(ValueError):viz.read_comparison(self.csv)
    def test_duplicate_method_rejected(self):
        self.mutate(lambda rs:rs+[rs[0]])
        with self.assertRaises(ValueError):viz.read_comparison(self.csv)
    def test_nonfinite_rejected(self):
        def f(rs):rs[0]['rmse_stored_units']='nan';return rs
        self.mutate(f)
        with self.assertRaises(ValueError):viz.read_comparison(self.csv)
    def test_wrong_cohort_rejected(self):
        def f(rs):rs[0]['n_records']='599';return rs
        self.mutate(f)
        with self.assertRaises(ValueError):viz.read_comparison(self.csv)
    def test_extra_private_columns_not_returned(self):
        def f(rs):
            for r in rs:r['private_path']='DO_NOT_EXPORT'
            return rs
        self.mutate(f);rows,_=viz.read_comparison(self.csv)
        self.assertNotIn('private_path',str(rows));self.assertNotIn('DO_NOT_EXPORT',str(rows))
    def test_relative_delta_and_pp_distinguished(self):
        rows,_=viz.read_comparison(self.csv);changes=viz.relative_changes(rows)
        c=next(r for r in changes if r['method']=='new_hjd_rf' and r['metric']=='x_mean_pct')
        self.assertAlmostEqual(c['absolute_delta'],.594450313,places=8)
        self.assertEqual(c['absolute_delta_unit'],'percentage_points')
        self.assertAlmostEqual(c['relative_change_percent'],8.9067314727,places=5)
        self.assertTrue(next(r for r in changes if r['method']=='new_hjd_rf' and r['metric']=='rmse_stored_units')['relative_change_percent']<0)
    def test_snapshot_rounding_accepted_mismatch_rejected(self):
        rows,_=viz.read_comparison(self.csv);rows['new_hjd_rf']['rmse_stored_units']*=1+1e-10
        viz.compare_snapshot(rows,SOURCE)
        rows['new_hjd_rf']['rmse_stored_units']*=1.01
        with self.assertRaises(ValueError):viz.compare_snapshot(rows,SOURCE)
    def test_saved_report_matches(self):
        self.saved_report();self.assertEqual(viz.validate_saved_report(self.root,SOURCE),self.csv)
    def test_saved_report_tamper_rejected(self):
        self.saved_report();self.csv.write_text(self.csv.read_text()+'\n')
        with self.assertRaises(ValueError):viz.validate_saved_report(self.root,SOURCE)
    def test_incomplete_scope_rejected_even_with_correct_hash(self):
        self.saved_report();p=self.root/'REPORT_READY.json';s=json.loads(p.read_text());s['new_method_comparison_completed']=False;p.write_text(json.dumps(s));self.rehash()
        with self.assertRaises(ValueError):viz.validate_saved_report(self.root,SOURCE)
    def test_output_lifecycle(self):
        def fake_figures(rows,changes,out):out.mkdir();(out/'fixture_only.txt').write_text('software test')
        with patch.object(viz,'make_figures',side_effect=fake_figures):
            out=viz.build(self.csv,self.root/'out','user_reported_aggregate')
            self.assertEqual(out,viz.build(self.csv,out,'user_reported_aggregate'))
            with self.assertRaises(FileExistsError):viz.build(self.csv,out,'saved_report_csv')
            self.assertEqual(viz.verify_output(out)['status'],'aggregate_figures_generated')
            (out/'relative_changes_vs_phase10d.csv').write_text('changed')
            with self.assertRaises(ValueError):viz.verify_output(out)
    def test_all_figures_single_axes_and_zero_origin_bars(self):
        rows,_=viz.read_comparison(self.csv);seen=[]
        def inspect(fig,out,stem):
            self.assertEqual(len(fig.axes),1)
            if stem.startswith(('01','02','03','04','07')):self.assertEqual(fig.axes[0].get_xlim()[0],0)
            seen.append(stem);viz.plt.close(fig)
        with patch.object(viz,'_save',side_effect=inspect):viz.make_figures(rows,viz.relative_changes(rows),self.root/'figs')
        self.assertEqual(seen,[r[0] for r in viz.FIGURES])
    def test_no_statistical_intervals_or_new_experiments(self):
        text=(ROOT/'pg9viz.py').read_text()
        self.assertNotIn('import torch',text);self.assertNotIn('seaborn',text)
        self.assertNotIn('plt.subplots',text);self.assertNotIn('plt.style',text)
        self.assertNotIn('errorbar(',text)

if __name__=='__main__':unittest.main()
