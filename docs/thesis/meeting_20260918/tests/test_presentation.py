import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('render',ROOT/'make_figures.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class PresentationTests(unittest.TestCase):
    def setUp(self):self.rows=m.load_rows(ROOT/'data/comparison_reported.csv')
    def test_six_unchanged_identifiers(self):self.assertEqual(tuple(self.rows),m.ORDER)
    def test_exact_reported_values(self):
        self.assertEqual(self.rows['new_hjd_rf']['rmse_stored_units'],4.177383805e-6)
        self.assertEqual(self.rows['phase10d_strict']['x_mean_pct'],6.674168481)
    def test_relative_changes(self):
        d=m.relative_changes(self.rows)
        self.assertAlmostEqual(d['new_hjd_rf']['rmse_stored_units'],-3.474448306925915,places=4)
        self.assertGreater(d['new_hjd_rf']['x_profile_rmse_stored_units'],0)
        self.assertLess(d['new_residual_rf']['rmse_stored_units'],0)
    def test_mapping_and_no_opaque_plot_names(self):
        labels=json.loads((ROOT/'method_names.json').read_text())
        self.assertEqual(set(labels),set(m.ORDER))
        for row in labels.values():
            for token in ['V1','HJD',' RF','Phase']:self.assertNotIn(token,row['plot'])
    def test_seven_figures(self):self.assertEqual(len(m.FIGURES),7)
    def test_target_table_unchanged_scope(self):
        with (ROOT/'data/comparison_reported.csv').open() as f:rows=list(csv.DictReader(f))
        self.assertTrue(all(r['n_cases']=='2' and r['n_records']=='600' for r in rows))
    def test_refuses_changed_scope(self):
        s=(ROOT/'data/comparison_reported.csv').read_text().replace(',2,600,',',2,560,')
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'bad.csv';p.write_text(s)
            with self.assertRaises(ValueError):m.load_rows(p)
    def test_refuses_nonempty_output(self):
        with tempfile.TemporaryDirectory() as t:
            out=Path(t)/'out';out.mkdir();(out/'keep.txt').write_text('old')
            with self.assertRaises(FileExistsError):m.build(ROOT,out)
            self.assertEqual((out/'keep.txt').read_text(),'old')

if __name__=='__main__':unittest.main()
