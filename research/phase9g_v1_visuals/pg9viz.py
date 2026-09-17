"""Read-only visual summaries of the frozen real Phase9G V1 pilot.

No torch, medical arrays, model loading, training or inference. The main input
is an already computed model-level comparison CSV. Each figure has one axes.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import html
import io
import json
import math
import platform
import sys
from pathlib import Path, PurePosixPath

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ORDER = ('phase5e_rf', 'phase9d_plus', 'phase9g', 'phase10d_strict',
         'new_residual_rf', 'new_hjd_rf')
LABELS = {
    'phase5e_rf': 'Phase5E RF [upstream]',
    'phase9d_plus': 'Phase9D-plus',
    'phase9g': 'Phase9G [frozen shared base]',
    'phase10d_strict': 'Phase10D-strict [old final]',
    'new_residual_rf': 'New residual RF',
    'new_hjd_rf': 'New HJD RF',
}
NEW = ('new_residual_rf', 'new_hjd_rf')
METRICS = ('rmse_stored_units', 'mae_stored_units', 'x_mean_pct', 'x_max_pct',
           'x_profile_rmse_stored_units', 'y_profile_rmse_stored_units',
           'z_profile_rmse_stored_units')
COLS = ('method', *METRICS, 'n_cases', 'n_records')
DELTA_KEYS = ('rmse_stored_units','mae_stored_units',
              'x_profile_rmse_stored_units','y_profile_rmse_stored_units',
              'z_profile_rmse_stored_units','x_mean_pct')
DELTA_LABELS = ('Whole-volume RMSE','Whole-volume MAE','Legacy-x profile RMSE',
                'Legacy-y profile RMSE','Legacy-z profile RMSE',
                'Legacy-x mean percentage error')
PARAMETERS = {'new_residual_rf': 340545, 'new_hjd_rf': 104144}
SCOPE = 'REAL_PHASE9G_ERROR_CORRECTION_PILOT_NOT_WATER_NOT_BLIND_FINAL_TEST'
VERSION = 'pg9-v1-visuals-1.0.0'
FOOTER = ('600 development records / 2 cases; new heads: seed 17. Equal-case means of record metrics.\n'
          'Learned Phase9G reference, not water; not a blind final test. No uncertainty intervals supplied.')
FIGURES = [
 ('01_volume_rmse','Whole-volume RMSE',
  'All six saved methods, in fixed lineage order; bars begin at zero. Values are means of record RMSEs, not one pooled RMSE.'),
 ('02_volume_mae','Whole-volume MAE',
  'All six saved methods, same order and zero origin. Scaling by 10^6 is display only, not a conversion to Gy.'),
 ('03_x_percentage_error','Legacy-x percentage error',
  'Mean local percentage error on the original GT-peak profile. The original evaluator keeps positions at least 1% of that GT line peak. Not an independently verified physical beam axis.'),
 ('04_xyz_profile_rmse','Absolute profile RMSE',
  'Whole-line absolute errors, one grouped figure for x/y/z. Legacy x=W, y=H, z=D. The large upstream RF bar is retained; figure 05 resolves smaller relative differences.'),
 ('05_relative_changes_vs_phase10d','Changes relative to Phase10D-strict',
  '100*(new/Phase10D-1), computed separately for each metric. Negative means lower error; positive means higher error. These are relative percentages, NOT percentage-point differences and NOT a combined score.'),
 ('06_global_profile_tradeoff','Global/profile trade-off',
  'Detail view of five correction systems; the upstream Phase5E is excluded from this detail view only and remains in figures 01-04. Both axes are errors: lower left is better. Axes are zoomed and labeled. No connecting curve, significance claim, or Pareto guarantee.'),
 ('07_correction_parameter_count','New correction-network size',
  'Only the two newly trained correction networks. Counts come from the user-supplied resource table, not from timing. Shared upstream weights and runtime/particle costs are excluded; fewer parameters does not prove faster inference.'),
]


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def jwrite(path: Path, value: object) -> None:
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def read_comparison(path: Path) -> tuple[dict[str, dict], bytes]:
    """Validate the six-method, 600-record aggregate; discard unneeded columns."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Comparison must be an existing non-symlink CSV.')
    if path.stat().st_size > 2_000_000:
        raise ValueError('Expected a small aggregate CSV, not per-record data.')
    raw = path.read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
    if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise ValueError('Missing or duplicate CSV columns.')
    missing = set(COLS) - set(reader.fieldnames)
    if missing:
        raise ValueError('Missing required aggregate columns: ' + ', '.join(sorted(missing)))
    records = {}
    for row in reader:
        name = row['method']
        if name not in ORDER or name in records:
            raise ValueError('Unknown or duplicate method: ' + name)
        item = {'method': name}
        for key in METRICS:
            try: value = float(row[key])
            except (ValueError, TypeError): raise ValueError('Non-numeric metric: ' + key) from None
            if not math.isfinite(value) or value < 0:
                raise ValueError('Metric must be finite and nonnegative: ' + key)
            item[key] = value
        for key, expected in [('n_cases', 2), ('n_records', 600)]:
            value = float(row[key])
            if not math.isfinite(value) or value != expected:
                raise ValueError(f'{key} must be {expected}; do not silently mix cohorts.')
            item[key] = int(value)
        records[name] = item
    if set(records) != set(ORDER):
        raise ValueError('All six methods are required; no zero or omitted-method fallback.')
    return {key: records[key] for key in ORDER}, raw


def relative_changes(rows: dict[str, dict]) -> list[dict]:
    baseline = rows['phase10d_strict']
    out = []
    for name in NEW:
        for key in DELTA_KEYS:
            b, v = baseline[key], rows[name][key]
            if b <= 0:
                raise ValueError('Relative changes require a strictly positive baseline.')
            out.append({'method': name, 'baseline': 'phase10d_strict', 'metric': key,
                        'baseline_value': b, 'method_value': v, 'absolute_delta': v-b,
                        'relative_change_percent': 100*(v/b-1),
                        'absolute_delta_unit': 'percentage_points' if key.endswith('_pct') else 'stored_numerical_units'})
    return out


def write_csv(path: Path, rows: list[dict], fields: tuple | list) -> None:
    with path.open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: format(row[k], '.17g') if isinstance(row[k], float) else row[k] for k in fields})


def compare_snapshot(rows: dict[str, dict], snapshot: Path, rtol: float = 1e-7) -> dict:
    """Compare Drive aggregates with the displayed-precision snapshot, not fitted values."""
    reference, _ = read_comparison(snapshot)
    failures = []
    max_relative = 0.0
    for name in ORDER:
        for key in METRICS:
            a, b = rows[name][key], reference[name][key]
            rel = abs(a-b)/max(abs(b), 1e-300)
            max_relative = max(max_relative, rel)
            if not math.isclose(a, b, rel_tol=rtol, abs_tol=0.0):
                failures.append(name + ':' + key)
    if failures:
        raise ValueError('Saved CSV differs from the reported V1 snapshot; no retuning or silent fallback: ' + ', '.join(failures))
    return {'comparison': 'aggregate-only, within reported display precision', 'rtol': rtol,
            'max_relative_metric_difference': max_relative,
            'not_proof_of_record_or_model_identity': True}


def validate_saved_report(report_dir: Path, snapshot: Path) -> Path:
    """Read exactly three small report files; do not open arrays or other run artifacts."""
    report_dir = Path(report_dir)
    names = ('comparison.csv', 'REPORT_READY.json', 'FILES.json')
    for name in names:
        p = report_dir/name
        if not p.is_file() or p.is_symlink() or p.stat().st_size > 2_000_000:
            raise ValueError('Missing, symlink or over-limit report metadata: ' + name)
    manifest = json.loads((report_dir/'FILES.json').read_text(encoding='utf-8'))
    for name in ('comparison.csv', 'REPORT_READY.json'):
        if sha((report_dir/name).read_bytes()) != manifest.get(name):
            raise ValueError('Saved report byte hash mismatch: ' + name)
    status = json.loads((report_dir/'REPORT_READY.json').read_text(encoding='utf-8'))
    if (status.get('scope') != SCOPE or status.get('new_method_comparison_completed') is not True
        or status.get('n_records') != 600 or status.get('n_validation_cases') != 2):
        raise ValueError('Wrong or incomplete real V1 report scope.')
    rows, _ = read_comparison(report_dir/'comparison.csv')
    compare_snapshot(rows, snapshot)
    return report_dir/'comparison.csv'


def _figure(title: str, subtitle: str = '', height: float = 6.7):
    fig = plt.figure(figsize=(11.8, height))
    ax = fig.add_axes([0.28, 0.24, 0.65, 0.59])
    fig.text(0.045, 0.95, title, fontsize=16, weight='bold', va='top')
    fig.text(0.045, 0.892, subtitle, fontsize=10, va='top')
    fig.text(0.045, 0.045, FOOTER, fontsize=8.6, va='bottom')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fig, ax


def _save(fig, out: Path, stem: str):
    # PNG for slides/Trello, SVG for editable vector placement. No PDF/fonts exported.
    for suffix in ('png','svg'):
        fig.savefig(out/f'{stem}.{suffix}', dpi=200, bbox_inches='tight',
                    metadata={'Software': VERSION} if suffix=='png' else {'Creator': VERSION, 'Date': None})
    plt.close(fig)


def _bar(rows, key, scale, title, xlabel, out, stem):
    vals = [rows[m][key]*scale for m in ORDER]
    fig, ax = _figure(title, 'All six methods | lower is better | zero-origin bars')
    bars = ax.barh(np.arange(6), vals, height=.62)
    bars[3].set_hatch('//')
    for i in (4,5): bars[i].set_hatch('..')
    ax.set_yticks(range(6), [LABELS[m] for m in ORDER], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals)*1.24)
    ax.set_xlabel(xlabel, fontsize=11, labelpad=10)
    ax.xaxis.grid(True, alpha=.22)
    ax.set_axisbelow(True)
    for bar, val in zip(bars, vals):
        ax.text(val + max(vals)*.017, bar.get_y()+bar.get_height()/2, f'{val:.5f}', va='center', fontsize=10)
    _save(fig,out,stem)


def make_figures(rows, changes, out: Path):
    out.mkdir()
    _bar(rows,'rmse_stored_units',1e6,'01 | Whole-volume RMSE',
         r'Mean record RMSE [$10^{-6}$ stored units]',out,'01_volume_rmse')
    _bar(rows,'mae_stored_units',1e6,'02 | Whole-volume MAE',
         r'Mean record MAE [$10^{-6}$ stored units]',out,'02_volume_mae')
    _bar(rows,'x_mean_pct',1,'03 | Legacy-x mean percentage error',
         'Mean percentage error [%] | legacy x = array W',out,'03_x_percentage_error')
    fig, ax = _figure('04 | Absolute profile RMSE in three array directions',
        'All six methods | full-line RMSE | zero-origin bars; see figure 05 for relative changes', height=8.5)
    offsets = [-.23,0,.23]
    keys = ['x_profile_rmse_stored_units','y_profile_rmse_stored_units','z_profile_rmse_stored_units']
    maxval = max(rows[m][k]*1e6 for m in ORDER for k in keys)
    for j,(key,offset) in enumerate(zip(keys,offsets)):
        vals=[rows[m][key]*1e6 for m in ORDER]
        bars=ax.barh(np.arange(6)+offset,vals,height=.21,label=f'legacy {"xyz"[j]}')
        for bar,v in zip(bars,vals):
            ax.text(v+maxval*.011,bar.get_y()+bar.get_height()/2,f'{v:.3f}',fontsize=8,va='center')
    ax.set_yticks(range(6),[LABELS[m] for m in ORDER],fontsize=10);ax.invert_yaxis()
    ax.set_xlim(0,maxval*1.17);ax.xaxis.grid(True,alpha=.22);ax.set_axisbelow(True)
    ax.set_xlabel(r'Mean record profile RMSE [$10^{-6}$ stored units]',labelpad=10)
    ax.legend(ncol=3,loc='upper left',bbox_to_anchor=(0,-.13),frameon=False)
    _save(fig,out,'04_xyz_profile_rmse')
    fig,ax=_figure('05 | Global gains versus profile regressions',
        'Relative to Phase10D-strict | 100 × (new / old − 1), separately for each metric',height=7.8)
    ax.set_position([.34,.25,.59,.56])
    bykey={(r['method'],r['metric']):r['relative_change_percent'] for r in changes}
    for j,name in enumerate(NEW):
        vals=[bykey[name,key] for key in DELTA_KEYS]
        bars=ax.barh(np.arange(6)+(j-.5)*.30,vals,height=.28,label=LABELS[name])
        for bar,v in zip(bars,vals):
            ax.text(v+(.35 if v>=0 else -.35),bar.get_y()+bar.get_height()/2,
                    f'{v:+.2f}%',ha='left' if v>=0 else 'right',va='center',fontsize=9)
    lim=max(10.0,max(abs(r['relative_change_percent']) for r in changes)*1.22)
    ax.set_xlim(-lim,lim);ax.axvline(0,linewidth=1)
    ax.set_yticks(range(6),DELTA_LABELS,fontsize=10);ax.invert_yaxis()
    ax.set_xlabel('Relative change [%]     ← lower error | higher error →',labelpad=10)
    ax.xaxis.grid(True,alpha=.22);ax.set_axisbelow(True)
    ax.legend(ncol=2,loc='upper left',bbox_to_anchor=(0,-.18),frameon=False)
    _save(fig,out,'05_relative_changes_vs_phase10d')
    fig,ax=_figure('06 | Full-volume / legacy-x trade-off',
        'Detail view: five correction systems; upstream Phase5E is shown in figures 01–04')
    ax.set_position([.13,.23,.80,.59])
    positions={'phase9d_plus':(30,-25),'phase9g':(35,30),'phase10d_strict':(12,-32),
               'new_residual_rf':(-15,18),'new_hjd_rf':(-115,12)}
    markers=['o','s','D','^','v']
    for marker,name in zip(markers,ORDER[1:]):
        x,y=rows[name]['rmse_stored_units']*1e6,rows[name]['x_mean_pct']
        ax.scatter([x],[y],s=90,marker=marker,label=LABELS[name])
        ax.annotate(LABELS[name],(x,y),xytext=positions[name],textcoords='offset points',
                    fontsize=10,arrowprops={'arrowstyle':'-','linewidth':.8})
    ax.set_xlim(3.95,4.60);ax.set_ylim(6.20,8.25)
    ax.set_xlabel(r'Mean record whole-volume RMSE [$10^{-6}$ stored units] — lower is better',labelpad=10)
    ax.set_ylabel('Legacy-x mean percentage error [%]\nlower is better',labelpad=12)
    ax.grid(True,alpha=.20)
    _save(fig,out,'06_global_profile_tradeoff')
    fig,ax=_figure('07 | Trainable size of the NEW correction only',
        'Not total-system parameters; excludes frozen Phase9G. Parameter count is not latency.',height=5.7)
    vals=[PARAMETERS[n] for n in NEW]
    bars=ax.barh(range(2),vals,height=.45)
    ax.set_yticks(range(2),[LABELS[n] for n in NEW],fontsize=11);ax.invert_yaxis()
    ax.set_xlim(0,max(vals)*1.25);ax.set_xlabel('Trainable correction parameters')
    ax.xaxis.grid(True,alpha=.22);ax.set_axisbelow(True)
    for bar,v in zip(bars,vals):
        ax.text(v+max(vals)*.02,bar.get_y()+bar.get_height()/2,f'{v:,}',va='center',fontsize=11)
    _save(fig,out,'07_correction_parameter_count')


def verify_output(out: Path) -> dict:
    out=Path(out)
    mf=out/'FIGURE_FILES.json'
    if not mf.is_file():raise FileNotFoundError('No completed figure receipt: '+str(mf))
    mapping=json.loads(mf.read_text(encoding='utf-8'))
    actual={p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file()}
    if actual != set(mapping)|{'FIGURE_FILES.json'}:
        raise ValueError('Figure output has missing or unreviewed extra files.')
    for rel,expected in mapping.items():
        p=PurePosixPath(rel)
        if p.is_absolute() or '..' in p.parts or '\\' in rel:
            raise ValueError('Unsafe receipt path.')
        target=out/rel
        if target.is_symlink() or any(x.is_symlink() for x in target.parents if x!=out.parent):
            raise ValueError('Symlink in output path.')
        if sha(target.read_bytes()) != expected:raise ValueError('Figure bytes changed: '+rel)
    return json.loads((out/'FIGURES_READY.json').read_text(encoding='utf-8'))


def build(comparison_csv: Path, out: Path, source_kind: str) -> Path:
    if source_kind not in ('user_reported_aggregate','saved_report_csv'):
        raise ValueError('Declare source provenance explicitly.')
    rows,raw=read_comparison(comparison_csv)
    changes=relative_changes(rows)
    out=Path(out).absolute()
    identity={'input_csv_sha256':sha(raw),'plot_code_sha256':sha(Path(__file__).read_bytes()),
              'source_kind':source_kind,'version':VERSION}
    if out.exists():
        ready=verify_output(out)
        if ready['identity'] != identity:raise FileExistsError('Different source/code: use a NEW output directory.')
        return out
    out.mkdir(parents=True,exist_ok=False)
    write_csv(out/'comparison_approved_columns.csv',list(rows.values()),COLS)
    write_csv(out/'relative_changes_vs_phase10d.csv',changes,list(changes[0]))
    jwrite(out/'source_provenance.json',dict(identity,
        input_name=Path(comparison_csv).name,
        no_raw_input_copy=True,export_columns=list(COLS),
        statistical_scope={'n_cases':2,'n_records':600,'new_training_seed':17,
            'aggregation':'equal-case mean of record-level metrics',
            'uncertainty':'not supplied; no error bars or significance inference',
            'units':'stored numerical units; not certified Gy',
            'reference':'frozen learned Phase9G; not water',
            'validation':'development, monitor40 overlaps val600',
            'parameters':'user-reported correction-only counts; common upstream excluded'},
        runtime={'python':platform.python_version(),'numpy':np.__version__,'matplotlib':matplotlib.__version__},
        reads='one aggregate CSV; no checkpoints, arrays, raw medical images, or model code'))
    make_figures(rows,changes,out/'figures')
    header=('# Frozen V1 — visual comparison\n\n'
       'These figures show the existing REAL Phase9G-error correction pilot; no new training or inference.\n'
       '**600 validation records from 2 development cases; one new-model seed. No independent final-test, water-dose or clinical claim.**\n\n'
       f'Input provenance: `{source_kind}`. Input CSV SHA256: `{sha(raw)}`.\n'
       'Parameter counts in figure 07 are the separately user-reported correction-network counts.\n\n'
       'Bar charts use zero origins. Figure 06 is an explicitly zoomed scatter view.\n'
       'No error bars are invented from aggregate means; mean record RMSE is not pooled RMSE.\n\n'
       'Main observation: new global RMSE/MAE are lower, but absolute x/y/z profile RMSE is higher than Phase10D-strict.\n'
       'This does not establish an overall winner, significance or where profile errors occur.\n\n')
    md=header
    for stem,title,caption in FIGURES:
        md+=f'## {title}\n\n![{title}](figures/{stem}.png)\n\n{caption}\n\n'
    (out/'README.md').write_text(md,encoding='utf-8')
    parts=['<!doctype html><html lang="en"><meta charset="utf-8"><title>V1 visual comparison</title>',
      '<main style="max-width:1100px;margin:2rem auto;font-family:Arial,sans-serif;line-height:1.55">',
      '<h1>Frozen V1: global / profile comparison</h1>',
      '<p>600 records, 2 development cases, one new-model seed. Learned Phase9G reference, NOT water. '
      'No blind final-test or clinical claim. Aggregate inputs only; no new inference.</p>',
      '<p>Input provenance: '+html.escape(source_kind)+'. Parameters are correction-only. '
      'Negative relative change means lower error, not statistical significance.</p>']
    for stem,title,caption in FIGURES:
        parts += ['<h2>'+html.escape(title)+'</h2>',
           f'<img style="width:100%" alt="{html.escape(title)}" src="figures/{stem}.png">',
           '<p>'+html.escape(caption)+'</p>']
    parts.append('</main></html>');(out/'gallery.html').write_text('\n'.join(parts),encoding='utf-8')
    jwrite(out/'FIGURES_READY.json',{'status':'aggregate_figures_generated','identity':identity,
        'n_figures':len(FIGURES),'formats':['png','svg'],
        'model_training_performed':False,'model_inference_performed':False,
        'new_scientific_validation':False,'github_or_trello_written':False})
    jwrite(out/'FIGURE_FILES.json',{p.relative_to(out).as_posix():sha(p.read_bytes()) for p in sorted(out.rglob('*')) if p.is_file()})
    verify_output(out)
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--source-kind',choices=['user_reported_aggregate','saved_report_csv'],required=True)
    a=parser.parse_args()
    print('FIGURES:',build(a.csv,a.out,a.source_kind))

if __name__=='__main__':main()
