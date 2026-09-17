"""Redraw already reported comparison values with reader-facing method names.

No models, training, checkpoint loading, medical images or per-case data.
One axes per figure; Matplotlib default color cycle, no custom palette/style.
Original numeric method identifiers are retained in the input CSV and mapping.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ORDER = ('phase5e_rf', 'phase9d_plus', 'phase9g', 'phase10d_strict',
         'new_residual_rf', 'new_hjd_rf')
NEW = ('new_residual_rf','new_hjd_rf')
BASELINE = 'phase10d_strict'
METRICS = ('rmse_stored_units','mae_stored_units','x_mean_pct','x_max_pct',
           'x_profile_rmse_stored_units','y_profile_rmse_stored_units','z_profile_rmse_stored_units')
CHANGE_KEYS = ('rmse_stored_units','mae_stored_units','x_profile_rmse_stored_units',
               'y_profile_rmse_stored_units','z_profile_rmse_stored_units','x_mean_pct')
CHANGE_LABELS = ('Whole-cube dose error\n(root mean square)',
                 'Whole-cube dose error\n(mean absolute)',
                 'Dose-line error, array x\n(root mean square)',
                 'Dose-line error, array y\n(root mean square)',
                 'Dose-line error, array z\n(root mean square)',
                 'Dose-line percentage error, array x')
PARAMETERS = {'new_residual_rf':340545,'new_hjd_rf':104144}
FOOTER = ('600 validation cubes from 2 cases; equal-case averages of per-cube errors; one training seed for the new methods.\n'
          '40 model-selection cubes are included. Development comparison, not an independent final test; no uncertainty intervals.\n'
          'The shared starting prediction is learned, not water dose. Array directions are not verified physical beam directions.')
FIGURES = [
 ('01_whole_cube_squared_error','Overall dose error: root mean square',
  'All six methods. Each cube contributes a root-mean-square error, averaged within case and then equally across the two cases. Bars start at zero; the displayed factor of one million does not establish physical units.'),
 ('02_whole_cube_absolute_error','Overall dose error: mean absolute',
  'All six methods in the same fixed order. Smaller values are better. Errors are in stored numerical units, multiplied by one million only for display.'),
 ('03_x_dose_line_percentage_error','Average percentage error along the x-direction dose line',
  'The line passes through the target-dose peak along the original array x direction (array W). Percentage errors include only positions at or above1% of the target line peak; this is not a verified physical beam depth.'),
 ('04_dose_line_errors','Absolute dose-line errors in three array directions',
  'Root-mean-square differences use the entire line. The three legacy array directions x/y/z correspond to W/H/D. All six methods are shown, including the upstream predictor; the relative-change figure resolves smaller differences.'),
 ('05_changes_from_previous_system','Changes relative to the previous final system',
  'For each metric separately, change=100*(new/previous final system-1). Negative means a lower error, positive a higher error. These are relative percentages, not percentage-point differences. They must not be summed into an overall score.'),
 ('06_volume_and_profile_accuracy','Whole-volume accuracy versus dose-profile accuracy',
  'A zoomed detail view of the five correction systems. The upstream predictor remains in figures01-04. Both axes are errors and smaller is better. No connecting frontier, uncertainty interval or overall winner is inferred.'),
 ('07_correction_model_size','Size of the two newly trained correction models',
  'User-reported trainable correction-network parameters only. Shared upstream parameters, particle-reconstruction cost and runtime are not included. Parameter count is not a measure of latency.'),
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path: Path) -> dict:
    path=Path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_size>2_000_000:
        raise ValueError('Expected a small, existing aggregate CSV, not medical or per-record data.')
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        fields=reader.fieldnames or []
        if len(fields)!=len(set(fields)) or not set(('method','n_records','n_cases',*METRICS)).issubset(fields):
            raise ValueError('Missing or duplicate CSV columns.')
        rows={}
        for row in reader:
            name=row['method']
            if name not in ORDER or name in rows:raise ValueError('Unknown or duplicate method: '+name)
            r={k:float(row[k]) for k in METRICS}
            if any(not math.isfinite(v) or v<0 for v in r.values()):raise ValueError('Nonfinite or negative metric.')
            if float(row['n_records'])!=600 or float(row['n_cases'])!=2:raise ValueError('Wrong validation scope.')
            rows[name]=r
    if set(rows)!=set(ORDER):raise ValueError('All six existing methods are required.')
    return {name:rows[name] for name in ORDER}


def relative_changes(rows: dict) -> dict:
    if any(rows[BASELINE][k]<=0 for k in CHANGE_KEYS):raise ValueError('Baseline must be positive for ratios.')
    return {name:{k:100*(rows[name][k]/rows[BASELINE][k]-1) for k in CHANGE_KEYS} for name in NEW}


def figure(title, subtitle, height=8.4, left=.36, width=.58):
    fig=plt.figure(figsize=(14.0,height))
    ax=fig.add_axes([left,.27,width,.51])
    fig.text(.025,.965,title,fontsize=19,fontweight='bold',va='top')
    fig.text(.025,.892,subtitle,fontsize=11.5,va='top')
    fig.text(.025,.035,FOOTER,fontsize=9.0,va='bottom')
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    return fig,ax


def save(fig, out, stem):
    assert len(fig.axes)==1
    for ext in ('png','svg'):
        fig.savefig(out/f'{stem}.{ext}',dpi=170,bbox_inches='tight',
                    metadata={'Software':'Dose-prediction meeting presentation'} if ext=='png' else {'Creator':'Dose-prediction meeting presentation','Date':None})
    plt.close(fig)


def bar(rows,names,key,scale,title,xlabel,out,stem):
    fig,ax=figure(title,'All six methods; smaller errors are better. Descriptive names refer to the unchanged methods.')
    vals=[rows[n][key]*scale for n in ORDER]
    bars=ax.barh(np.arange(6),vals,height=.63)
    bars[3].set_hatch('//')
    for i in (4,5):bars[i].set_hatch('..')
    ax.set_yticks(range(6),[names[n]['plot'] for n in ORDER],fontsize=10.5)
    ax.invert_yaxis();ax.set_xlim(0,max(vals)*1.2)
    ax.set_xlabel(xlabel,fontsize=11.5,labelpad=12)
    ax.xaxis.grid(True,alpha=.22);ax.set_axisbelow(True)
    for b,v in zip(bars,vals):
        ax.text(v+max(vals)*.012,b.get_y()+b.get_height()/2,f'{v:.5f}',va='center',fontsize=10.5)
    save(fig,out,stem)


def build(root: Path, out: Path | None=None):
    root=Path(root).resolve();out=Path(out).resolve() if out else root/'figures'
    data=root/'data/comparison_reported.csv'
    labels_path=root/'method_names.json'
    names=json.loads(labels_path.read_text(encoding='utf-8'))
    if set(names)!=set(ORDER):raise ValueError('Display-name mapping does not match the unchanged method identifiers.')
    rows=load_rows(data);changes=relative_changes(rows)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a new or empty figure directory; existing presentations are preserved.')
    out.mkdir(parents=True,exist_ok=True)
    bar(rows,names,'rmse_stored_units',1e6,FIGURES[0][1],
        'Mean per-cube root-mean-square error [10⁻⁶ stored dose units]',out,FIGURES[0][0])
    bar(rows,names,'mae_stored_units',1e6,FIGURES[1][1],
        'Mean per-cube absolute error [10⁻⁶ stored dose units]',out,FIGURES[1][0])
    bar(rows,names,'x_mean_pct',1,FIGURES[2][1],
        'Mean local percentage error [%]; x is the original array direction',out,FIGURES[2][0])
    fig,ax=figure(FIGURES[3][1],
        'Whole-line root-mean-square error; lines pass through the target-dose peak. Smaller is better.',height=10.1)
    keys=METRICS[-3:];mv=max(rows[n][k]*1e6 for n in ORDER for k in keys)
    for key,offset,axis in zip(keys,[-.23,0,.23],'xyz'):
        vals=[rows[n][key]*1e6 for n in ORDER]
        bars=ax.barh(np.arange(6)+offset,vals,height=.21,label='Array '+axis)
        for b,v in zip(bars,vals):
            ax.text(v+mv*.012,b.get_y()+b.get_height()/2,f'{v:.3f}',fontsize=8.8,va='center')
    ax.set_yticks(range(6),[names[n]['plot'] for n in ORDER],fontsize=10.5);ax.invert_yaxis()
    ax.set_xlim(0,mv*1.16);ax.xaxis.grid(True,alpha=.22);ax.set_axisbelow(True)
    ax.set_xlabel('Mean per-cube dose-line root-mean-square error [10⁻⁶ stored dose units]',labelpad=12,fontsize=11)
    ax.legend(ncol=3,loc='upper left',bbox_to_anchor=(0,-.14),frameon=False)
    save(fig,out,FIGURES[3][0])
    fig,ax=figure(FIGURES[4][1],
        'Each metric: 100 × (new error / previous final-system error − 1). These are relative percentages.',height=9.6,left=.37,width=.57)
    for j,name in enumerate(NEW):
        vals=[changes[name][k] for k in CHANGE_KEYS]
        bars=ax.barh(np.arange(6)+(j-.5)*.30,vals,height=.28,label=names[name]['plot'])
        for b,v in zip(bars,vals):
            ax.text(v+(.40 if v>=0 else -.40),b.get_y()+b.get_height()/2,
                    f'{v:+.2f}%',ha='left' if v>=0 else 'right',va='center',fontsize=10)
    ax.set_xlim(-22,22);ax.axvline(0,linewidth=1)
    ax.set_yticks(range(6),CHANGE_LABELS,fontsize=10.5);ax.invert_yaxis()
    ax.set_xlabel('Relative change [%]     ← lower error | higher error →',fontsize=11.5,labelpad=12)
    ax.xaxis.grid(True,alpha=.22);ax.set_axisbelow(True)
    ax.legend(ncol=2,loc='upper left',bbox_to_anchor=(-.08,-.20),frameon=False,fontsize=10.5)
    save(fig,out,FIGURES[4][0])
    fig,ax=figure(FIGURES[5][1],
        'Zoomed detail of five correction systems. The predictor before refinement is retained in figures 1–4.',height=8.9,left=.13,width=.79)
    points={'phase9d_plus':(24,-23),'phase9g':(30,35),'phase10d_strict':(25,-15),
            'new_residual_rf':(-10,27),'new_hjd_rf':(-110,28)}
    for marker,name in zip(('o','s','D','^','v'),ORDER[1:]):
        x=rows[name]['rmse_stored_units']*1e6;y=rows[name]['x_mean_pct']
        ax.scatter([x],[y],s=100,marker=marker)
        ax.annotate(names[name]['plot'],(x,y),xytext=points[name],textcoords='offset points',
                    fontsize=10.5,arrowprops={'arrowstyle':'-','linewidth':.8})
    ax.set_xlim(3.95,4.64);ax.set_ylim(6.20,8.50)
    ax.set_xlabel('Whole-cube root-mean-square dose error [10⁻⁶ stored units]\nSmaller is better',fontsize=11.5,labelpad=10)
    ax.set_ylabel('Mean percentage error along the x-direction dose line [%]\nSmaller is better',fontsize=11,labelpad=12)
    ax.grid(True,alpha=.22)
    save(fig,out,FIGURES[5][0])
    fig,ax=figure(FIGURES[6][1],
        'Correction-network parameters only; the shared predictor is excluded. Model size does not measure speed.',height=7.4)
    vals=[PARAMETERS[n] for n in NEW]
    bars=ax.barh(range(2),vals,height=.43)
    ax.set_yticks(range(2),[names[n]['plot'] for n in NEW],fontsize=12);ax.invert_yaxis()
    ax.set_xlim(0,max(vals)*1.24);ax.set_xlabel('Number of trainable correction parameters',fontsize=12,labelpad=12)
    ax.xaxis.grid(True,alpha=.22);ax.set_axisbelow(True)
    for b,v in zip(bars,vals):ax.text(v+max(vals)*.02,b.get_y()+b.get_height()/2,f'{v:,}',va='center',fontsize=12)
    save(fig,out,FIGURES[6][0])
    receipt={'scope':'Presentation relabeling only, no new model results.',
             'input_csv_sha256':digest(data),'labels_sha256':digest(labels_path),
             'plot_code_sha256':digest(Path(__file__)),
             'matplotlib_version':matplotlib.__version__,
             'source_kind':'user_reported_aggregate',
             'files':{p.name:digest(p) for p in sorted(out.iterdir()) if p.suffix in ('.png','.svg')}}
    (out/'FIGURES_RECORD.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True,help='New or empty output directory; never overwrite earlier plots.')
    args=parser.parse_args();print(build(Path(__file__).parent,args.out))

if __name__=='__main__':main()
