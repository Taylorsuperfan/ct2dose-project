"""Command-line entry points; no training or historical-test evaluation command."""
import argparse,json
from pathlib import Path
from .verify import verify_source


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='cmd',required=True)
    sub.add_parser('verify-source')
    q=sub.add_parser('inspect');q.add_argument('--root',type=Path,required=True);q.add_argument('--out',type=Path,required=True);q.add_argument('--trusted',action='store_true')
    q=sub.add_parser('evaluate');q.add_argument('--root',type=Path,required=True);q.add_argument('--out',type=Path,required=True)
    q.add_argument('--mode',choices=['smoke','historical-val600'],default='smoke');q.add_argument('--device',default='cpu');q.add_argument('--trusted',action='store_true');q.add_argument('--no-plots',action='store_true')
    q=sub.add_parser('compare-external');q.add_argument('--baseline-run',type=Path,required=True);q.add_argument('--predictions',type=Path,required=True);q.add_argument('--out',type=Path,required=True)
    a=p.parse_args();verify_source()
    if a.cmd=='verify-source':return
    if a.cmd=='inspect':
        from .pipeline import LegacyPipeline
        from .io import write_json,environment,code_hashes
        m=LegacyPipeline(a.root,trusted=a.trusted)
        write_json(a.out,dict(pipeline=m.contract,environment=environment(),source_hashes=code_hashes()))
        print(json.dumps(m.contract['state_stats'],indent=2));print('SAVED:',a.out)
    elif a.cmd=='evaluate':
        from .evaluate import run
        run(a.root,a.out,mode=a.mode,device=a.device,trusted=a.trusted,make_plots=not a.no_plots)
    elif a.cmd=='compare-external':
        from .external import compare_external
        compare_external(a.baseline_run,a.predictions,a.out)

if __name__=='__main__':main()
