"""Command-line entry points. No default command trains or opens medical arrays."""
import argparse, json
from pathlib import Path
from .config import Config


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    for cmd in ('prepare','train','evaluate','compare','fork'):
        q=sub.add_parser(cmd);q.add_argument('--cache',type=Path,required=True);q.add_argument('--out',type=Path,required=True)
        if cmd in ('prepare','train'):q.add_argument('--config',type=Path,required=True)
        if cmd in ('train','fork'):q.add_argument('--plans',type=Path,required=True)
        if cmd in ('train','evaluate','fork'):q.add_argument('--device',default='cuda')
        if cmd=='train':q.add_argument('--max-updates',type=int,default=128)
        if cmd=='evaluate':q.add_argument('--run',type=Path,required=True)
        if cmd=='compare':
            q.add_argument('--new-evaluation',type=Path,required=True);q.add_argument('--hjd-evaluation',type=Path,required=True);q.add_argument('--direct-evaluation',type=Path,required=True)
        if cmd=='fork':q.add_argument('--parent',type=Path,required=True);q.add_argument('--updates',type=int)
    args=p.parse_args()
    from .data import RealCache
    cache=RealCache(args.cache)
    if args.command in ('prepare','train'):cfg=Config(**json.loads(args.config.read_text())).validate()
    if args.command=='prepare':
        from .plans import prepare
        prepare(cache,args.out,cfg)
    elif args.command=='train':
        from .train import train
        if args.max_updates<1:raise ValueError('max-updates must be positive.')
        train(cache,args.plans,args.out,cfg,args.device,args.max_updates)
    elif args.command=='evaluate':
        from .evaluate import evaluate
        evaluate(cache,args.run,args.out,args.device)
    elif args.command=='compare':
        from .evaluate import compare
        compare(cache,args.new_evaluation,args.hjd_evaluation,args.direct_evaluation,args.out)
    else:
        from .train import fork_run
        fork_run(cache,args.plans,args.parent,args.out,args.device,args.updates)

if __name__=='__main__':main()
