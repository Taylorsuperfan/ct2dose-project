"""Compare restored R0 Python source with this handoff; no weights/data are read."""
from pathlib import Path
import argparse
import hashlib
import json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--code',type=Path,required=True)
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    origin=json.loads((root/'provenance/source_archives.json').read_text())[0]
    checked=0
    for item in origin['copied_files']:
        if not item['path'].endswith('.py'):continue
        target=a.code/item['path']
        if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest()!=item['sha256']:
            raise RuntimeError('Restored source differs: '+item['path'])
        checked+=1
    print(f'PASS: {checked} Python source files match the supplied R0 distribution.')
    print('This does not verify private training outputs or grant publication permission.')
if __name__=='__main__':main()
