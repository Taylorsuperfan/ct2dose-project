"""Verify extracted definitions against the supplied source-segment hashes."""
from pathlib import Path
import ast,hashlib
from .io import read_json


def verify_source():
    root=Path(__file__).parent.parent;path=root/'p10recover/legacy_defs.py'
    s=path.read_text();lines=s.splitlines(keepends=True)
    found={}
    for n in ast.parse(s).body:
        if isinstance(n,(ast.ClassDef,ast.FunctionDef)):
            lo=min([n.lineno]+[d.lineno for d in getattr(n,'decorator_list',[])])
            seg=''.join(lines[lo-1:n.end_lineno]).rstrip('\n')
            found[n.name]=hashlib.sha256(seg.encode()).hexdigest()
    provenance=read_json(root/'records/source_provenance.json')
    for r in provenance['definitions']:
        if found.get(r['name'])!=r['sha256']:raise ValueError('Recovered source changed: '+r['name'])
    print(f'PASS: {len(provenance["definitions"])} exact source segments verified.')
    return provenance
