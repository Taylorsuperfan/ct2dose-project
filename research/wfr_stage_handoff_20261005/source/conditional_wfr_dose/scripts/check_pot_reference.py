"""Optional CPU comparison to installed POT; not a medical or learning experiment."""
import os
os.environ.update({'POT_BACKEND_DISABLE_JAX':'1','POT_BACKEND_DISABLE_TENSORFLOW':'1',
                  'POT_BACKEND_DISABLE_CUPY':'1','CUDA_VISIBLE_DEVICES':''})
import importlib.metadata
import numpy as np
import ot
from cwfr.geometry import mm_uot,cost_matrix
rng=np.random.default_rng(17);x=rng.random((7,3));y=rng.random((9,3))
a=rng.uniform(.1,1,size=7);a/=a.sum();b=rng.uniform(.1,1,size=9);b*=1.7/b.sum()
C=cost_matrix(x,y,1.)
ours,stats,_,_=mm_uot(a,b,C,max_iter=100,rtol=0)
reference=ot.unbalanced.mm_unbalanced(a,b,C,reg_m=[1.,1.],reg=0,div='kl',numItermax=100,stopThr=0,log=False)
error=float(np.max(np.abs(ours-reference)))
np.testing.assert_allclose(ours,reference,atol=1e-11,rtol=1e-9)
print('POT version:',importlib.metadata.version('POT'))
print('Maximum absolute coupling difference:',error)
print('PASS: fixed-iteration NumPy MM agrees with installed POT on this small example.')
print('This is not a convergence or full-grid optimality certificate.')
