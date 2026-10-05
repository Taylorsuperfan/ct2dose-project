"""WFR travelling paths and a bounded, explicitly approximate weighted OET solver.

Derived from WFR-FM equations 3.4--3.8 and Theorem 3.1, with kappa=delta**2.
The MM update is the unregularized KL marginal-penalty update documented by POT.
This NumPy implementation records scalars only; it does not retain all iterates.
"""
from __future__ import annotations
import numpy as np


def centers(shape):
    return np.stack(np.meshgrid(*[(np.arange(n)+.5)/n for n in shape],indexing='ij'),axis=-1).reshape(-1,3)


def cost_matrix(x,y,delta):
    d=np.sqrt(np.sum((x[:,None]-y[None])**2,axis=-1))
    if not np.isfinite(d).all() or np.any(d>=np.pi*delta): raise ValueError('Unsupported WFR cutoff pair.')
    return -2*np.log(np.cos(d/(2*delta)))


def objective(G,a,b,C):
    r=G.sum(1);c=G.sum(0)
    if np.any(r<=0) or np.any(c<=0): raise FloatingPointError('Nonpositive solver marginal.')
    return float(np.sum(C*G)+np.sum(r*np.log(r/a)-r+a)+np.sum(c*np.log(c/b)-c+b))


def mm_uot(a,b,C,max_iter=1000,rtol=1e-5):
    a=np.asarray(a,np.float64);b=np.asarray(b,np.float64);C=np.asarray(C,np.float64)
    if a.ndim!=1 or b.ndim!=1 or C.shape!=(len(a),len(b)) or (a<=0).any() or (b<=0).any() or not all(np.isfinite(v).all() for v in (a,b,C)):
        raise ValueError('The bounded travelling solver requires positive finite endpoint weights.')
    G=np.outer(a,b); K=np.sqrt(a[:,None]*b[None,:])*np.exp(-C/2)
    initial=objective(G,a,b,C);previous=initial;converged=False
    for k in range(max_iter):
        den=np.sqrt(G.sum(1)[:,None]*G.sum(0)[None,:])
        if np.any(den<=0): raise FloatingPointError('Underflow in coupling marginals.')
        updated=G*K/den
        rel=float(np.linalg.norm(updated-G)/max(np.linalg.norm(G),np.finfo(float).tiny))
        G=updated
        if rel<rtol: converged=True;break
    final=objective(G,a,b,C)
    if final>initial+1e-10*max(1,abs(initial)): raise FloatingPointError('MM objective increased beyond tolerance.')
    rows=G.sum(1); cols=G.sum(0)
    gamma0=G*(a/rows)[:,None];gamma1=G*(b/cols)[None,:]
    if not (np.allclose(gamma0.sum(1),a,rtol=1e-10,atol=1e-12) and np.allclose(gamma1.sum(0),b,rtol=1e-10,atol=1e-12)):
        raise FloatingPointError('Semi-coupling endpoint constraints failed.')
    return G,{'iterations':k+1,'relative_iterate_change':rel,'converged_by_iterate_test':converged,
              'initial_objective':initial,'final_objective':final,
              'optimality_claim':'finite-iteration approximate OET coupling; no exact optimality certificate'},gamma0,gamma1


def make_bank(residual,cfg,rng):
    shape=residual.shape;rho=np.abs(residual.astype(np.float64)).reshape(-1);N=len(rho)
    if not np.isfinite(rho).all() or rho.sum()<=0:
        raise ValueError('Exactly zero/nonfinite target: this nonzero pilot stops instead of adding epsilon mass.')
    grid=centers(shape);ns=min(cfg.source_points_per_bank,N);nt=cfg.target_draws_per_bank
    src=rng.choice(N,size=ns,replace=False)
    drawn=rng.choice(N,size=nt,replace=True,p=rho/rho.sum())
    target,counts=np.unique(drawn,return_counts=True)
    # Both full measures are divided by the SAME N. Target total is not normalized away.
    a=np.full(ns,1/ns); b=(rho.sum()/N)*(counts/nt)
    C=cost_matrix(grid[src],grid[target],cfg.delta)
    G,stats,g0,g1=mm_uot(a,b,C,cfg.coupling_iterations,cfg.coupling_relative_tolerance)
    q=g0/a.sum();cdf=np.cumsum(q.reshape(-1));cdf[-1]=1.
    row_ratio=G.sum(1)/a;col_ratio=b/G.sum(0)
    endpoint=float(np.sum(q*(row_ratio[:,None]*col_ratio[None,:])))
    if not np.isclose(endpoint,b.sum(),rtol=1e-10): raise FloatingPointError('Sampled path endpoint total differs.')
    stats.update(source_count=ns,target_unique_count=len(target),target_draws=nt,source_total=1.,
                 target_total=float(b.sum()),normalized_target_voxel_sum=float(rho.sum()),
                 max_matrix_shape=[ns,len(target)],path_endpoint_total=endpoint)
    return {'x0':grid[src],'x1':grid[target],'cdf':cdf,'row_ratio':row_ratio,'col_ratio':col_ratio},stats


def sample_pairs(bank,n,rng):
    ids=np.searchsorted(bank['cdf'],rng.random(n),side='right');n1=len(bank['x1'])
    i=ids//n1;j=ids%n1
    return bank['x0'][i],bank['x1'][j],bank['row_ratio'][i]*bank['col_ratio'][j]


def path_targets(x0,x1,m1,t,delta):
    """Initial conditional mass is one because q is the source semi-coupling."""
    x0=np.asarray(x0,np.float64);x1=np.asarray(x1,np.float64);m1=np.asarray(m1,np.float64).reshape(-1)
    t=np.asarray(t,np.float64).reshape(-1)
    if x0.shape!=x1.shape or x0.shape!=(len(m1),3) or t.shape!=m1.shape or (m1<=0).any() or (t<0).any() or (t>1).any():
        raise ValueError('Invalid travelling-path batch.')
    displacement=x1-x0;d=np.linalg.norm(displacement,axis=1);theta=d/(2*delta)
    if np.any(theta>=np.pi/2): raise ValueError('WFR travel cutoff exceeded.')
    root=np.sqrt(m1);re=1-t+t*root*np.cos(theta);im=t*root*np.sin(theta)
    mass=re*re+im*im
    if not np.isfinite(mass).all() or np.any(mass<=0): raise FloatingPointError('Nonfinite/zero conditional mass.')
    direction=np.divide(displacement,d[:,None],out=np.zeros_like(displacement),where=d[:,None]>0)
    pos=x0+2*delta*np.arctan2(im,re)[:,None]*direction
    velocity=(2*delta*root*np.sin(theta)/mass)[:,None]*direction
    growth=2*(re*(root*np.cos(theta)-1)+im*root*np.sin(theta))/mass
    return pos,velocity,growth,mass


def deposit(points,weights,shape,method='trilinear'):
    points=np.asarray(points,np.float64);weights=np.asarray(weights,np.float64).reshape(-1)
    if points.shape!=(len(weights),3) or not np.isfinite(points).all() or not np.isfinite(weights).all() or (points<0).any() or (points>1).any():
        raise ValueError('Invalid/out-of-box deposition point. No silent clipping of particles.')
    out=np.zeros(shape,np.float64);n=np.asarray(shape)
    if method=='histogram':
        ids=np.minimum((points*n).astype(int),n-1);np.add.at(out,tuple(ids.T),weights)
    elif method=='trilinear':
        import itertools
        u=points*n-.5;lo=np.floor(u).astype(int);f=u-lo
        for bits in itertools.product((0,1),repeat=3):
            idx=np.clip(lo+bits,0,n-1);co=np.prod(np.where(np.asarray(bits),f,1-f),axis=1)
            np.add.at(out,tuple(idx.T),weights*co)
    else: raise ValueError('Unknown reconstruction.')
    if not np.isclose(out.sum(),weights.sum(),rtol=1e-9,atol=1e-9): raise FloatingPointError('Deposition did not preserve net mass.')
    return out
