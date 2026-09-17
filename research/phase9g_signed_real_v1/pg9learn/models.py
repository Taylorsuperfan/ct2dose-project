"""Signed dense RF control and spatial HJD two-flow conditional prototype.

HJD follows the advisor's positive/negative spatial distribution construction.
Engineering choices: CT+frozen Phase9G conditioning, uniform spatial source,
learned mean-mass heads, and endpoint auxiliary losses. Not a water model, not
an unbalanced OT solver, and the auxiliary rollout is NOT simulation-free.
"""
import math
from functools import lru_cache
import torch
from torch import nn
from torch.nn import functional as F
from p10recover.legacy_defs import ConditionalUNetFlow3D


def decompose(r):
    parts=torch.cat([r.clamp_min(0),(-r).clamp_min(0)],1)
    total=parts.flatten(2).sum(-1);active=total>0
    p=parts/torch.where(active,total,torch.ones_like(total))[:,:,None,None,None]
    return parts,p,total/r[0,0].numel(),active


def sample_target(p,active,n):
    b,c,d,h,w=p.shape;flat=p.flatten(2).reshape(b*c,-1)
    prob=torch.where(active.reshape(-1,1),flat,torch.ones_like(flat))
    ids=torch.multinomial(prob,n,replacement=True)
    x=ids%w;y=(ids//w)%h;z=ids//(w*h)
    pts=(torch.stack([x,y,z],-1).to(p.dtype)+torch.rand(b*c,n,3,device=p.device,dtype=p.dtype))/p.new_tensor([w,h,d])
    return pts.reshape(b,c,n,3)


@lru_cache(maxsize=16)
def _sobol_cpu(n,seed):
    return torch.quasirandom.SobolEngine(3,scramble=True,seed=int(seed)).draw(n)


def sobol(n,seed,like):return _sobol_cpu(n,seed).to(device=like.device,dtype=like.dtype)


def splat(points,shape):
    """Trilinear unit-sum deposition. Clamped boundary deposition is reported."""
    b,n,_=points.shape;d,h,w=map(int,shape)
    if n<1 or not torch.isfinite(points).all():raise ValueError('Invalid particle locations.')
    outside=((points<0)|(points>1)).any(-1).to(points.dtype).mean(-1)
    pos=points*points.new_tensor([w,h,d])-0.5
    pos=pos.maximum(torch.zeros_like(pos)).minimum(points.new_tensor([w-1,h-1,d-1]))
    lo=pos.floor().long();frac=pos-lo.to(pos.dtype);out=points.new_zeros((b,d*h*w))
    for dz in (0,1):
        for dy in (0,1):
            for dx in (0,1):
                idx=(lo+lo.new_tensor([dx,dy,dz])).minimum(lo.new_tensor([w-1,h-1,d-1]))
                a=frac[...,0] if dx else 1-frac[...,0]
                bb=frac[...,1] if dy else 1-frac[...,1]
                c=frac[...,2] if dz else 1-frac[...,2]
                linear=idx[...,2]*(h*w)+idx[...,1]*w+idx[...,0]
                out=out.scatter_add(1,linear,a*bb*c/n)
    out=out/out.sum(-1,keepdim=True).clamp_min(torch.finfo(points.dtype).tiny)
    return out.view(b,1,d,h,w),outside


class DenseResidualRF(nn.Module):
    def __init__(self,width=16):
        super().__init__()
        # Same known U-Net building blocks; TWO conditions instead of CT alone.
        # Concatenation is [state, CT, frozen_Phase9G/base_scale, time].
        self.net=ConditionalUNetFlow3D(in_ch=4,out_ch=1,base_ch=width)
        nn.init.zeros_(self.net.out_conv.weight);nn.init.zeros_(self.net.out_conv.bias)
    def forward(self,x,t,condition):return self.net(x,condition,t)
    def predict(self,condition,steps,particles=0,seed=31415):
        x=torch.zeros_like(condition[:,:1]);dt=1./steps
        for k in range(steps):
            t=x.new_full((x.shape[0],),k*dt)
            x=x+dt*self(x,t,condition)  # Signed state: NEVER clamp inside the ODE.
        return dict(residual=x,parts=torch.cat([x.clamp_min(0),(-x).clamp_min(0)],1),
                    outside=x.new_zeros((x.shape[0],2)),boundary=x.sum()*0)


class Block(nn.Module):
    def __init__(self,a,b):
        super().__init__();g=4 if b%4==0 else 1
        self.net=nn.Sequential(nn.Conv3d(a,b,3,padding=1),nn.GroupNorm(g,b),nn.SiLU(),
                               nn.Conv3d(b,b,3,padding=1),nn.GroupNorm(g,b),nn.SiLU())
    def forward(self,x):return self.net(x)


class SpatialVelocity(nn.Module):
    def __init__(self,features,glob,hidden):
        super().__init__();self.net=nn.Sequential(nn.Linear(3+1+features+glob,hidden),nn.SiLU(),
                        nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,3))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x,t,local,global_features):
        b,n,_=x.shape
        feat=F.grid_sample(local,(2*x-1).reshape(b,n,1,1,3),mode='bilinear',padding_mode='border',align_corners=False)
        feat=feat[:,:, :,0,0].transpose(1,2)
        inp=torch.cat([x,t.reshape(b,1,1).expand(b,n,1),feat,global_features[:,None].expand(b,n,-1)],-1)
        return self.net(inp)


class HJDSpatialRF(nn.Module):
    def __init__(self,width=8,features=8,hidden=64,mass_init=(0.1,0.1)):
        super().__init__()
        self.e0=Block(2,width);self.e1=Block(width,2*width);self.mid=Block(2*width,4*width)
        self.d1=Block(6*width,2*width);self.d0=Block(3*width,width);self.local=nn.Conv3d(width,features,1)
        self.mass=nn.Sequential(nn.Linear(4*width,2*width),nn.SiLU(),nn.Linear(2*width,2))
        nn.init.zeros_(self.mass[-1].weight)
        with torch.no_grad():
            a=torch.as_tensor(mass_init,dtype=torch.float32).clamp_min(1e-5)
            self.mass[-1].bias.copy_(a+torch.log(-torch.expm1(-a)))
        self.positive_flow=SpatialVelocity(features,4*width,hidden)
        self.negative_flow=SpatialVelocity(features,4*width,hidden)
    def encode(self,c):
        a=self.e0(c);b=self.e1(F.avg_pool3d(a,2));mid=self.mid(F.avg_pool3d(b,2))
        d=self.d1(torch.cat([F.interpolate(mid,size=b.shape[-3:],mode='nearest'),b],1))
        d=self.d0(torch.cat([F.interpolate(d,size=a.shape[-3:],mode='nearest'),a],1))
        glob=mid.mean((2,3,4));return self.local(d),glob,F.softplus(self.mass(glob))
    def integrate(self,enc,steps,particles,seed):
        local,glob,mass=enc;b=local.shape[0];shape=local.shape[-3:];dt=1./steps
        source=sobol(particles,seed,local)[None].expand(b,-1,-1)
        probs=[];outs=[];bounds=[]
        for field in [self.positive_flow,self.negative_flow]:
            x=source.clone()
            for k in range(steps):
                x=x+dt*field(x,x.new_full((b,),k*dt),local,glob)
                bounds.append((F.relu(-x).square()+F.relu(x-1).square()).mean())
            p,o=splat(x,shape);probs.append(p);outs.append(o)
        prob=torch.cat(probs,1);parts=prob*(mass*math.prod(shape))[:,:,None,None,None]
        return dict(residual=parts[:,:1]-parts[:,1:],parts=parts,probabilities=prob,mass_mean=mass,
                    outside=torch.stack(outs,1),boundary=torch.stack(bounds).mean())
    def predict(self,condition,steps,particles,seed=31415):return self.integrate(self.encode(condition),steps,particles,seed)


def build(method,recipe,norm):
    if method=='residual_rf':return DenseResidualRF(recipe['dense_width'])
    if method=='hjd_rf':return HJDSpatialRF(recipe['hjd_width'],recipe['features'],recipe['hidden'],norm['initial_mass_mean_scaled'])
    raise ValueError('Only residual_rf and hjd_rf are implemented; no fallback model.')


def loss(model,batch,method,recipe):
    c,r=batch['condition'],batch['residual'];b=r.shape[0]
    if method=='residual_rf':
        t=torch.rand(b,device=r.device);xt=t[:,None,None,None,None]*r
        fm=F.mse_loss(model(xt,t,c),r)
        return fm,dict(flow=fm,total=fm)
    truth,p,mass,active=decompose(r);enc=model.encode(c);local,glob,mp=enc
    dest=sample_target(p,active,recipe['fm_particles']);fm=mp.sum()*0
    for j,field in enumerate([model.positive_flow,model.negative_flow]):
        end=dest[:,j];start=torch.rand_like(end);t=torch.rand(b,device=r.device)
        x=(1-t[:,None,None])*start+t[:,None,None]*end
        err=(field(x,t,local,glob)-(end-start)).square().mean((1,2));a=active[:,j].to(err.dtype)
        fm=fm+(err*a).sum()/a.sum().clamp_min(1)
    # Endpoint and final evaluation use the SAME particle count and step count.
    z=model.integrate(enc,recipe['ode_steps'],recipe['particles'],recipe['source_seed'])
    comp=(z['parts']-truth).square().mean();recon=(z['residual']-r).square().mean()
    ml=F.mse_loss(mp,mass)  # Masses measured in train residual-RMS units, not tiny dose units.
    overlap=(z['parts'][:,:1]*z['parts'][:,1:]).mean()
    ls=dict(flow=fm,mass=ml,component=comp,reconstruction=recon,overlap=overlap,boundary=z['boundary'])
    total=fm+recipe['lambda_mass']*ml+recipe['lambda_component']*comp+recipe['lambda_reconstruction']*recon+recipe['lambda_overlap']*overlap+recipe['lambda_boundary']*z['boundary']
    return total,dict(total=total,**ls)
