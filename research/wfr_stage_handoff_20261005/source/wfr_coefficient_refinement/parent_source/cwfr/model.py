"""CT-conditioned spatial velocity/growth and an independent grid-valued sign head."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .geometry import centers, deposit

class Block(nn.Module):
    def __init__(self,cin,cout):
        super().__init__()
        self.net=nn.Sequential(nn.Conv3d(cin,cout,3,padding=1),nn.SiLU(),
                               nn.Conv3d(cout,cout,3,padding=1),nn.SiLU())
    def forward(self,x): return self.net(x)

class Encoder(nn.Module):
    def __init__(self,c):
        super().__init__();self.first=Block(2,c);self.low=Block(c,2*c);self.mix=Block(3*c,c)
    def forward(self,x):
        hi=self.first(x);lo=self.low(F.avg_pool3d(hi,2))
        local=self.mix(torch.cat([hi,F.interpolate(lo,size=hi.shape[2:],mode='trilinear',align_corners=False)],dim=1))
        return local,lo.mean(dim=(2,3,4))

class ConditionalWFR(nn.Module):
    def __init__(self,channels=16,hidden=64):
        super().__init__();self.transport_encoder=Encoder(channels);self.sign_encoder=Encoder(channels)
        self.field=nn.Sequential(nn.Linear(3*channels+4,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,4))
        self.sign_out=nn.Conv3d(channels,1,1)
        nn.init.zeros_(self.field[-1].weight);nn.init.zeros_(self.field[-1].bias)
        nn.init.zeros_(self.sign_out.weight);nn.init.zeros_(self.sign_out.bias)
    def encode_transport(self,condition): return self.transport_encoder(condition)
    def sign_logits(self,condition): return self.sign_out(self.sign_encoder(condition)[0])
    def vector_field(self,points,t,features):
        """points use array-axis order. PyTorch grid_sample expects the reversed order."""
        local,global_feature=features;B,P,_=points.shape
        grid=(2*points.flip(-1)-1).reshape(B,P,1,1,3)
        samples=F.grid_sample(local,grid,mode='bilinear',padding_mode='border',align_corners=False)
        samples=samples[:,:, :,0,0].transpose(1,2)
        if t.ndim==1:t=t[:,None,None].expand(B,P,1)
        if t.ndim==2:t=t[...,None]
        inputs=torch.cat([points,t,samples,global_feature[:,None].expand(-1,P,-1)],dim=-1)
        output=self.field(inputs);logit_velocity=output[...,:3];growth=output[...,3]
        # Box-tangent parameterization: x=sigmoid(y), dx/dt=x*(1-x)*dy/dt.
        velocity=points*(1-points)*logit_velocity
        return velocity,growth,logit_velocity

    @torch.no_grad()
    def predict(self,condition,steps=16,query_chunk=8192):
        """Inference accepts observable conditions only; no target mass/sign/residual."""
        if condition.ndim!=5 or condition.shape[:2]!=(1,2): raise ValueError('Predict one two-channel 3-D condition at a time.')
        if type(steps) is not int or steps<1 or type(query_chunk) is not int or query_chunk<1: raise ValueError('Positive integer solver steps and chunk size required.')
        self.eval();shape=tuple(condition.shape[2:]);device=condition.device
        xyz=torch.tensor(centers(shape),dtype=condition.dtype,device=device)
        features=self.encode_transport(condition)
        sign=torch.tanh(self.sign_logits(condition)/2)[0,0].cpu().numpy().astype(np.float64)
        all_points=[];all_weights=[];dt=1./steps
        for start in range(0,len(xyz),query_chunk):
            x=xyz[start:start+query_chunk][None];y=torch.logit(x);logw=torch.zeros_like(x[...,0])
            for k in range(steps):
                t=torch.full((1,),k*dt,device=device,dtype=x.dtype)
                _,g,u=self.vector_field(torch.sigmoid(y),t,features)
                _,g2,u2=self.vector_field(torch.sigmoid(y+dt*u),t+dt,features)
                y=y+dt*(u+u2)/2;logw=logw+dt*(g+g2)/2
                if not torch.isfinite(y).all() or not torch.isfinite(logw).all(): raise FloatingPointError('Nonfinite transport trajectory.')
            x=torch.sigmoid(y)[0].cpu().numpy().astype(np.float64)
            lw=logw[0].cpu().numpy().astype(np.float64)
            if (np.abs(lw)>40).any(): raise FloatingPointError('Extreme log mass. No mass clipping or target-total correction was applied.')
            all_points.append(x);all_weights.append(np.exp(lw))
        points=np.concatenate(all_points);weights=np.concatenate(all_weights)
        magnitude=deposit(points,weights,shape,'trilinear')
        u=magnitude*sign
        return {'residual_normalized':u,'magnitude':magnitude,'sign':sign,
                'particle_positions':points,'particle_weights':weights,
                'diagnostics':{'inference_source_particles':len(points),'source_total':float(len(points)),
                  'predicted_magnitude_total':float(weights.sum()),'grid_magnitude_total':float(magnitude.sum()),
                  'absolute_signed_total':float(np.abs(u).sum()),'signed_total':float(u.sum()),
                  'weighted_absolute_sign':float(np.abs(u).sum()/weights.sum()),
                  'velocity_growth_calls':2*steps,'query_chunks':int(np.ceil(len(points)/query_chunk)),
                  'reconstruction':'trilinear_magnitude_then_grid_soft_sign','solver':'Heun_in_logit_position_and_log_mass',
                  'target_used_at_inference':False,'output_clamped_in_this_function':False}}
