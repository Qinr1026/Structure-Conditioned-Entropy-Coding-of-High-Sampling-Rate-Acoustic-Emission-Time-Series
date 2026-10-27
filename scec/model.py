"""SCEC neural context and coset/parity-conditioned probability offsets."""
import numpy as np
import torch
from .context import E8DecoderContextPrior
from .schedule import progressive_masks

class StateContext(E8DecoderContextPrior):
    def __init__(self, prior, block_size, quantizer='e8', neural=True,
                 conditioned=False, hidden=48):
        super().__init__(prior['pooled_support'][:,:,0],
            prior['pooled_scale'][:,:,0],prior['coset'],prior['steps'],
            block_size,anchor_period=8,hidden=hidden,
            sign_probability=prior['pooled_sign'][:,:,0],predict_sign=True)
        self.quantizer=quantizer; self.neural=neural; self.conditioned=conditioned
        for key in ['support','scale','sign']:
            a=torch.as_tensor(np.asarray(prior[key]),dtype=torch.float32)
            b=torch.as_tensor(np.asarray(prior['pooled_'+key]),dtype=torch.float32)
            if key=='scale': offset=torch.log(a.clamp_min(.03)/b.clamp_min(.03))
            else: offset=torch.logit(a.clamp(1e-5,1-1e-5))-torch.logit(b.clamp(1e-5,1-1e-5))
            self.register_buffer('state_'+key+'_offset',offset)

    def common_fields(self,q,c,profile,known):
        groups=q.shape[1]
        if self.neural and bool(known.any()):
            f=super().forward(q,c,profile,context_mask=known.view(1,groups,1))
            return f.support_probability,f.magnitude_scale,f.coset_probability,f.sign_probability
        p,s,pc=self.base_fields(profile,groups)
        return p,s,pc,self.base_sign_fields(profile,groups)

    def condition(self,fields,q,c,profile,mode='full'):
        p,s,pc,g=fields
        if not self.conditioned:return fields
        batch,groups,_=q.shape
        st=(2*c.long()).unsqueeze(-1).expand(-1,-1,8).clone()
        if mode=='full':st[:,:,7]+=q[:,:,:7].long().sum(-1).remainder(2)
        band=torch.arange(groups,device=q.device)*self.bands//groups
        j=torch.arange(8,device=q.device).view(1,1,8)
        idx=(profile[:,None,None],band[None,:,None],st,j)
        p=torch.sigmoid(torch.logit(p.clamp(1e-5,1-1e-5))+self.state_support_offset[idx])
        s=s*torch.exp(self.state_scale_offset[idx])
        g=torch.sigmoid(torch.logit(g.clamp(1e-5,1-1e-5))+self.state_sign_offset[idx])
        return p.clamp(1e-5,1-1e-5),s.clamp_min(.03),pc,g.clamp(1e-5,1-1e-5)

    def rate(self,q,c,profile):
        anchor,targets=progressive_masks(q.shape[1],8,q.device)
        known=torch.zeros_like(anchor);total=torch.zeros(q.shape[0],device=q.device)
        for mask in [anchor,*targets]:
            fields=self.common_fields(q,c,profile,known)
            p,s,pc,g=self.condition(fields,q,c,profile)
            total+=(self._coordinate_bits(q,p,s,g)*mask[None,:,None]).sum((1,2))
            if self.quantizer=='e8':total+=(self._coset_bits(c,pc)*mask).sum(1)
            known=known|mask
        return total
