"""Released D-A-T CNN–LSTM encoders with a new dual-radar pose decoder."""
import sys
from pathlib import Path
import torch
from torch import nn
import common

class CubeLearnPose(nn.Module):
    def __init__(self,cubelearn_root,lpp_lr):
        super().__init__()
        sys.path.insert(0,str(Path(cubelearn_root).resolve()))
        from network_har import DAT_2DCNNLSTM_HAR, LPP_MODULES
        self.lpp_modules=LPP_MODULES
        # Separate branches let the radar orientations learn different features.
        self.hori=DAT_2DCNNLSTM_HAR(n_frames=8,n_classes=6)
        self.vert=DAT_2DCNNLSTM_HAR(n_frames=8,n_classes=6)
        self.hori.fc_3=nn.Identity()
        self.vert.fc_3=nn.Identity()
        self.seed_map=nn.Linear(256,32*8*8)
        self.decoder=nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),
            nn.Conv2d(32,32,3,padding=1),nn.ReLU(),
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),
            nn.Conv2d(32,16,3,padding=1),nn.ReLU(),
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),
            nn.Conv2d(16,14,3,padding=1))
        # Change trainability after all initialization to keep paired seeds identical.
        for p in self.complex_parameters():p.requires_grad_(lpp_lr>0)
    def complex_parameters(self):
        return [p for branch in (self.hori,self.vert) for name in self.lpp_modules
                for p in getattr(branch,name).parameters()]
    def parameter_groups(self,lr,lpp_lr):
        complex_params=self.complex_parameters(); ids={id(p) for p in complex_params}
        groups=[{'params':[p for p in self.parameters() if id(p) not in ids],'lr':lr}]
        if lpp_lr>0:groups.append({'params':complex_params,'lr':lpp_lr})
        return groups
    def forward(self,hori,vert):
        x=torch.cat([self.hori(hori),self.vert(vert)],dim=1)
        x=self.seed_map(x).reshape(-1,32,8,8)
        return torch.sigmoid(self.decoder(x))
