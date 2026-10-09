"""Released R-D-A-T 3D CNN–LSTM encoders with a dual-radar pose decoder."""
import sys
from pathlib import Path
import torch
from torch import nn


def make_encoder(cubelearn_root, n_frames=8):
    sys.path.insert(0, str(Path(cubelearn_root).resolve()))
    from network import RDAT_3DCNNLSTM, CplxModulus

    class PoseEncoder(RDAT_3DCNNLSTM):
        def __init__(self):
            super().__init__()
            self.n_frames = n_frames
            self.fc_3 = nn.Identity()

        def forward(self, x):
            assert x.ndim == 5 and tuple(x.shape[1:]) == (self.n_frames, 64, 8, 128)
            x = x[:, :, :, 0:8, :].contiguous()
            x = self.range_net(x)
            x = x.view(-1, 64, 8, 128)
            x = self.cplx_transpose(1, 3)(x)
            x = self.doppler_net(x)
            x = self.cplx_transpose(2, 3)(x)
            x = self.aoa_net(x)
            x = CplxModulus()(x)
            x = x.view(-1, 1, 128, 64, 64)
            for conv, bn in ((self.conv1, self.bn1),
                             (self.conv2, self.bn2),
                             (self.conv3, self.bn3)):
                x = self.maxpool(torch.nn.functional.relu(bn(conv(x))))
            x = x.view(-1, self.n_frames, 11760)
            output, _ = self.lstm(x)
            return self.fc_3(torch.nn.functional.relu(self.fc_2(output[:, -1, :])))

    return PoseEncoder()


class CubeLearnPose(nn.Module):
    def __init__(self, cubelearn_root, lpp_lr):
        super().__init__()
        self.hori = make_encoder(cubelearn_root)
        self.vert = make_encoder(cubelearn_root)
        self.seed_map = nn.Linear(256, 32 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(16, 14, 3, padding=1))
        for p in self.complex_parameters():
            p.requires_grad_(lpp_lr > 0)

    def complex_parameters(self):
        return [p for branch in (self.hori, self.vert)
                for name in ('range_net', 'doppler_net', 'aoa_net')
                for p in getattr(branch, name).parameters()]

    def parameter_groups(self, lr, lpp_lr):
        complex_params = self.complex_parameters()
        ids = {id(p) for p in complex_params}
        groups = [{'params': [p for p in self.parameters() if id(p) not in ids], 'lr': lr}]
        if lpp_lr > 0:
            groups.append({'params': complex_params, 'lr': lpp_lr})
        return groups

    def forward(self, hori, vert):
        x = torch.cat([self.hori(hori), self.vert(vert)], dim=1)
        x = self.seed_map(x).reshape(-1, 32, 8, 8)
        return torch.sigmoid(self.decoder(x))
