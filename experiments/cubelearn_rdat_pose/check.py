"""Verify released-encoder equivalence, paired initialization and real-data gradients."""
import gc
import sys
import common
import torch
import yaml
from data import RawPoseDataset
from model import CubeLearnPose, make_encoder


def main():
    torch.set_num_threads(2)
    assert torch.cuda.is_available(), 'Run checks inside a GPU allocation'
    device = torch.device('cuda')
    cube = common.ROOT.parent / 'cubelearn'
    sys.path.insert(0, str(cube))
    from network import RDAT_3DCNNLSTM
    torch.manual_seed(0)
    original = RDAT_3DCNNLSTM().to(device).eval()
    original.fc_3 = torch.nn.Identity()
    adapted = make_encoder(cube, n_frames=10).to(device).eval()
    adapted.load_state_dict(original.state_dict())
    x = torch.randn(1, 10, 64, 8, 128, dtype=torch.complex64, device=device)
    with torch.no_grad():
        y0, y1 = original(x), adapted(x)
    assert torch.equal(y0, y1), (y0 - y1).abs().max().item()
    print('PASS exact released R-D-A-T feature output at original ten-frame length', flush=True)
    del original, adapted, x, y0, y1
    gc.collect(); torch.cuda.empty_cache()
    torch.manual_seed(0)
    fixed = CubeLearnPose(cube, 0).to(device).eval()
    torch.manual_seed(0)
    learned = CubeLearnPose(cube, .001).to(device).eval()
    assert all(torch.equal(v, learned.state_dict()[k]) for k, v in fixed.state_dict().items())
    cfg = yaml.safe_load((common.ROOT / 'config/mscsa_prgcn.yaml').read_text())
    sample = RawPoseDataset('/mnt/weka/fgeikyan/rf-datas/hupr', 'train', cfg)[100]
    args = [sample[k].unsqueeze(0).to(device) for k in ('hori', 'vert')]
    with torch.no_grad():
        y0, y1 = fixed(*args), learned(*args)
    assert y0.shape == (1, 14, 64, 64) and torch.equal(y0, y1)
    assert all(not p.requires_grad for p in fixed.complex_parameters())
    del fixed, y0, y1
    gc.collect(); torch.cuda.empty_cache()
    branch = learned.hori
    for layer, shape, n, shifted in [
        (branch.range_net, (2, 128), 128, False),
        (branch.doppler_net, (2, 64), 64, True),
        (branch.aoa_net, (2, 8), 64, True),
    ]:
        z = torch.randn(*shape, dtype=torch.complex64, device=device)
        with torch.no_grad():
            actual = layer(z)
        ref = torch.fft.fft(z, n=n, dim=-1)
        if shifted:
            ref = torch.fft.fftshift(ref, dim=-1)
        assert torch.allclose(torch.complex(actual.real, actual.imag), ref, atol=3e-5, rtol=3e-5)
    learned.train()
    loss = torch.nn.functional.binary_cross_entropy(learned(*args), sample['target'].unsqueeze(0).to(device))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
               for p in learned.complex_parameters())
    opt = torch.optim.Adam(learned.parameter_groups(.0003, .001))
    opt.step()
    assert all(torch.isfinite(p).all() for p in learned.parameters())
    print('PASS matched fixed/learned predictions, DFT initialization, real-data gradients and optimizer step', flush=True)
    print('Parameters:', sum(p.numel() for p in learned.parameters()), 'loss:', loss.item(), flush=True)


if __name__ == '__main__':
    main()
