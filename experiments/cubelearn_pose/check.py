"""CPU checks for decoding, temporal alignment, initialization and gradients."""
import common
import tempfile,sys
from pathlib import Path
import numpy as np
import torch,yaml
from data import decode_frame,cube_frame,frame_indices,RawPoseDataset,FRAME_BYTES
from model import CubeLearnPose

def main():
    torch.set_num_threads(2)
    raw=Path('/mnt/weka/fgeikyan/mmwave_data/hupr')
    path=raw/'radar/single_2/hori/adc_data.bin'
    packed=np.memmap(path,dtype='<i2',mode='r',shape=(600,FRAME_BYTES//2))
    sys.path.insert(0,str(common.ROOT/'preprocessing'))
    from process_iwr1843 import RadarObject
    with tempfile.TemporaryDirectory() as temp:
        np.asarray(packed[:2]).tofile(Path(temp)/'adc_data.bin')
        reference=RadarObject().getadcDataFromDCA1000(temp)
        for f in range(2):
            ref=reference[:,f*192:(f+1)*192,:].transpose(1,0,2)
            assert np.array_equal(ref,decode_frame(packed[f]))
            expected=np.concatenate((ref[0::3],ref[2::3]),axis=1)[...,:128]
            assert np.array_equal(expected,cube_frame(packed[f]))
    print('PASS exact ADC decoder and virtual-antenna mapping',flush=True)
    # Reproduce released frame selection for every frame, including both boundaries.
    for f in range(600):
        idx=f-5;expected=[]
        for j in range(8):
            if j+f<=4:idx=0
            elif j>599-f+4:idx=599
            else:idx+=1
            expected.append(idx)
        assert np.array_equal(expected,frame_indices(f)),f
    print('PASS all 600 temporal windows and target alignment',flush=True)
    torch.manual_seed(0);fixed=CubeLearnPose(common.ROOT.parent/'cubelearn',0)
    torch.manual_seed(0);learned=CubeLearnPose(common.ROOT.parent/'cubelearn',.001)
    assert all(torch.equal(v,learned.state_dict()[k]) for k,v in fixed.state_dict().items())
    x=torch.randn(1,8,64,8,128,dtype=torch.complex64)
    # Complex layers must reproduce their unnormalized FFTs at initialization.
    for branch in [fixed.hori,learned.vert]:
        r=branch.range_net(x); ref=torch.fft.fft(x,dim=-1)
        got=torch.complex(r.real,r.imag);assert torch.allclose(got,ref,atol=3e-5,rtol=3e-5)
        d=torch.randn(2,128,8,64,dtype=torch.complex64)
        got=branch.doppler_net(d);ref=torch.fft.fftshift(torch.fft.fft(d,dim=-1),dim=-1)
        assert torch.allclose(torch.complex(got.real,got.imag),ref,atol=3e-5,rtol=3e-5)
        a=torch.randn(2,128,64,8,dtype=torch.complex64)
        got=branch.aoa_net(a);ref=torch.fft.fftshift(torch.fft.fft(a,n=64,dim=-1),dim=-1)
        assert torch.allclose(torch.complex(got.real,got.imag),ref,atol=3e-5,rtol=3e-5)
    cfg=yaml.safe_load((common.ROOT/'config/mscsa_prgcn.yaml').read_text())
    ds=RawPoseDataset(raw,'train',cfg);sample=ds[100]
    args=[sample[k].unsqueeze(0) for k in ('hori','vert')]
    fixed.eval();learned.eval()
    with torch.no_grad():y1=fixed(*args);y2=learned(*args)
    assert y1.shape==(1,14,64,64) and torch.equal(y1,y2)
    learned.train()
    loss=torch.nn.functional.binary_cross_entropy(learned(*args),sample['target'].unsqueeze(0));loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in learned.complex_parameters())
    assert not any(p.requires_grad for p in fixed.complex_parameters())
    assert all(p.grad.abs().sum()>0 for p in learned.complex_parameters())
    print('PASS exact paired predictions, Fourier initialization, finite nonzero complex-layer gradients',flush=True)
    print('PASS real example: input',tuple(sample['hori'].shape),'output',tuple(y1.shape),'loss',loss.item(),flush=True)
if __name__=='__main__':main()
