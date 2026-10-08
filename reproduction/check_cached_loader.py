"""Compare cached and original inputs at sequence boundaries without training."""
import argparse, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch, yaml
from main import obj
from datasets.dataset import HuPR3D_horivert
from fast_data import CachedDataset

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    a=p.parse_args();os.environ['HUPR_NORMALIZED_CACHE']=str(a.cache.resolve())
    torch.set_num_threads(1)
    cfg=obj(yaml.safe_load((ROOT/'config/mscsa_prgcn.yaml').read_text()))
    def fixture(cls):
        d=cls.__new__(cls)
        for attr,field in [('duration','duration'),('numFrames','numFrames'),('numGroupFrames','numGroupFrames'),('numChirps','numChirps'),('r','rangeSize'),('w','azimuthSize'),('h','elevationSize')]:
            setattr(d,attr,getattr(cfg.DATASET,field))
        d.random=True;d.sampling_ratio=1;d.phase='train'
        d.transformFunc=d.getTransformFunc(cfg)
        d.VRDAEPaths_hori=[];d.VRDAEPaths_vert=[];d.annots=[]
        for seq in cfg.DATASET.trainName[:2]:
            for f in range(d.duration):
                for view in ('hori','vert'):
                    getattr(d,'VRDAEPaths_'+view).append(str(a.data/f'single_{seq}/{view}/{f:09d}.npy'))
                d.annots.append(dict(joints=[[0,0]]*14,bbox=[0,0,256,256],imageId=seq*100000+f))
        return d
    original,cached=fixture(HuPR3D_horivert),fixture(CachedDataset)
    for index in (0,100,599,600,1199):
        x,y=original[index],cached[index]
        for key in x:
            if isinstance(x[key],torch.Tensor):assert torch.equal(x[key],y[key]),(index,key)
            else:assert x[key]==y[key],(index,key)
        print('Exact match:',index,flush=True)
if __name__=='__main__':main()
