"""Normalize each released frame once; atomically publish verified float32 arrays."""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fast_data import read_frame
from datasets.base import Normalize
from torchvision.transforms import ToTensor


def build(seq, data_root, cache_root):
    os.umask(0)
    torch.set_num_threads(1)
    dest=cache_root/f'single_{seq}'
    dest.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    checked=[]
    for view in ['hori','vert']:
        temp=dest/f'{view}.partial.npy'
        target=np.lib.format.open_memmap(temp, mode='w+',dtype=np.float32,shape=(600,8,2,64,64,8))
        for frame in range(600):
            source=data_root/f'single_{seq}/{view}/{frame:09d}.npy'
            value=read_frame(str(source),4,8)
            if not torch.isfinite(value).all(): raise RuntimeError(f'Nonfinite input: {source}')
            if frame in (0,300,599):
                raw=np.load(source,mmap_mode='r')
                ref=torch.empty_like(value)
                for d in range(8):
                    for c,part in enumerate([raw[d+4].real,raw[d+4].imag]):
                        ref[d,c]=Normalize()(ToTensor()(part)).permute(1,2,0)
                if not torch.equal(value,ref): raise RuntimeError(f'Normalization mismatch: {source}')
                checked.append(f'{view}/{frame}')
            target[frame]=value.numpy()
            if frame%100==0: print(seq,view,frame,flush=True)
        target.flush(); del target
        os.rename(temp,dest/f'{view}.npy')
    receipt=dict(sequence=seq,frames=600,dtype='float32',checked_exact=checked,seconds=time.monotonic()-start,
                 builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (dest/'complete.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sequence',type=int,required=True)
    p.add_argument('--data',type=Path,required=True,help='Released preprocessed radar maps')
    p.add_argument('--cache',type=Path,required=True)
    a=p.parse_args();build(a.sequence,a.data.resolve(),a.cache.resolve())
