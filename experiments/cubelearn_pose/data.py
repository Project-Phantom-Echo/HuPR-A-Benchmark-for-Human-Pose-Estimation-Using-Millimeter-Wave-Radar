"""Raw ADC -> CubeLearn's 64-chirp, eight-antenna, 128-sample input.

No FFT, normalized HuPR cache, interpolation or learned preprocessing is applied
here. Decode the released LVDS layout and retain TX0/TX2 as in HuPR's azimuth ULA.
"""
from collections import OrderedDict
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import common
from misc.utils import generateTarget

FRAME_BYTES = 192*4*256*2*2

def decode_frame(packed):
    """One frame's int16 LVDS words -> (chirp=192,RX=4,ADC=256) complex64."""
    q=np.asarray(packed).reshape(192,4,128,4)
    real=q[...,:2].reshape(192,4,256).astype(np.float32)
    imag=q[...,2:].reshape(192,4,256).astype(np.float32)
    return real+1j*imag

def cube_frame(packed):
    adc=decode_frame(packed)
    # Released geometry: TX0 RX0..3 followed by TX2 RX0..3; TX1 is elevation-only.
    return np.concatenate((adc[0::3],adc[2::3]),axis=1)[...,:128].copy()

def frame_indices(frame):
    # Same eight-frame context and repeated-edge padding as HuPR's __getitem__.
    return np.clip(np.arange(frame-4,frame+4),0,599)

class RawPoseDataset(Dataset):
    def __init__(self,raw_root,phase,cfg):
        self.raw_root=Path(raw_root)
        self.phase=phase
        self.maps=OrderedDict()
        self.records=[]
        seqs=cfg['DATASET'][phase+'Name']
        annotations=json.loads((self.raw_root/'annotations'/f'hrnet_annot_{phase}.json').read_text())
        assert len(seqs)==len(annotations)
        for seq,labels in zip(seqs,annotations):
            assert len(labels)==600
            for f,label in enumerate(labels):
                assert int(Path(label['image']).stem)==f
                self.records.append((seq,f,label))
    def __len__(self):return len(self.records)
    def _mapped(self,seq,view):
        key=(seq,view)
        if key not in self.maps:
            path=self.raw_root/'radar'/f'single_{seq}'/view/'adc_data.bin'
            if path.stat().st_size!=600*FRAME_BYTES:raise ValueError(f'Unexpected ADC size: {path}')
            self.maps[key]=np.memmap(path,dtype='<i2',mode='r',shape=(600,FRAME_BYTES//2))
            if len(self.maps)>8:self.maps.popitem(last=False)
        self.maps.move_to_end(key)
        return self.maps[key]
    def __getitem__(self,index):
        seq,f,label=self.records[index]
        frames=frame_indices(f)
        views=[]
        for view in ('hori','vert'):
            mapped=self._mapped(seq,view)
            unique={int(i):cube_frame(mapped[i]) for i in np.unique(frames)}
            views.append(torch.from_numpy(np.stack([unique[int(i)] for i in frames])))
        # Released HuPR loader converts joints to integer coordinates before targets.
        joints=np.asarray(label['joints'],dtype=np.int64)
        target,_=generateTarget(joints,14,64,256)
        return {'hori':views[0],'vert':views[1],'target':torch.from_numpy(target),
                'image_id':seq*100000+f}
