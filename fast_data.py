"""Alternative loader; preserves release indexing and float64 normalization."""
import random
import numpy as np
import torch
from datasets.dataset import HuPR3D_horivert


def read_frame(path, start, count):
    # Map only the selected Doppler bins instead of reading all sixteen.
    a = np.load(path, mmap_mode='r')[start:start+count]
    # Same CHW layout and float64 operations as ToTensor + Normalize, batched
    # over Doppler bins and real/imaginary parts. Cast only at original assignment.
    components = []
    for component in (a.real, a.imag):
        x = torch.from_numpy(np.array(component.transpose(0,3,1,2), copy=True, order='C'))
        flat = x.reshape(-1, x.shape[-2]*x.shape[-1])
        lo = flat.min(1, keepdim=True).values
        zero = flat-lo
        norm = zero/zero.max(1, keepdim=True).values
        std, mean = torch.std_mean(norm, dim=1, keepdim=True)
        y = ((norm-mean)/std).reshape(x.shape).permute(0,2,3,1)
        components.append(y.float())
    return torch.stack(components, dim=1)


class FastDataset(HuPR3D_horivert):
    def __getitem__(self, index):
        if self.random:
            index *= random.randint(1, self.sampling_ratio)
        else:
            index *= self.sampling_ratio
        pad = index % self.duration
        idx = index-self.numGroupFrames//2-1
        shape = (self.numGroupFrames,self.numFrames,2,self.r,self.w,self.h)
        hori, vert = torch.empty(shape), torch.empty(shape)
        # Boundary padding repeats frames; reuse only within the current sample.
        frames = {}
        for j in range(self.numGroupFrames):
            if j+pad <= self.numGroupFrames//2:
                idx = index-pad
            elif j > self.duration-1-pad+self.numGroupFrames//2:
                idx = index+self.duration-1-pad
            else:
                idx += 1
            for paths, output in ((self.VRDAEPaths_hori,hori),(self.VRDAEPaths_vert,vert)):
                path = paths[idx]
                if path not in frames:
                    frames[path] = read_frame(path, self.numChirps//2-self.numFrames//2,self.numFrames)
                output[j] = frames[path]
        return dict(VRDAEmap_hori=hori,VRDAEmap_vert=vert,
                    jointsGroup=torch.LongTensor(self.annots[index]['joints']),
                    bbox=torch.FloatTensor(self.annots[index]['bbox']),
                    imageId=self.annots[index]['imageId'])

class CachedDataset(HuPR3D_horivert):
    """Read already normalized tensors; original __getitem__ selects the frames."""
    def __getitem__(self,index):
        if self.random:
            index *= random.randint(1,self.sampling_ratio)
        else:
            index *= self.sampling_ratio
        pad=index%self.duration
        idx=index-self.numGroupFrames//2-1
        indices=[]
        for j in range(self.numGroupFrames):
            if j+pad <= self.numGroupFrames//2: idx=index-pad
            elif j>self.duration-1-pad+self.numGroupFrames//2: idx=index+self.duration-1-pad
            else: idx+=1
            indices.append(idx)
        import os
        from pathlib import Path
        seq=Path(self.VRDAEPaths_hori[index]).parents[1].name
        cache=Path(os.environ['HUPR_NORMALIZED_CACHE'])/seq
        if not (cache/'complete.json').is_file(): raise RuntimeError(f'Incomplete cache: {cache}')
        frames=np.array(indices)%self.duration
        outputs=[]
        for view in ['hori','vert']:
            a=np.load(cache/f'{view}.npy',mmap_mode='r')
            outputs.append(torch.from_numpy(a[frames]))
        return dict(VRDAEmap_hori=outputs[0],VRDAEmap_vert=outputs[1],
                    jointsGroup=torch.LongTensor(self.annots[index]['joints']),
                    bbox=torch.FloatTensor(self.annots[index]['bbox']),
                    imageId=self.annots[index]['imageId'])
