"""Render saved HuPR adaptation predictions against validation labels on CPU."""
from pathlib import Path
import sys,os,json,argparse
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'experiments/cubelearn_pose'))
import common
import torch,yaml,numpy as np
os.environ['MPLCONFIGDIR']='/tmp/hupr-mpl'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import CubeLearnPose
from data import RawPoseDataset
from misc.metrics import get_max_preds
from PIL import Image

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--images',action='store_true',help='Overlay on locally available validation camera frames')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();run=args.run;torch.set_num_threads(2)
    provenance=json.loads((run/'manifest.json').read_text())
    manifest=provenance['arguments']
    assert provenance['source_sha256']==common.source_hashes(Path(manifest['cubelearn_root'])), 'Source changed since this run'
    net=CubeLearnPose(manifest['cubelearn_root'],float(manifest['lpp_lr']))
    ckpt=torch.load(run/'best.pth',map_location='cpu');net.load_state_dict(ckpt['model']);net.eval()
    ds=RawPoseDataset(manifest['raw_root'],'val',yaml.safe_load((root/'config/mscsa_prgcn.yaml').read_text()))
    # Fixed validation examples selected before seeing predictions, across six recordings.
    indices=[100,700,1300,1900,2500,3100]
    edges=[(6,7),(6,11),(11,12),(12,13),(6,8),(8,9),(9,10),(6,3),(3,4),(4,5),(6,0),(0,1),(1,2)]
    fig,axes=plt.subplots(2,3,figsize=(10,7))
    for ax,idx in zip(axes.flat,indices):
        sample=ds[idx];seq,frame,label=ds.records[idx]
        with torch.no_grad():pred=net(sample['hori'][None],sample['vert'][None])
        xy,_=get_max_preds(pred.numpy());xy=xy[0]*4;gt=np.asarray(label['joints'])
        image=Path(manifest['raw_root'])/'frames'/f'single_{seq}'/'processed/images'/f'{frame:09d}.jpg'
        if args.images and image.exists():ax.imshow(Image.open(image).resize((256,256)))
        for pts,color in [(gt,'#00d787'),(xy,'#ff435d')]:
            for i,j in edges:ax.plot(pts[[i,j],0],pts[[i,j],1],color=color,linewidth=1.5)
            ax.scatter(pts[:,0],pts[:,1],c=color,s=8)
        ax.set(xlim=(0,256),ylim=(256,0),title=f'Validation sequence {seq}, frame {frame}');ax.axis('off')
    fig.suptitle(f'{run.name}: epoch {ckpt["epoch"]} — green: labels; red: predictions')
    fig.tight_layout();fig.savefig(args.output or run/'validation_poses.png',dpi=160)
if __name__=='__main__':main()
