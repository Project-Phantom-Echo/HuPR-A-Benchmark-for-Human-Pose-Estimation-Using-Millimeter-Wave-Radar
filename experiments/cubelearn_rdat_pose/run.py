"""Train, benchmark or evaluate the paired R-D-A-T CubeLearn-on-HuPR experiment."""
import argparse,json,os,random,time,sys
from pathlib import Path
import common
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from data import RawPoseDataset
from model import CubeLearnPose

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

def atomic_save(state,path):
    temp=path.with_suffix('.partial');torch.save(state,temp);os.replace(temp,path)

def make_loader(ds,batch,workers,shuffle,seed):
    g=torch.Generator().manual_seed(seed)
    opts=dict(batch_size=batch,num_workers=workers,shuffle=shuffle,pin_memory=True,generator=g)
    if workers:opts.update(prefetch_factor=2,persistent_workers=True)
    return DataLoader(ds,**opts)

def evaluate(net,loader,gt_path,device):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from misc.metrics import get_max_preds
    predictions=[];total=0.;count=0
    net.eval()
    with torch.no_grad():
        for b in loader:
            h=net(b['hori'].to(device,non_blocking=True),b['vert'].to(device,non_blocking=True))
            total+=torch.nn.functional.binary_cross_entropy(h,b['target'].to(device)).item()*len(h);count+=len(h)
            xy,_=get_max_preds(h.cpu().numpy());xy*=4
            for image_id,points in zip(b['image_id'].tolist(),xy):
                predictions.append(dict(image_id=image_id,category_id=1,score=1.,keypoints=np.concatenate((points,np.ones((14,1))),axis=1).reshape(-1).tolist()))
    gt=COCO(str(gt_path));ev=COCOeval(gt,gt.loadRes(predictions),'keypoints');ev.params.useSegm=None
    ev.evaluate();ev.accumulate();ev.summarize()
    return dict(loss=total/count,AP=float(ev.stats[0]),AP50=float(ev.stats[1]),AP75=float(ev.stats[2]),samples=count),predictions

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['benchmark','train','test'],default='train')
    p.add_argument('--raw-root',type=Path,default=Path('/mnt/weka/fgeikyan/rf-datas/hupr'))
    p.add_argument('--gt-root',type=Path,default=common.ROOT/'local/data')
    p.add_argument('--cubelearn-root',type=Path,default=common.ROOT.parent/'cubelearn')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--lpp-lr',type=float,required=True)
    p.add_argument('--lr',type=float,default=.0003)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--epochs',type=int,default=2)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--benchmark-steps',type=int,default=30)
    a=p.parse_args()
    assert a.lpp_lr>=0 and a.lr>0 and a.epochs>0 and a.batch_size>0
    os.umask(0);torch.set_num_threads(1)
    common.check_evaluator()
    if not torch.cuda.is_available():raise RuntimeError('GPU allocation required')
    device=torch.device('cuda');seed_all(a.seed)
    cfg=yaml.safe_load((common.ROOT/'config/mscsa_prgcn.yaml').read_text())
    groups=[set(cfg['DATASET'][phase+'Name']) for phase in ['train','val','test']]
    assert not (groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
    hashes=common.source_hashes(a.cubelearn_root)
    if a.mode=='test' or a.resume:
        manifest=json.loads((a.output/'manifest.json').read_text())
        assert manifest['source_sha256']==hashes,'Source changed since this run'
        for key in ['lpp_lr','lr','batch_size','seed','workers','raw_root','gt_root','cubelearn_root']:
            assert manifest['arguments'][key]==str(getattr(a,key)),'Resume config differs: '+key
    else:
        a.output.mkdir(parents=True,exist_ok=False)
        manifest=dict(arguments={k:str(v) for k,v in vars(a).items()},source_sha256=hashes,
                      torch=torch.__version__,gpu=torch.cuda.get_device_name(),protocol='cubelearn-rdat-pose-v1')
        (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    net=CubeLearnPose(a.cubelearn_root,a.lpp_lr).to(device)
    opt=torch.optim.Adam(net.parameter_groups(a.lr,a.lpp_lr),weight_decay=0.)
    print('Parameters:',sum(p.numel() for p in net.parameters()),'trainable:',sum(p.numel() for p in net.parameters() if p.requires_grad),flush=True)
    if a.mode=='test':
        ckpt=torch.load(a.output/'best.pth',map_location=device);net.load_state_dict(ckpt['model'])
        loader=make_loader(RawPoseDataset(a.raw_root,'test',cfg),a.batch_size,a.workers,False,a.seed)
        metrics,preds=evaluate(net,loader,a.gt_root/'test_gt.json',device)
        (a.output/'test.json').write_text(json.dumps(dict(epoch=ckpt['epoch'],**metrics),indent=2)+'\n')
        (a.output/'test_predictions.json').write_text(json.dumps(preds)+'\n');return
    train=RawPoseDataset(a.raw_root,'train',cfg)
    if a.mode=='benchmark':
        loader=make_loader(train,a.batch_size,a.workers,True,a.seed)
        net.train();times=[];start=time.monotonic()
        for i,b in enumerate(loader):
            opt.zero_grad(set_to_none=True)
            pred=net(b['hori'].to(device,non_blocking=True),b['vert'].to(device,non_blocking=True))
            loss=torch.nn.functional.binary_cross_entropy(pred,b['target'].to(device))
            assert torch.isfinite(loss);loss.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
            opt.step();torch.cuda.synchronize();now=time.monotonic()
            if i>=5:times.append(now-start)
            start=now
            if i+1>=a.benchmark_steps:break
        train_memory=torch.cuda.max_memory_allocated()
        val_loader=make_loader(RawPoseDataset(a.raw_root,'val',cfg),a.batch_size,a.workers,False,a.seed)
        net.eval();val_times=[];start=time.monotonic()
        with torch.no_grad():
            for j,b in enumerate(val_loader):
                h=net(b['hori'].to(device,non_blocking=True),b['vert'].to(device,non_blocking=True))
                loss=torch.nn.functional.binary_cross_entropy(h,b['target'].to(device,non_blocking=True))
                assert torch.isfinite(loss)
                h.cpu();torch.cuda.synchronize();now=time.monotonic()
                if j>=5:val_times.append(now-start)
                start=now
                if j+1>=30:break
        epoch_seconds=float(np.mean(times)*len(loader)+np.mean(val_times)*len(val_loader)+15)
        result=dict(mean_step_seconds=float(np.mean(times)),median_step_seconds=float(np.median(times)),
                    p95_step_seconds=float(np.percentile(times,95)),samples_per_second=a.batch_size/np.mean(times),
                    train_epoch_seconds=float(np.mean(times)*len(loader)),validation_step_seconds=float(np.mean(val_times)),
                    epoch_with_validation_seconds=epoch_seconds,max_memory_bytes=train_memory,
                    steps=i+1,validation_steps=j+1,last_loss=float(loss.item()),
                    caveat='Short real-data benchmark, including optimizer steps; extrapolated validation includes 15 seconds for metrics/checkpoint overhead. No checkpoint retained; full-epoch timing may differ.')
        (a.output/'benchmark.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True);return
    start_epoch=0;best=(-1.,float('-inf'));history=[]
    if a.resume:
        ckpt=torch.load(a.output/'latest.pth',map_location='cpu')
        net.load_state_dict(ckpt['model']);opt.load_state_dict(ckpt['optimizer'])
        start_epoch=ckpt['epoch'];best=tuple(ckpt['best']);history=ckpt['history']
        torch.set_rng_state(ckpt['rng']);torch.cuda.set_rng_state_all(ckpt['cuda_rng'])
        random.setstate(ckpt['python_rng']);np.random.set_state(ckpt['numpy_rng'])
    val=make_loader(RawPoseDataset(a.raw_root,'val',cfg),a.batch_size,a.workers,False,a.seed)
    for epoch in range(start_epoch,a.epochs):
        loader=make_loader(train,a.batch_size,a.workers,True,a.seed+epoch)
        net.train();total=0.;count=0;t=time.monotonic()
        for i,b in enumerate(loader):
            opt.zero_grad(set_to_none=True)
            pred=net(b['hori'].to(device,non_blocking=True),b['vert'].to(device,non_blocking=True))
            loss=torch.nn.functional.binary_cross_entropy(pred,b['target'].to(device,non_blocking=True))
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward();opt.step();total+=loss.item()*len(pred);count+=len(pred)
            if i%500==0:print(f'epoch={epoch+1} batch={i}/{len(loader)} loss={loss.item():.6f}',flush=True)
        train_seconds=time.monotonic()-t;t=time.monotonic()
        metrics,_=evaluate(net,val,a.gt_root/'val_gt.json',device)
        rec=dict(epoch=epoch+1,train_loss=total/count,train_seconds=train_seconds,val_seconds=time.monotonic()-t,validation=metrics)
        history.append(rec);key=(metrics['AP'],-metrics['loss']);improved=key>best
        if improved:best=key
        ckpt=dict(epoch=epoch+1,model=net.state_dict(),optimizer=opt.state_dict(),best=best,history=history,
                  rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate(),numpy_rng=np.random.get_state())
        atomic_save(ckpt,a.output/'latest.pth')
        if improved:atomic_save(ckpt,a.output/'best.pth')
        (a.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
        print(json.dumps(rec),flush=True)
    (a.output/'complete.json').write_text(json.dumps(dict(epochs=a.epochs,best_val_AP=best[0]))+'\n')
if __name__=='__main__':main()
