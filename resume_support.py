"""Epoch-boundary recovery; leave model, optimizer and selection rules unchanged."""
import os,random
import numpy as np
import torch


def install(base):
    original_save=base.saveModelWeight
    original_load=base.loadModelWeight
    def save(self,epoch,acc):
        original_save(self,epoch,acc)
        state=dict(epoch=epoch,model_state_dict=self.model.state_dict(),
                   optimizer_state_dict=self.optimizer.state_dict(),best_ap=self.logger.showBestAP(),
                   torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                   numpy_rng=np.random.get_state(),python_rng=random.getstate())
        path=os.path.join(self.dir,'resume.pth')
        torch.save(state,path+'.partial');os.replace(path+'.partial',path)
    def load(self,mode):
        path=os.path.join(self.dir,'resume.pth')
        if mode!='checkpoint' or self.args.eval or not os.path.isfile(path):
            return original_load(self,mode)
        state=torch.load(path,map_location='cpu')
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.start_epoch=state['epoch']+1
        self.logger.bestAP=state['best_ap']
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
        print(f'Resumed after epoch {self.start_epoch}; model/optimizer/best AP/RNG restored',flush=True)
    base.saveModelWeight=save
    base.loadModelWeight=load
