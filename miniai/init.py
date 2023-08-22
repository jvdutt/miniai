# cell starts
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from miniai.utils import Hook,to_cpu,show_images

from functools import partial
# cell ends

# cell starts
class ActivationStats():
    def __init__(self,ms):
        self.hooks = [Hook(m,self.append_stats) for m in ms]
    def append_stats(self,h,mod,inp,out):
        if not hasattr(h,'stats'):h.stats=[[],[],[]]
        out = to_cpu(out)
        h.stats[0].append(out.mean())
        h.stats[1].append(out.std())
        h.stats[2].append(out.abs().histc(40,0,10))
    def get_hist_image(self,h):
        return torch.stack(h.stats[2]).T.float().log1p()
    def get_dead_percentage(self,h):
        hist = torch.stack(h.stats[2])
        return hist[:,0]/hist.sum(1)
    def plot_histograms(self,imgs_per_row = 2,scale=scale,tfmx = lambda x:x.flip(0),tfmy = lambda y:str(y),**kwargs):
        hists = [self.get_hist_image(h) for h in self.hooks]
        labels = list(range(len(hists))
        show_images(hists,labels,len(hists),imgs_per_row = imgs_per_row,scale=scale,tfmx = tfmx,tfmy = tfmy,**kwargs)
    def plot_dead_chart(self,figsize=None):
        for h in self.hooks:
            fig,ax = plt.subplots(figsize=figsize)
            ax.plot(self.get_dead_percentage(h))
            ax.set_ylim(0,1)
            plt.show()
            
    def plot_stats(self,figsize=None):
        fig,axs = plt.subplots(1,2,figsize=figsize)
        for e,h in enumerate(self.hooks):
            for i in 0,1:
                axs[i].plot(h.stats[i],label=f'{e}')
        axs[0].set_title("Means")
        axs[1].set_title("Stds")
        plt.legend()
        plt.show()
# cell ends

# cell starts
class Normalizer(nn.Module):
    def __init__(self,*args,**kwargs):
        super().__init__()
        self.n = nn.BatchNorm2d(*args,**kwargs)
        for p in self.n.parameters():p.requires_grad_(False)

    def forward(self,x):
        return self.n(x)
# cell ends

# cell starts
class LsuvInit():
    def __init__(self,learner,modules,dl,take_first_batch=False,max_iter=10,tol=1e-5,orthogonal=True,device=None):
        if device is None:device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        learner.model.to(device)
        data = next(iter(dl)) if take_first_batch else iter(self.circle(dl))
        for m in modules:
            h = Hook(m,self.get_stats)
            self.orthogonal_init(m,orthogonal)
            for i in range(max_iter):
                learner.batch = data if take_first_batch else next(data)
                learner.batch = self.to_device(learner.batch,device)
                with torch.no_grad():
                    learner.predict()
                    print(m,h.mean,h.std)
                    if abs(h.std-1)<tol:
                        if hasattr(m,"bias") and (m.bias is not None):
                            if abs(h.mean)<tol:break
                        else:break
                    if hasattr(m,"weight") and (m.weight is not None):m.weight/=h.std
                    if hasattr(m,"bias") and (m.bias is not None):m.bias-=h.mean
            h.remove()
            
    def orthogonal_init(self,m,orth):
        if orth and hasattr(m,"weight") and (m.weight is not None):
            with torch.no_grad():m.weight.data = torch.from_numpy(self.svd_orthonormal(m.weight.numpy()))
            
    def svd_orthonormal(self,w):
        shape = w.shape
        if len(shape) < 2:
            raise RuntimeError("Only shapes of length 2 or more are supported.")
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)#w;
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        return q.astype(np.float32)
                
    def get_stats(self,h,mod,inp,out):
        out = to_cpu(out)
        h.mean = out.mean()
        h.std = out.std()
        
    def circle(self,dl):
        while True:
            for batch in dl:yield batch 
            
    def to_device(self,x,device):
        if isinstance(x, dict): return {k:self.to_device(v,device) for k,v in x.items()}
        if isinstance(x, list): return [self.to_device(o,device) for o in x]
        if isinstance(x, tuple): return tuple(self.to_device(list(x),device))
        return x.to(device)
# cell ends

# cell starts
def get_modules(model,module_names):
    return list(filter(partial(lambda x,y:isinstance(y,x),module_names),model.modules()))
# cell ends

