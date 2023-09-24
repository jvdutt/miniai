# cell starts
import random,math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from functools import partial
import sys,gc,traceback
# cell ends

# cell starts
class CancelBatchException(Exception):pass
class CancelEpochException(Exception):pass
class CancelFitException(Exception):pass
# cell ends

# cell starts
def set_seed(seed,deterministic=False):
    torch.use_deterministic_algorithms(deterministic)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
# cell ends

# cell starts
def to_cpu(x):
    if isinstance(x, dict): return {k:to_cpu(v) for k,v in x.items()}
    if isinstance(x, list): return [to_cpu(o) for o in x]
    if isinstance(x, tuple): return tuple(to_cpu(list(x)))
    return x.detach().cpu()
# cell ends

# cell starts
def _clean_ipython_hist():
    # Code in this function mainly copied from IPython source
    if not 'get_ipython' in globals(): return
    ip = get_ipython()
    user_ns = ip.user_ns
    ip.displayhook.flush()
    pc = ip.displayhook.prompt_count + 1
    for n in range(1, pc): user_ns.pop('_i'+repr(n),None)
    user_ns.update(dict(_i='',_ii='',_iii=''))
    hm = ip.history_manager
    hm.input_hist_parsed[:] = [''] * pc
    hm.input_hist_raw[:] = [''] * pc
    hm._i = hm._ii = hm._iii = hm._i00 =  ''
    

def _clean_tb():
    # h/t Piotr Czapla
    if hasattr(sys, 'last_traceback'):
        traceback.clear_frames(sys.last_traceback)
        delattr(sys, 'last_traceback')
    if hasattr(sys, 'last_type'): delattr(sys, 'last_type')
    if hasattr(sys, 'last_value'): delattr(sys, 'last_value')
    
    
def clean_mem():
    _clean_tb()
    _clean_ipython_hist()
    gc.collect()
    torch.cuda.empty_cache()
# cell ends

# cell starts
class Hook():
    def __init__(self,module,func,pre=False):
        if pre:self.hook = module.register_forward_pre_hook(partial(fun,self))
        else:self.hook = module.register_forward_hook(partial(func,self))
    def remove(self):self.hook.remove()
    def __del__(self):self.remove()
# cell ends

# cell starts
class SingleBatchCB():
    def after_predict(self):
        raise CancelFitException

def summary(learner,dl):
    learner.fit(dl,1,tmp_cbs=[SingleBatchCB()])
    mod_names,inp_shapes,out_shapes,num_params=[],[],[],[]
    flops=[]
    
    def _flops(x, h, w):
        if x.dim()<3: return x.numel()
        if x.dim()==4: return x.numel()*h*w

    def _shape(x):
        if isinstance(x, dict): return {k:_shape(v) for k,v in x.items()}
        if isinstance(x, list): return [_shape(o) for o in x]
        if isinstance(x, tuple): return tuple(_shape(list(x)))
        return x.shape       

    def hook_func(h,mod,inp,out):
        mod_names.append(type(mod).__name__)
        inp_shapes.append(_shape(inp))
        out_shapes.append(_shape(out))
        num_params.append(sum(p.numel() for p in mod.parameters()))
        *_,h,w = out.shape
        flops.append(sum(_flops(o, h, w) for o in mod.parameters())/1e6)
    hooks = [Hook(m,hook_func) for m in learner.model.children()]
    with torch.no_grad():learner.predict()
    for h in hooks:h.remove()
    tot_params = sum(num_params)
    print(f"Total number of parameters:{tot_params}")
    print(f"Total number of parameters:{sum(flops)}")
    d = {"Module":mod_names,"InputShape":inp_shapes,"OutputShape":out_shapes,"NumParams":num_params}
    d["PercentageParams"]=[p/tot_params for p in num_params]
    d["Mflops"]=flops
    return pd.DataFrame(d)
# cell ends

# cell starts
def show_images(imgs,labels=None,n=None,imgs_per_row = 4,scale=1,tfmx = lambda x:x,tfmy = lambda y:str(y),**kwargs):
    if labels is not None:assert(len(imgs)==len(labels))
    if n is None:
        n = len(imgs)
        idxs = range(n)
    else:
        idxs = random.sample(range(len(imgs)),n)
    num_rows,num_cols = math.ceil(n/imgs_per_row),imgs_per_row
    figsize=(num_cols*scale,num_rows*scale)
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False,figsize=figsize)
    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            k = row_idx*imgs_per_row+col_idx
            if k>=n:break
            ax = axs[row_idx, col_idx]
            img,label = imgs[idxs[k]],None if labels is None else labels[idxs[k]]
            ax.imshow(np.asarray(tfmx(img)), **kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[],title=tfmy(label))
            
    plt.tight_layout()
# cell ends

