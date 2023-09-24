# cell starts
import torch
from torchmetrics import MeanMetric
from IPython.display import display
import matplotlib.pyplot as plt

from miniai.utils import to_cpu,CancelFitException,CancelBatchException,CancelEpochException
# cell ends

# cell starts
class with_clb():
    def __init__(self,name):
        self.name=name
    def __call__(self,f_):
        def f(o,*args,**kwargs):

            if True in args:o.train_phase = True
            if False in args:o.train_phase = False
            if 'train' in kwargs.keys():o.train_phase = True if kwargs['train'] else False
            
            try:
                o.callback(f'{self.name}_start')
                f_(o,*args,**kwargs)
                o.callback(f'{self.name}_end')
            except globals()[f'Cancel{self.name.title()}Exception']: pass
            finally:o.callback(f'{self.name}_cleanup')
        
        return f
# cell ends

# cell starts
class Learner():
    
    def __init__(self,model,loss_func,optim,cbs=[]):
        self.model,self.loss_func,self.optim,self.cbs = model,loss_func,optim,cbs
        for cb in self.cbs:cb.learner = self
        
    def predict(self):
        self.xb,self.yb = self.batch
        self.preds = self.model(self.xb)
    def get_loss(self):self.loss = self.loss_func(self.preds,self.yb)
    def backward(self):self.loss.backward()
    def step(self):self.optim.step()
    def zero_grad(self):self.optim.zero_grad()
        
        
    @with_clb('batch')
    def one_batch(self,train=True):
        self.model.training = train
        self.predict()
        self.callback('after_predict')
        self.get_loss()
        self.callback('after_loss')
        if self.model.training:
            self.backward()
            self.callback('after_backward')
            self.step()
            self.callback('after_step')
            self.zero_grad()
            
    @with_clb('epoch')
    def one_epoch(self,dl,train=True):
        
        for self.batch in dl:
            self.one_batch(train) 
     
    @with_clb('fit')
    def _fit(self):
        for self.epoch in range(1,self.n_epochs+1):
            self.one_epoch(self.train_dl,True)
            if self.valid_dl is not None:self.one_epoch(self.valid_dl,False)
    
    
    def fit(self,train_dl,n_epochs,valid_dl=None,tmp_cbs=[]):       
        self.n_epochs,self.train_dl,self.valid_dl = n_epochs,train_dl,valid_dl
        self.add_cbs(tmp_cbs)
        self._fit()
        self.remove_cbs(tmp_cbs)
        
    def callback(self,name):
        for cb in self.cbs:
            method = getattr(cb,name,None)
            if method is not None:method()
            
    def add_cbs(self,cbs):
        for cb in cbs:
            self.cbs.append(cb)
            cb.learner=self
        
    def remove_cbs(self,cbs):
        for cb in cbs:self.cbs.remove(cb)
            
# cell ends

class MPLearner(Learner):
    def __init__(self,model,loss_func,optim,cbs=[],autocast_enable=False,gradscale_enable=False):
        self.model,self.loss_func,self.optim,self.cbs = model,loss_func,optim,cbs
        for cb in self.cbs:cb.learner = self
        self.scaler = torch.cuda.amp.GradScaler()
        self.autocast_enable,self.gradscale_enable = autocast_enable,gradscale_enable
        
    def enable_amp(self):
        self.autocast_enable,self.gradscale_enable = True,True
    def disable_amp(self):
        self.autocast_enable,self.gradscale_enable = False,False
        
    def predict(self):
        if self.autocast_enable:
            self.autocast = torch.autocast(device_type="cuda",dtype=torch.float16)
            self.autocast.__enter__()
        self.xb,self.yb = self.batch
        self.preds = self.model(self.xb)
            
    def get_loss(self):
        self.loss = self.loss_func(self.preds,self.yb)
        if self.autocast_enable:
            self.autocast.__exit__(None,None,None)
    def backward(self):
        if self.gradscale_enable:
            self.scaler.scale(self.loss).backward()
        else:
            self.loss.backward()
    def step(self):
        if self.gradscale_enable:
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            self.optim.step()
        

# cell starts
class DeviceCB():
    def __init__(self,device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        print(f'is cuda available:{torch.cuda.is_available()}')
        print(f'device count:{torch.cuda.device_count()}')
        self.device = device
    def fit_start(self):self.learner.model.to(self.device)
    def batch_start(self):self.learner.batch = self.to_device(self.learner.batch)
    def to_device(self,x):
        if isinstance(x, dict): return {k:self.to_device(v) for k,v in x.items()}
        if isinstance(x, list): return [self.to_device(o) for o in x]
        if isinstance(x, tuple): return tuple(self.to_device(list(x)))
        return x.to(self.device)
       
# cell ends

# cell starts
class MetricsCB():
    def __init__(self,*ms,**metrics):
        for m in ms:metrics[type(m).__name__]=m
        self.metrics = metrics
        self.loss = MeanMetric()
    def fit_start(self):self.learner.metrics = self
    def epoch_start(self):
        for m in self.metrics.values():m.reset()
        self.loss.reset()
    def batch_end(self):
        preds,yb = to_cpu(self.learner.preds),to_cpu(self.learner.yb)
        for m in self.metrics.values():m.update(preds,yb)
        self.loss.update(to_cpu(self.learner.loss),len(yb))
    def _log(self,log):
        print(log)
    def epoch_end(self):
        log={'epoch':self.learner.epoch,'istrain':self.learner.train_phase}
        log['loss']=self.loss.compute()
        for k,v in self.metrics.items():log[k]= v.compute()
        self._log(log)
# cell ends

# cell starts
class PlotCB():

    def __init__(self,skip=25,trackmetrics=False,xlim=None,ylim=None,**kwargs):
        self.skip = skip
        self.trackmetrics=trackmetrics
        self.fig,self.ax = plt.subplots(**kwargs)
        if xlim is not None:self.ax.set_xlim(*xlim)
        if ylim is not None:self.ax.set_ylim(*ylim)
        plt.close()
        self.graph={}
        self._c = ['k','y','m','c','g','r','b']
        
    def display(self):self.fig_out = display(self.fig,display_id=True)
    
    def add_point(self,x,y,label='default'):
        if label not in self.graph.keys():self.graph[label]={'x':[x],'y':[y],'c':self._c.pop()}
        else:
            self.graph[label]['x'].append(x)
            self.graph[label]['y'].append(y)

    def plot(self,*args,**kwargs):
        self.ax.clear()
        for label,d in self.graph.items():
            ispoint = (len(d['x'])==1)
            self.ax.plot(d['x'],d['y'],f'{d["c"]}{"o--" if ispoint else ""}',label=label,*args,**kwargs)
        self.ax.legend()
        self.fig_out.update(self.fig)
        
    def fit_start(self):
        if not hasattr(self.learner,'metrics'):raise Exception("PlotCB shoudn't be before MetricsCB")
        self.metrics = self.learner.metrics
        if not hasattr(self,'batch_count'):self.batch_count = 0
        self.display()
        
    def batch_end(self):
        if self.learner.train_phase:
            self.batch_count = self.batch_count+1
            self.add_point(self.batch_count,to_cpu(self.learner.loss),label="train_loss")
            if self.batch_count%self.skip==1:
                self.plot()

    def epoch_end(self):
        if not self.learner.train_phase:
            self.add_point(self.batch_count,self.metrics.loss.compute(),label="valid_loss") 
            if self.trackmetrics:
                for name,m in self.metrics.metrics.items():
                    self.add_point(self.batch_count,m.compute(),label=f'valid_{name}')
            self.plot()
            
# cell ends

