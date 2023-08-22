# cell starts
import torch
import math
import matplotlib.pyplot as plt
from utils import to_cpu,CancelEpochException,CancelFitException
# cell ends

# cell starts
class SchedulerFunction:
    def __init__(self,sp,ep,L):
        self.sp,self.ep,self.L = sp,ep,L
    def __call__(self,i):
        pass
    def plot(self,x):
        plt.plot(x,[self(i) for i in x],"--ok")
        
class cos_sched(SchedulerFunction):
    def __call__(self,i):
        sp,ep,L = self.sp,self.ep,self.L
        m,c,x = sp-ep,(ep+sp)/2,math.cos(i*math.pi/L)/2
        return m*x + c
    
class exp_sched(SchedulerFunction):
    def __call__(self,i):
        sp,ep,L = self.sp,self.ep,self.L
        return sp*(ep/sp)**(i/L)
    
class lin_sched(SchedulerFunction):
    def __call__(self,i):
        sp,ep,L = self.sp,self.ep,self.L
        return sp + i*((ep-sp)/L) 
    
class concat_scheds(SchedulerFunction):
    
    def __init__(self,scheds,L=None,perc=None):
        if L is not None:
            if perc is None:perc=[1/len(scheds)]*len(scheds)
            assert len(perc)==len(scheds)
            tot = sum(perc) 
            Ls = [int(L*p/tot) for p in perc]
            for sched,l in zip(scheds,Ls):sched.L=l
        x = [0]
        for sched in scheds:x.append(x[-1]+sched.L)
        self.x = x[:-1]
        self.scheds = scheds
        
    def __call__(self,i):
        for e,point in enumerate(self.x[1:]):
            if i<=point:return self.scheds[e](i-self.x[e])
        return self.scheds[-1](i-self.x[-1])
    
class stack_scheds(SchedulerFunction):
    def __init__(self,scheds):
        self.scheds=scheds
    def __call__(self,i):
        return tuple(sched(i) for sched in self.scheds)
        
# cell ends

# cell starts
class Schedule:
    def __init__(self,optim,**kwargs):
        self.optim = optim
        self.optim_init_state_dict = self.optim.state_dict()
        self.count = 0
        n = len(self.optim.param_groups)
        
        self.recorder=[{} for i in range(n)]
        self.schedulers=[{} for i in range(n)]
        for k,v in kwargs.items():
            if k not in self.optim.param_groups[0]:raise AttributeError(f"No {k} in optimizers") 
            if type(v) is not list:kwargs[k]=[v for i in range(n)]
            assert n==len(kwargs[k])
            for recorder_pg in self.recorder:recorder_pg[k]=None
            for sched,sched_pg in zip(kwargs[k],self.schedulers):sched_pg[k]=sched

        
    def reset(self,optim=True,count=True,recorder=False):
        if optim:self.optim.load_state_dict(self.optim_init_state_dict)
        if count:self.count=0
        if recorder:
            for recorder_pg in self.recorder:
                for k in recorder_pg:recorder_pg[k]=None
    
    def step(self):
        for optim_pg,sched_pg,recorder_pg in zip(self.optim.param_groups,self.schedulers,self.recorder):
            for k in sched_pg:
                value = sched_pg[k](self.count)
                optim_pg[k] = value
                recorder_pg[k] = value
        self.count+=1
# cell ends

# cell starts
class BatchScheduleCB:
    def __init__(self,schedule):
        self.schedule = schedule
    
    def fit_start(self):
        self.schedule.step()
        
    def batch_end(self):
        if self.learner.train_phase:self.schedule.step()
        
class EpochScheduleCB:
    def __init__(self,schedule):
        self.schedule = schedule
    
    def fit_start(self):
        self.schedule.step()
        
    def epoch_end(self):
        if self.learner.train_phase:self.schedule.step()
# cell ends

# cell starts

class LRfinderCB:
    def __init__(self,sps=[1e-5],gammas=[1.1],max_mul=3,mom=0.9):
        self.sched_funcs = []
        for sp,gamma in zip(sps,gammas):self.sched_funcs.append(exp_sched(sp,sp*gamma,1))
        self.max_mul=max_mul
        self.buffer,self.mom,self.n=torch.tensor(0.0),mom,0
        
    def smooth_loss(self,loss):
        self.n+=1
        self.buffer.lerp_(loss,1-self.mom)
        return self.buffer/(1-self.mom**self.n)
        
        
    def fit_start(self):
        assert len(self.sched_funcs)==len(self.learner.optim.param_groups)
        self.sched = Schedule(self.learner.optim,lr=self.sched_funcs)
        self.exp_avg_losses=[]
        self.losses=[]
        self.min_loss = math.inf
        self.lr_records = [[] for i in range(len(self.learner.optim.param_groups))]
        self.sched.step()
        for e,r in enumerate(self.sched.recorder):self.lr_records[e].append(r["lr"])
        
     
    def batch_end(self):
        if not self.learner.train_phase:raise CancelEpochException()
        loss = to_cpu(self.learner.loss)
        self.losses.append(loss)
        loss = self.smooth_loss(loss)
        self.exp_avg_losses.append(loss)
        if loss<self.min_loss:self.min_loss=loss
        if math.isnan(loss) or loss>self.max_mul*self.min_loss:raise CancelFitException()
        self.sched.step()
        for e,r in enumerate(self.sched.recorder):self.lr_records[e].append(r["lr"])
        
    def fit_cleanup(self):
        for lr_record in self.lr_records:
            plt.plot(lr_record[:-1],self.exp_avg_losses[:-1])
            plt.xscale("log")
            plt.show()
# cell ends

# cell starts
def one_cycle_schedfuncs(L,beta2=None,max_lr=[1e-1],div_factor=[25],max_mom=[0.95],min_mom=[0.85],pct=[0.3],final_div_factor=None):
    if final_div_factor is None:final_div_factor=div_factor
    lr_sched_funcs,beta1_sched_funcs = [],[]
    for lr,df,hm,lm,p,f_df in zip(max_lr,div_factor,max_mom,min_mom,pct,final_div_factor):
        lr_sched_funcs.append(concat_scheds([cos_sched(lr/df,lr,1),cos_sched(lr,lr/f_df,1)],L,[p,1-p]))
        beta1_sched_funcs.append(concat_scheds([cos_sched(hm,lm,1),cos_sched(lm,hm,1)],L,[p,1-p]))
    if beta2 is not None:
        if not isinstance(beta2,float):raise Exception("beta2 is not float")
        beta2_sched_func = lin_sched(beta2,beta2,1)
        for i in range(len(beta1_sched_funcs)):beta1_sched_funcs[i] = stack_scheds([beta1_sched_funcs[i],beta2_sched_func])
    return lr_sched_funcs,beta1_sched_funcs
# cell ends

