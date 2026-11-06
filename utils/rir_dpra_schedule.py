"""Original PWSR RIR/DPRA event schedule; no relative adaptive recovery rounds."""
import math
from types import SimpleNamespace

from utils.schedule_utils import TrainingScheduler


class RIRDPRA:
    def __init__(self,config,target):
        self.config=dict(config)
        self.start=int(config.get('virtual_start',15000))
        self.end=int(config.get('virtual_end',25199))
        self.pruning_start=int(config.get('pruning_start_iter',15200))
        self.pruning_stop=int(config.get('pruning_stop_iter',25200))
        self.interval=int(config.get('pruning_interval',500))
        self.maximum=int(config.get('pruning_interval_maximum',1500))
        self.beta=float(config.get('beta',.99))
        if not self.start<self.pruning_start<self.end<self.pruning_stop:
            raise ValueError('Invalid RIR/DPRA iteration range')
        if self.interval<=0 or self.maximum<self.interval or not 0<self.beta<=1:
            raise ValueError('Invalid RIR/DPRA trigger settings')
        self.target=int(target)
        self.scheduler=TrainingScheduler(SimpleNamespace(pruning_stop_iter=self.pruning_stop))
        self.loss_stamp=1.;self.iteration_no_prune=0;self.pruning_count=0;self.ratio=0.

    def initialize(self,iteration,current):
        if iteration!=self.pruning_start or self.pruning_count:
            raise ValueError('RIR/DPRA initialization must occur at pruning_start_iter')
        if not 0<self.target<current:raise ValueError('Invalid final budget')
        self.scheduler.calculate_pruning_target(current,self.target)
        self.pruning_count=2
        self.ratio=self.scheduler.get_pruning_rate(iteration,current,1)

    def is_check(self,iteration):
        return (self.pruning_start<iteration<self.pruning_stop and
                (iteration-self.pruning_start)%self.interval==0 and 0<self.pruning_count<=6)

    def decide(self,iteration,current,loss):
        if not self.is_check(iteration):raise ValueError('Not a RIR check iteration')
        if not math.isfinite(loss):raise FloatingPointError('Nonfinite reconstruction loss')
        forced=self.iteration_no_prune+self.interval>=self.maximum
        recovered=loss<self.beta*self.loss_stamp
        result=dict(iteration=iteration,current_loss=float(loss),loss_stamp=self.loss_stamp,
                    threshold=self.beta*self.loss_stamp,forced=forced,recovered=recovered,
                    trigger=bool(recovered or forced),pruning_count=self.pruning_count,
                    pruning_ratio=self.ratio,waiting_iterations=self.iteration_no_prune,
                    round_target=max(self.target,int(round(current*(1.-self.ratio)))))
        if not result['trigger']:self.iteration_no_prune+=self.interval
        return result

    def complete(self,iteration,after,pre_prune_loss):
        if not self.is_check(iteration):raise ValueError('Unexpected pruning event')
        self.loss_stamp=float(pre_prune_loss);self.iteration_no_prune=0
        self.ratio=self.scheduler.get_pruning_rate(iteration,after,self.pruning_count)
        self.pruning_count+=1

    def state_dict(self):
        return dict(loss_stamp=self.loss_stamp,iteration_no_prune=self.iteration_no_prune,
                    pruning_count=self.pruning_count,ratio=self.ratio,scheduler=dict(self.scheduler.__dict__))

    def load_state_dict(self,state):
        for key in ('loss_stamp','iteration_no_prune','pruning_count','ratio'):setattr(self,key,state[key])
        self.scheduler.__dict__.update(state['scheduler'])
