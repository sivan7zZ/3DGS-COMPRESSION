import unittest
from types import SimpleNamespace
import contextlib
import io

import numpy as np

from utils.rir_dpra_schedule import RIRDPRA
from utils.schedule_utils import TrainingScheduler
from utils.adaptive_constraints import select_feasible_prune


class RIRDPRACompatibilityTests(unittest.TestCase):
    def test_schedule_matches_original_loop_for_recovered_and_forced_events(self):
        for target in (178809,89405):
            for recover in (True,False):
                with contextlib.redirect_stdout(io.StringIO()):
                    new=RIRDPRA({},target)
                    original=TrainingScheduler(SimpleNamespace(pruning_stop_iter=25200))
                    count=1788093;ref_count=count;stamp=1.;waiting=0;pruning_count=0;ratio=0.;events=[]
                    for it in range(15001,25200):
                        if it==15200:
                            original.calculate_pruning_target(ref_count,target);pruning_count=2
                            ratio=original.get_pruning_rate(it,ref_count,1);new.initialize(it,count)
                        elif it>15200 and (it-15200)%500==0 and 0<pruning_count<=6:
                            loss=.01 if recover else 2.
                            force=waiting+500>=1500;trigger=loss<.99*stamp or force
                            self.assertTrue(new.is_check(it));d=new.decide(it,count,loss)
                            self.assertEqual(d['trigger'],trigger);self.assertEqual(d['forced'],force)
                            if trigger:
                                ref_count=int(round(ref_count*(1-ratio)));stamp=loss;waiting=0
                                count=d['round_target'];self.assertEqual(count,ref_count)
                                ratio=original.get_pruning_rate(it,ref_count,pruning_count);pruning_count+=1
                                new.complete(it,count,loss);events.append((it,count))
                            else:waiting+=500
                    self.assertEqual(count,target);self.assertEqual(ref_count,target)
                    self.assertEqual(new.pruning_count,pruning_count)
                    self.assertEqual(len([x for x in events if x[1]>target]),4)

    def test_resume_preserves_pending_wait_and_dpra_state(self):
        with contextlib.redirect_stdout(io.StringIO()):
            first=RIRDPRA({},100);first.initialize(15200,1000)
            self.assertFalse(first.decide(15700,1000,2.)['trigger'])
            second=RIRDPRA({},100);second.load_state_dict(first.state_dict())
            for iteration in (16200,16700):
                a=first.decide(iteration,1000,2.);b=second.decide(iteration,1000,2.)
                self.assertEqual(a,b)
            self.assertTrue(a['forced'])

    def test_official_singular_pup_scores_keep_stable_constrained_budget(self):
        score=np.array([-np.inf,-np.inf,-np.inf,1.,2.])
        mask,audit=select_feasible_prune(score,[0,0,0,1,1],{0:1,1:1},3,allow_negative_infinity=True)
        self.assertEqual(mask.tolist(),[True,True,False,True,False])
        self.assertTrue(audit['target_reached'])
        with self.assertRaises(ValueError):select_feasible_prune(score,[0]*5,{},1)
        for bad in (np.nan,np.inf):
            with self.assertRaises(ValueError):select_feasible_prune([bad,0.],[0,0],{},1,allow_negative_infinity=True)


if __name__=='__main__':unittest.main()
