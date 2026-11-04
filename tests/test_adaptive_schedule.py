import unittest
import numpy as np
from utils.adaptive_schedule import AdaptiveSchedule, survival_quotas, audit_quotas

class AdaptiveScheduleTests(unittest.TestCase):
    def cfg(self):
        return dict(initial_prune_ratio=.2,min_prune_ratio=.05,max_prune_ratio=.3,
                    min_recovery_steps=4,max_recovery_steps=10,convergence_window=2,convergence_threshold=.001,final_recovery_steps=12)
    def test_timing_bounds_and_convergence(self):
        s=AdaptiveSchedule(self.cfg())
        self.assertFalse(s.recovery_done([1,1,1])[0])
        self.assertTrue(s.recovery_done([1]*4)[0])
        self.assertTrue(s.recovery_done(list(range(10)))[0])
        self.assertFalse(s.recovery_done([1]*10,True)[0])
        self.assertTrue(s.recovery_done([1]*12,True)[0])
    def test_ratio_recovery_and_exact_target(self):
        s=AdaptiveSchedule(self.cfg())
        s.update(1,2,1.05); self.assertGreater(s.ratio,.2)
        prev=s.ratio; s.update(1,2,1.9); self.assertLess(s.ratio,prev)
        self.assertEqual(s.round_target(102,100),100)
    def test_joint_floors_unknown_boundary(self):
        d=dict(instance_ids=np.array([-1,-1,0,0,0]),instance_fragility=np.ones(5),boundary_mask=np.array([1,1,1,0,0],bool))
        c=dict(instance_protection=dict(enabled=True,base_keep_ratio=.1,fragility_beta=.2,max_keep_ratio=.5,min_instance_gaussians=2),boundary_protection=dict(enabled=True,boundary_keep_ratio=.5))
        a,b,rows=survival_quotas(d,c)
        self.assertEqual(a,{0:2});self.assertEqual(b,{-1:1,0:1});audit_quotas(d,a,b)
        with self.assertRaises(RuntimeError):audit_quotas({k:v[:2] for k,v in d.items()},a,b)
if __name__=='__main__':unittest.main()
