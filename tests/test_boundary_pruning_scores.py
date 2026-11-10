"""Boundary weighting changes global rank, never instance/topology feasibility."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from train_pup_rir_compression import compression_score_policy,scoring_snapshot
from utils.adaptive_constraints import graph_from_edges,select_feasible_prune
from utils.adaptive_schedule import survival_quotas
from utils.pruning_scores import boundary_weighted_scores,fixed_budget_mask


class BoundaryPruningScoreTests(unittest.TestCase):
    def test_log_weight_increases_negative_and_positive_scores_without_mutation(self):
        raw=torch.tensor([-12.,-12.,8.,8.],dtype=torch.float64)
        original=raw.clone()
        weighted=boundary_weighted_scores(raw,[0,1,0,1],2.)
        torch.testing.assert_close(weighted,torch.tensor([-12.,-10.,8.,10.],dtype=torch.float64),rtol=0,atol=0)
        torch.testing.assert_close(raw,original,rtol=0,atol=0)
        torch.testing.assert_close(weighted.exp(),raw.exp()*torch.tensor([1.,np.exp(2),1.,np.exp(2)]),rtol=1e-14,atol=0)

    def test_singular_scores_remain_singular_and_ties_stay_deterministic(self):
        raw=torch.tensor([-float('inf'),-float('inf'),-3.,-3.,2.])
        weighted=boundary_weighted_scores(raw,[1,0,0,1,0],2.)
        self.assertEqual(weighted.tolist(),[-float('inf'),-float('inf'),-3.,-1.,2.])
        self.assertEqual(fixed_budget_mask(weighted,3).tolist(),[True,True,True,False,False])
        torch.testing.assert_close(boundary_weighted_scores(raw,[1]*5,0),raw,rtol=0,atol=0)

    def test_invalid_weights_alignment_or_nonfinite_input_fail_before_ranking(self):
        for weight in (-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):boundary_weighted_scores(torch.zeros(2),[0,1],weight)
        for mask in ([0],[[0,1]],[-.1,1],[0,1.1],[0,float('nan')]):
            with self.assertRaises(ValueError):boundary_weighted_scores(torch.zeros(2),mask,2)
        for raw in ([float('nan'),1],[float('inf'),1]):
            with self.assertRaises(FloatingPointError):boundary_weighted_scores(torch.tensor(raw),[0,1],2)

    def test_weight_is_soft_while_instance_and_topology_constraints_still_apply(self):
        weighted=boundary_weighted_scores(torch.tensor([-20.,-19.,5.,0.,1.]),[1,0,0,0,0],2.)
        topology=dict(graph=graph_from_edges([0,1,2],[[0,1],[1,2]]),
                      critical_mask=np.array([False,True,False,False,False]))
        mask,audit=select_feasible_prune(weighted.numpy(),[0]*5,{0:2},3,
            boundary_mask=[1,0,0,0,0],boundary_min_counts={},boundary_mode='none',topology=topology)
        # The boundary remains eligible; the bridge is a separate topology guard.
        np.testing.assert_array_equal(mask,[True,False,False,True,True])
        self.assertEqual(audit['boundary_protected'],0)
        self.assertEqual(audit['topology_protected'],1)
        self.assertTrue(audit['target_reached'])

    def test_policy_removes_boundary_floors_and_keeps_original_instance_floors(self):
        config=json.loads((Path(__file__).resolve().parents[1]/'configs/city_pup_rir_dpra_10x.json').read_text())
        meta=dict(instance_ids=np.array([0,0,0,1,1,-1]),instance_fragility=np.zeros(6),
                  boundary_mask=np.array([1,1,0,1,0,1],bool))
        original_config=copy.deepcopy(config)
        old_amin,old_bmin,old_rows=survival_quotas(meta,config)
        policy,amin,bmin,rows=compression_score_policy(config,meta)
        self.assertEqual((amin,bmin,rows),(old_amin,old_bmin,old_rows))
        self.assertEqual(policy['mode'],'pup')
        self.assertTrue(policy['preserve_boundary_merge'])
        config.update(pruning_score='pup_boundary_weighted',boundary_score={'log_weight':2.})
        policy,amin,bmin,rows=compression_score_policy(config,meta)
        self.assertEqual(amin,old_amin)
        self.assertEqual(bmin,{})
        self.assertTrue(all(row['boundary_minimum']==0 for row in rows))
        self.assertEqual(policy['boundary_constraint_mode'],'none')
        self.assertFalse(policy['preserve_boundary_merge'])
        self.assertEqual(config['boundary_protection'],original_config['boundary_protection'])

    def test_current_row_metadata_applies_to_fresh_fisher_after_merge(self):
        config={'pup_fisher_resolution':1}
        policy={'mode':'pup_boundary_weighted','boundary_weighted':True,'log_weight':2.}
        meta={'row_ids':np.array([3,8,10]),'boundary_mask':np.array([False,True,True])}
        # Row 10 represents a newly merged Gaussian; no stale parent rank is used.
        raw=torch.tensor([-7.,-4.,-9.])
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'scores.npz'
            with patch('train_pup_rir_compression.pup_pruning_scores',return_value=raw) as fisher:
                scores,receipt=scoring_snapshot([],object(),object(),None,meta,config,policy,output)
            fisher.assert_called_once()
            np.testing.assert_array_equal(scores,[-7.,-2.,-7.])
            with np.load(output) as z:
                np.testing.assert_array_equal(z['scores'],scores)
                np.testing.assert_array_equal(z['raw_pup_scores'],raw.numpy())
                np.testing.assert_array_equal(z['row_ids'],[3,8,10])
                np.testing.assert_array_equal(z['boundary_mask'],meta['boundary_mask'])
            self.assertEqual(receipt['boundary_count'],2)
            self.assertEqual(receipt['count'],3)


if __name__=='__main__':unittest.main()
