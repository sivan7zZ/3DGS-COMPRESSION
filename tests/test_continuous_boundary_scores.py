"""Continuous score numerics, source identity and real native mutation replay."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_adaptive_mutation import make_model,metadata as base_metadata
from train_pup_rir_compression import (
    compression_score_policy,load_continuous_boundary_metadata,scoring_snapshot,
)
from utils.adaptive_mutation import CompressionState,moment_match_pairs
from utils.distill_utils import sha256_file
from utils.pruning_scores import CONTINUOUS_BOUNDARY_FIELDS,continuous_boundary_scores


SETTINGS=dict(sam_weight=.4,depth_weight=.4,rgb_edge_weight=.2,gain=5.,fusion='log1p')


def metadata(n=6):
    result=base_metadata(n)
    result['instance_fragility']=np.zeros(n,np.float32)
    result['visibility']=np.arange(1,n+1,dtype=np.float32)
    for offset,key in enumerate(CONTINUOUS_BOUNDARY_FIELDS):
        result[key]=np.linspace(.03*(offset+1),.9-.07*offset,n,dtype=np.float32)
    return result


class ContinuousBoundaryScoreTests(unittest.TestCase):
    def test_exact_modal_contributions_and_negative_logdet_direction(self):
        raw=torch.full((4,),-20.,dtype=torch.float64)
        sam=torch.tensor([1.,0.,0.,1.],dtype=torch.float64)
        depth=torch.tensor([0.,1.,0.,1.],dtype=torch.float64)
        rgb=torch.tensor([0.,0.,1.,1.],dtype=torch.float64)
        result=continuous_boundary_scores(raw,sam,depth,rgb,SETTINGS)
        expected=torch.tensor([.4,.4,.2,1.],dtype=torch.float64)
        torch.testing.assert_close(result['combined_boundary_strength'],expected,rtol=0,atol=0)
        torch.testing.assert_close(result['boundary_multiplier'],1+5*expected,rtol=0,atol=0)
        torch.testing.assert_close(result['scores'],raw+torch.log1p(5*expected),rtol=0,atol=0)
        self.assertTrue((result['scores']>raw).all())
        self.assertTrue(torch.equal(raw,torch.full_like(raw,-20.)))

    def test_no_threshold_preserves_tiny_and_intermediate_continuous_values(self):
        strengths=torch.tensor([0.,1e-7,.2,.55,.55001,.9,1.],dtype=torch.float64)
        result=continuous_boundary_scores(torch.zeros(7,dtype=torch.float64),strengths,strengths,strengths,SETTINGS)
        torch.testing.assert_close(result['combined_boundary_strength'],strengths,rtol=1e-15,atol=0)
        self.assertTrue((result['scores'][1:]>result['scores'][:-1]).all())
        self.assertLess(float(result['scores'][1]),1e-6)
        self.assertAlmostEqual(float(result['scores'][-1]),np.log(6.),places=14)

    def test_singular_pup_and_zero_gain_do_not_invent_importance(self):
        raw=torch.tensor([-float('inf'),-3.,0.,8.])
        values=torch.ones(4)
        result=continuous_boundary_scores(raw,values,values,values,SETTINGS)
        self.assertTrue(torch.isneginf(result['scores'][0]))
        self.assertTrue(torch.isfinite(result['scores'][1:]).all())
        unchanged=continuous_boundary_scores(raw,values,values,values,dict(SETTINGS,gain=0))
        torch.testing.assert_close(unchanged['scores'],raw,rtol=0,atol=0)
        torch.testing.assert_close(unchanged['boundary_multiplier'],torch.ones(4),rtol=0,atol=0)

    def test_bad_component_dtypes_ranges_alignment_and_settings_fail(self):
        raw=torch.zeros(3)
        good=torch.zeros(3)
        for bad in (torch.zeros(3,dtype=torch.bool),torch.zeros(3,dtype=torch.int64),torch.zeros(2),
                    torch.tensor([0.,1.1,0.]),torch.tensor([0.,-1e-6,0.]),torch.tensor([0.,float('nan'),0.])):
            with self.subTest(component=str(bad)):
                with self.assertRaises(ValueError):continuous_boundary_scores(raw,bad,good,good,SETTINGS)
        for changed in (dict(SETTINGS,fusion='threshold'),dict(SETTINGS,gain=-1),dict(SETTINGS,gain=float('inf')),
                        dict(SETTINGS,sam_weight=-.1),dict(SETTINGS,rgb_edge_weight=.3)):
            with self.subTest(settings=changed):
                with self.assertRaises(ValueError):continuous_boundary_scores(raw,good,good,good,changed)
        for value in (float('nan'),float('inf')):
            with self.assertRaises(FloatingPointError):
                continuous_boundary_scores(torch.tensor([0.,value,1.]),good,good,good,SETTINGS)

    def test_policy_keeps_instance_floors_without_binary_boundary_quota(self):
        config=json.loads((Path(__file__).resolve().parents[1]/'configs/city_pup_rir_dpra_10x.json').read_text())
        original=copy.deepcopy(config)
        meta=metadata()
        _,amin,_,_=compression_score_policy(config,meta)
        config.update(pruning_score='pup_continuous_boundary',boundary_score=SETTINGS)
        del config['boundary_protection']['boundary_keep_ratio']
        policy,new_amin,bmin,rows=compression_score_policy(config,meta)
        self.assertEqual(new_amin,amin)
        self.assertEqual(bmin,{})
        self.assertEqual(policy['boundary_constraint_mode'],'none')
        self.assertFalse(policy['preserve_boundary_merge'])
        self.assertTrue(policy['continuous_boundary'])
        self.assertEqual(policy['boundary_score'],SETTINGS)
        self.assertTrue(all(row['boundary_minimum']==0 for row in rows))
        self.assertEqual(config['instance_protection'],original['instance_protection'])

    def test_snapshot_records_all_components_and_ignores_binary_mask(self):
        meta=metadata(4)
        meta.update(row_ids=np.array([1,4,9,17]),original_ids=np.array([1,4,9,-1]))
        policy=dict(mode='pup_continuous_boundary',boundary_weighted=True,continuous_boundary=True,
                    log_weight=0.,boundary_score=SETTINGS)
        raw=torch.tensor([-float('inf'),-8.,0.,7.])
        expected=continuous_boundary_scores(raw,*(meta[key] for key in CONTINUOUS_BOUNDARY_FIELDS),config=SETTINGS)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'scores.npz'
            with patch('train_pup_rir_compression.pup_pruning_scores',return_value=raw):
                first,receipt=scoring_snapshot([],None,None,None,meta,{'pup_fisher_resolution':1},policy,path)
                meta['boundary_mask']=~meta['boundary_mask']
                second,_=scoring_snapshot([],None,None,None,meta,{'pup_fisher_resolution':1},policy,Path(tmp)/'second.npz')
            np.testing.assert_array_equal(first,second)
            with np.load(path) as archive:
                for key in CONTINUOUS_BOUNDARY_FIELDS+('row_ids','original_ids'):
                    np.testing.assert_array_equal(archive[key],meta[key])
                for key,value in expected.items():np.testing.assert_array_equal(archive[key],value.numpy())
                np.testing.assert_array_equal(archive['raw_pup_scores'],raw.numpy())
            self.assertTrue(receipt['continuous_boundary'])
            self.assertEqual(receipt['boundary_score'],SETTINGS)
            self.assertNotIn('boundary_log_weight',receipt)


class ContinuousSourceIdentityTests(unittest.TestCase):
    def fixture(self,folder):
        shared=folder/'shared'
        shared.mkdir()
        source=metadata(4)
        np.savez_compressed(str(shared/'adaptive_metadata.npz'),**source)
        shared_manifest=dict(source_count=4,source_ply_sha256='a'*64,source_xyz_sha256='b'*64,
                             train_frames=['frame_001'],test_frames_excluded=['frame_000'])
        sidecar=folder/'continuous_metadata.npz'
        arrays={key:source[key].copy() for key in CONTINUOUS_BOUNDARY_FIELDS+('original_ids','visibility')}
        arrays.update(source_ply_sha256=np.array('a'*64),source_xyz_sha256=np.array('b'*64))
        np.savez_compressed(str(sidecar),**arrays)
        manifest=dict(shared_manifest,status='complete',shared_metadata_sha256=sha256_file(shared/'adaptive_metadata.npz'),
                      sidecar_sha256=sha256_file(sidecar))
        (folder/'metadata_manifest.json').write_text(json.dumps(manifest))
        config=dict(shared_metadata=str(shared),continuous_boundary_metadata=str(sidecar))
        return config,source,shared_manifest,arrays,manifest

    def test_load_adds_only_exactly_aligned_continuous_float_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            config,source,shared_manifest,_,_=self.fixture(Path(tmp))
            fields={key:source[key].copy() for key in source if key not in CONTINUOUS_BOUNDARY_FIELDS}
            original={key:value.copy() for key,value in fields.items()}
            receipt=load_continuous_boundary_metadata(config,fields,shared_manifest)
            for key in CONTINUOUS_BOUNDARY_FIELDS:np.testing.assert_array_equal(fields[key],source[key])
            for key in original:np.testing.assert_array_equal(fields[key],original[key])
            self.assertTrue(receipt['row_identity_exact'])
            self.assertTrue(receipt['visibility_exact'])
            self.assertEqual(receipt['sha256'],sha256_file(config['continuous_boundary_metadata']))

    def test_rejects_swapped_rows_visibility_foreign_source_or_discrete_fields(self):
        for key,bad in (('original_ids',np.array([1,0,2,3],np.int64)),
                        ('visibility',np.zeros(4,np.float32)),
                        ('source_xyz_sha256',np.array('c'*64)),
                        ('sam_boundary_strength',np.array([0,1,0,1],bool)),
                        ('depth_boundary_strength',np.array([0.,1.01,0.,1.],np.float32))):
            with self.subTest(key=key),tempfile.TemporaryDirectory() as tmp:
                folder=Path(tmp)
                config,source,shared_manifest,arrays,manifest=self.fixture(folder)
                arrays[key]=bad
                np.savez_compressed(config['continuous_boundary_metadata'],**arrays)
                manifest['sidecar_sha256']=sha256_file(config['continuous_boundary_metadata'])
                (folder/'metadata_manifest.json').write_text(json.dumps(manifest))
                fields={key:source[key].copy() for key in source if key not in CONTINUOUS_BOUNDARY_FIELDS}
                with self.assertRaises(ValueError):load_continuous_boundary_metadata(config,fields,shared_manifest)
                self.assertFalse(any(key in fields for key in CONTINUOUS_BOUNDARY_FIELDS))

    def test_rejects_changed_sidecar_hash_train_split_or_shared_hash(self):
        for key,bad in (('sidecar_sha256','0'*64),('shared_metadata_sha256','0'*64),
                        ('train_frames',['frame_000']),('test_frames_excluded',[]),('source_count',3)):
            with self.subTest(key=key),tempfile.TemporaryDirectory() as tmp:
                folder=Path(tmp)
                config,source,shared_manifest,_,manifest=self.fixture(folder)
                manifest[key]=bad
                (folder/'metadata_manifest.json').write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):load_continuous_boundary_metadata(config,source,shared_manifest)


class ContinuousNativeMutationTests(unittest.TestCase):
    def test_prune_merge_pool_checkpoint_restore_replays_all_three_channels(self):
        model=make_model(6)
        original=metadata(6)
        original['boundary_mask'][0]=True
        state=CompressionState(model,original)
        state.remove(np.isin(state.metadata['original_ids'],[2,5]),1)
        _,weights=moment_match_pairs(state._params(),[[0,1]],state.metadata['visibility'])
        w=weights[0].astype(np.float64)
        expected={key:np.float32((w[0]*original[key][0]+w[1]*original[key][1])/w.sum())
                  for key in CONTINUOUS_BOUNDARY_FIELDS}
        self.assertEqual(state.merge([[0,1]],1,preserve_boundary=False),1)
        np.testing.assert_array_equal(state.metadata['original_ids'],[3,4,-1])
        self.assertTrue(state.metadata['boundary_mask'][-1])
        self.assertEqual(state.lineage[0]['merge_weights'],weights[0].tolist())
        for key in CONTINUOUS_BOUNDARY_FIELDS:
            self.assertEqual(state.metadata[key][-1],expected[key])
            self.assertNotEqual(state.metadata[key][-1],max(original[key][0],original[key][1]))
        with tempfile.TemporaryDirectory() as tmp:
            save=Path(tmp)/'state.pt'
            state.save(str(save),{'continuous_boundary_metadata':{'sha256':'c'*64}})
            loaded,extra=CompressionState.load(model,str(save))
            self.assertEqual(extra['continuous_boundary_metadata']['sha256'],'c'*64)
            self.assertEqual(loaded.lineage,state.lineage)
            for key in state.metadata:np.testing.assert_array_equal(loaded.metadata[key],state.metadata[key])
            self.assertEqual(loaded.restore([0],1,4),1)
            self.assertEqual(loaded.metadata['original_ids'][-1],2)
            for key in CONTINUOUS_BOUNDARY_FIELDS:
                self.assertEqual(loaded.metadata[key][-1],original[key][2])
                self.assertEqual(loaded.metadata[key][-2],expected[key])
            self.assertEqual(loaded.audit()['count'],4)

    def test_saturated_components_stay_bounded_when_float_weights_sum_above_one(self):
        model=make_model(4,warm_optimizer=False)
        meta=metadata(4)
        # Select an actual opacity*visibility ratio whose float32 weights sum
        # above one; both parents carry saturated continuous boundary evidence.
        ratios=np.logspace(-8,8,4000,dtype=np.float32)
        alpha=torch.sigmoid(model._opacity[0,0]).detach()
        raw=torch.stack([alpha*torch.from_numpy(ratios),alpha*torch.ones(len(ratios))],1)
        weights=(raw/raw.sum(1,keepdim=True)).numpy()
        ratio=ratios[np.flatnonzero(weights.sum(1)>1)[0]]
        meta['visibility'][:2]=[ratio,1]
        for key in CONTINUOUS_BOUNDARY_FIELDS:meta[key][:2]=1.
        state=CompressionState(model,meta)
        state.merge([[0,1]],1,preserve_boundary=False)
        self.assertGreater(sum(state.lineage[0]['merge_weights']),1.)
        for key in CONTINUOUS_BOUNDARY_FIELDS:self.assertEqual(state.metadata[key][-1],1.)
        self.assertEqual(state.audit()['count'],3)

    def test_reject_partial_nonfloat_and_nonfinite_continuous_metadata(self):
        for kind in ('partial','boolean','nonfinite','out_of_range'):
            meta=metadata(4)
            if kind=='partial':del meta[CONTINUOUS_BOUNDARY_FIELDS[-1]]
            elif kind=='boolean':meta[CONTINUOUS_BOUNDARY_FIELDS[0]]=np.zeros(4,bool)
            elif kind=='nonfinite':meta[CONTINUOUS_BOUNDARY_FIELDS[1]][0]=np.nan
            else:meta[CONTINUOUS_BOUNDARY_FIELDS[2]][0]=1.1
            with self.subTest(kind=kind):
                with self.assertRaises(ValueError):CompressionState(make_model(4),meta)


if __name__=='__main__':unittest.main()
