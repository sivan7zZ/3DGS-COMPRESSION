import tempfile
import unittest
from pathlib import Path
import numpy as np
from utils.adaptive_metadata import (load_kernel,accumulate_support,weighted_jaccard_pairs,aggregate_assignments,structure_risk_map)
from utils.structure_visibility import prepare_gaussians,render_visibility


class AdaptiveMetadataTest(unittest.TestCase):
    def test_weighted_jaccard_uses_mass_not_local_id(self):
        a=dict(gaussian_ids=np.array([0,1],np.int32),labels=np.array([1,1],np.int32),mass=np.array([2.,1.],np.float32))
        b=dict(gaussian_ids=np.array([0,1,2],np.int32),labels=np.array([7,7,1],np.int32),mass=np.array([1.,3.,100.],np.float32))
        x,y,s=weighted_jaccard_pairs(a,b)
        self.assertEqual(list(zip(x,y)),[(1,7)])
        self.assertAlmostEqual(float(s[0]),2./5.)
        with tempfile.TemporaryDirectory() as d:
            xx,yy,ss=weighted_jaccard_pairs(a,b,load_kernel(d))
        np.testing.assert_array_equal(xx,x);np.testing.assert_array_equal(yy,y);np.testing.assert_allclose(ss,s)

    def test_sparse_raster_accumulates_all_contributors_and_cell_mass(self):
        gaussian=prepare_gaussians(np.array([[0,0,4],[0,0,5]],np.float32),np.array([[.1,.1,.1],[.1,.1,.1]],np.float32),np.array([[1,0,0,0],[1,0,0,0]],np.float32),np.array([.8,.6],np.float32))
        camera=dict(camera=dict(width=32,height=32,fx=40.,fy=40.,cx=16.,cy=16.),R=np.eye(3),t=np.zeros(3))
        labels=np.ones((32,32),np.int32);labels[:,16:]=2
        risk=np.full((32,32),.25,np.float32)
        with tempfile.TemporaryDirectory() as d:
            result=accumulate_support(gaussian,camera,labels,risk,load_kernel(d),1,16,True)
            reference=render_visibility(gaussian,camera,label_map=labels,threads=1,build_dir=Path(d)/'reference')
        # Rear Gaussian contributes although front Gaussian wins most pixels.
        self.assertTrue(np.all(result['total']>0))
        np.testing.assert_allclose(result['total'],reference['mask_total_weight'],rtol=1e-6)
        np.testing.assert_allclose(result['risk_mass'],result['total']*.25,rtol=1e-6)
        for gid in (0,1):
            selected=result['support_gaussian_ids']==gid
            self.assertEqual(set(result['support_labels'][selected]),{1,2})
        np.testing.assert_allclose(np.bincount(result['support_gaussian_ids'],weights=result['support_mass'],minlength=2),result['total'],rtol=1e-6)
        np.testing.assert_allclose(np.bincount(result['cell_gaussian_ids'],weights=result['cell_mass'],minlength=2),result['total'],rtol=1e-6)

    def test_unknown_confidence_and_multiview(self):
        supports={f:dict(gaussian_ids=np.array([0,1,1],np.int32),labels=np.array([1,1,2],np.int32),mass=np.array([1.,1.,1.],np.float32)) for f in ('a','b')}
        mapping={f:np.array([-1,0,1],np.int32) for f in supports}
        ids,confidence,views,total=aggregate_assignments(supports,mapping,3,.6,2)
        np.testing.assert_array_equal(ids,[0,-1,-1]);np.testing.assert_allclose(confidence,[1,.5,0]);np.testing.assert_array_equal(views,[2,2,0])

    def test_structure_combines_sam_depth_and_rgb(self):
        labels=np.ones((16,16),np.int32);labels[:,8:]=2
        rgb=np.zeros((16,16,3),np.float32);rgb[8:]=1
        depth=np.ones((16,16),np.float32);depth[:,4:]=2
        result=structure_risk_map(rgb,labels,depth,np.ones((16,16),np.float32))
        self.assertTrue(result['sam_boundary'][:,7:9].all())
        self.assertGreater(float(result['depth_edge'][:,3:5].sum()),0)
        self.assertGreater(float(result['rgb_edge'][7:9].sum()),0)
        self.assertLessEqual(float(result['structure_risk'].max()),1.)


if __name__=='__main__':unittest.main()
