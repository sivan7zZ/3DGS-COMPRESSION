import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from make_reviewed_structure_evidence import main as reviewed_main
from evaluate_visible_structure import (oriented_matches, skeletonize, longest_component,
    contrast_evidence, contrast_retained, choose_cases, main, relative_gradient_matches)


class StructureMetricsTests(unittest.TestCase):
    def test_edge_distance_and_normal_angle_are_checked(self):
        ref = np.zeros((20,20),bool); ref[10,5:15]=True
        pred = np.zeros_like(ref); pred[12,5:15]=True
        b = np.zeros(ref.shape,np.int8)
        self.assertTrue(oriented_matches(ref,b,pred,b,2)[ref].all())
        self.assertFalse(oriented_matches(ref,b,pred,b,1).any())
        self.assertFalse(oriented_matches(ref,b,ref,b+4,2).any())

    def test_edge_survival_does_not_flip_due_to_canny_hysteresis_threshold(self):
        ref=np.zeros((20,20),bool);ref[10,5:15]=True
        bins=np.zeros(ref.shape,np.int8)
        strength=np.zeros(ref.shape,np.float32);strength[ref]=100
        # A 60%-strength visible edge remains supported even if its absolute
        # Canny high threshold would not be crossed.
        matched,amplitude=relative_gradient_matches(ref,bins,strength,bins,strength*.6,0,.35)
        self.assertTrue(matched[ref].all())
        self.assertTrue(np.allclose(amplitude[ref],.6))
        lost,_=relative_gradient_matches(ref,bins,strength,bins,strength*.2,0,.35)
        self.assertFalse(lost.any())

    def test_thinning_preserves_thin_connected_shape(self):
        mask=np.zeros((25,90),bool);mask[10:15,5:80]=True
        skel=skeletonize(mask)
        self.assertGreater(skel.sum(),60)
        self.assertLess(skel.sum(),80)
        self.assertEqual(longest_component(skel),int(skel.sum()))
        self.assertTrue(np.all(mask[skel]))

    def test_hole_is_not_filled_by_thinning(self):
        mask=np.zeros((40,40),np.uint8)
        cv2.rectangle(mask,(5,5),(34,34),1,5)
        skel=skeletonize(mask!=0)
        self.assertFalse(skel[20,20])
        self.assertGreater(longest_component(skel),50)

    def test_missing_and_inverted_object_contrast_are_rejected(self):
        mask=np.zeros((50,50),bool);mask[20:30,20:30]=True
        ref=np.full((50,50,3),.4,np.float32);ref[mask]=.9
        contrast,support=contrast_evidence(ref,ref,mask)
        identical,_=contrast_retained(contrast,ref,mask)
        missing,_=contrast_retained(contrast,np.full_like(ref,.4),mask)
        inverted=np.full_like(ref,.4);inverted[mask]=0
        inverse,_=contrast_retained(contrast,inverted,mask)
        self.assertGreater(support.sum(),60)
        self.assertTrue(identical[support].all())
        self.assertFalse(missing[support].any())
        self.assertFalse(inverse[support].any())
        _,teacher_missing=contrast_evidence(ref,np.full_like(ref,.4),mask)
        self.assertFalse(teacher_missing.any())

    def test_largest_missing_fragment_uses_eight_connectivity(self):
        mask=np.zeros((10,10),bool);mask[1,1]=True;mask[2,2]=True;mask[8,8]=True
        self.assertEqual(longest_component(mask),2)

    def test_selection_reserves_two_regressions(self):
        rows=[]
        for i,(b,p) in enumerate([(.1,.3),(.2,.4),(.5,.2),(.6,.3),(.8,.8)]):
            rows.append(dict(id=str(i),categories=['small_region'],models={
                'base':dict(contrast_recall=b,contrast_reference_pixels=30),
                'new':dict(contrast_recall=p,contrast_reference_pixels=30)}))
        chosen=choose_cases(rows,'base','new',2)
        negative=[r for r in chosen if r['bucket']=='proposed regression']
        self.assertEqual(len(negative),2)
        self.assertTrue(all(r['survival_improvement']<0 for r in negative))

    def fixture(self,root):
        for name in ['source','gt','teacher','base','new','sam']:(root/name).mkdir()
        (root/'test.txt').write_text('frame_000\n')
        (root/'train.txt').write_text('frame_001\n')
        labels=np.zeros((80,120),np.int32)
        labels[20:30,15:25]=1
        labels[48:51,15:105]=2
        im=np.full((80,120,3),100,np.uint8);im[labels==1]=230;im[labels==2]=20
        baseline=im.copy();baseline[labels==1]=100;baseline[48:51,40:75]=100
        for directory,arr,filename in [('source',im,'frame_000.png'),('gt',im,'00000.png'),
            ('teacher',im,'00000.png'),('base',baseline,'00000.png'),('new',im,'00000.png')]:
            Image.fromarray(arr).save(str(root/directory/filename))
        np.savez_compressed(str(root/'sam/frame_000.npz'),labels=labels,region_quality=np.array([.99,.99]))
        return ['--sam_masks',str(root/'sam'),'--test_frames',str(root/'test.txt'),
            '--train_frames',str(root/'train.txt'),'--ground_truth_dir',str(root/'gt'),
            '--source_images',str(root/'source'),'--teacher_renders',str(root/'teacher'),
            '--model','base='+str(root/'base'),'--model','new='+str(root/'new'),
            '--baseline','base','--proposed','new','--small_area_fraction','.05',
            '--output_dir',str(root/'report')]

    def test_cli_detects_object_erasure_and_line_break_and_exports_native_crops(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);args=self.fixture(root)
            with contextlib.redirect_stdout(io.StringIO()):main(args)
            data=json.loads((root/'report/visible_structure.json').read_text())
            object_row=next(r for r in data['candidates'] if r['id']=='frame_000_r1')
            thin_row=next(r for r in data['candidates'] if r['id']=='frame_000_r2')
            self.assertLess(object_row['models']['base']['contrast_recall'],.05)
            self.assertGreater(object_row['models']['new']['contrast_recall'],.95)
            self.assertIn('thin_region',thin_row['categories'])
            self.assertGreater(thin_row['models']['base']['largest_missing_centerline_fragment_px'],20)
            self.assertEqual(thin_row['models']['new']['largest_missing_centerline_fragment_px'],0)
            lines=[r for r in data['candidates'] if 'line_segment' in r['categories']]
            self.assertTrue(lines)
            self.assertTrue(any(r['models']['base']['thin_centerline_recall'] < r['models']['new']['thin_centerline_recall'] for r in lines))
            self.assertIn('interior_summary',data)
            self.assertFalse(object_row['touches_image_boundary'])
            files=list((root/'report/evidence').rglob('*_GT.png'))
            self.assertTrue(files)
            selected=json.loads((root/'report/evidence/selection.json').read_text())
            for row in selected:
                box=row['crop_bbox_xyxy']
                with Image.open(row['native_crops']['GT']) as image:
                    self.assertEqual(image.size,(box[2]-box[0],box[3]-box[1]))

    def test_fixed_reviewed_crops_preserve_exact_native_pixels(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.fixture(root)
            review=root/'reviews.json'
            review.write_text(json.dumps([dict(frame='frame_000',bbox_xyxy=[10,15,35,35],
                category='small_region',review_status='known synthetic object')]))
            args=['--reviews',str(review),'--test_frames',str(root/'test.txt'),
                '--source_images',str(root/'source'),'--ground_truth_dir',str(root/'gt'),
                '--teacher_renders',str(root/'teacher'),'--model','base='+str(root/'base'),
                '--output_dir',str(root/'reviewed')]
            with contextlib.redirect_stdout(io.StringIO()):reviewed_main(args)
            result=json.loads((root/'reviewed/reviewed_native_manifest.json').read_text())
            native=result['cases'][0]['files']['base']['native_crop']
            with Image.open(native) as crop, Image.open(root/'base/00000.png') as original:
                self.assertTrue(np.array_equal(np.asarray(crop),np.asarray(original)[15:35,10:35]))
            self.assertFalse(result['native_crops_resized'])

    def test_ground_truth_cannot_be_silently_used_as_teacher(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);args=self.fixture(root)
            args[args.index('--teacher_renders')+1]=str(root/'gt')
            with self.assertRaisesRegex(ValueError,'not the GT directory'):
                main(args)

    def test_wrong_gt_frame_mapping_fails_before_reporting(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);args=self.fixture(root)
            Image.fromarray(np.zeros((80,120,3),np.uint8)).save(str(root/'gt/00000.png'))
            with self.assertRaisesRegex(ValueError,'mapping mismatch'):
                main(args)
            self.assertFalse((root/'report/visible_structure.json').exists())


if __name__=='__main__':unittest.main()
