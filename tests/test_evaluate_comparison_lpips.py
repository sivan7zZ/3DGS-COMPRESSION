import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from evaluate_comparison_lpips import (
    FRAMES, MAIN_IDS, NAMES, cache_valid, collect_jobs, json_signature, standard_input,
)


class StandardLpipsProtocolTests(unittest.TestCase):
    def test_rgb_uint8_maps_to_standard_range_and_chw_float32(self):
        image = np.array([[[0, 127, 255], [255, 0, 128]]], dtype=np.uint8)
        out = standard_input(image)
        self.assertEqual(out.shape, (3, 1, 2))
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(out[0, 0, 0], -1)
        self.assertEqual(out[2, 0, 0], 1)
        self.assertAlmostEqual(float(out[1, 0, 0]), 2 * 127 / 255 - 1, places=7)
        self.assertTrue(out.flags.c_contiguous)
        with self.assertRaises(ValueError):
            standard_input(image.astype(np.float32))

    def test_range_conversion_equals_imagenet_normalization(self):
        rng = np.random.RandomState(42)
        rgb01 = rng.uniform(0, 1, size=(3, 8, 9))
        local_shift = np.array([-0.030, -0.088, -0.188])[:, None, None]
        local_scale = np.array([0.458, 0.448, 0.450])[:, None, None]
        image_mean = np.array([0.485, 0.456, 0.406])[:, None, None]
        image_std = np.array([0.229, 0.224, 0.225])[:, None, None]
        standardized = (2 * rgb01 - 1 - local_shift) / local_scale
        expected = (rgb01 - image_mean) / image_std
        self.assertTrue(np.allclose(standardized, expected, atol=1e-14))
        legacy = (rgb01 - local_shift) / local_scale
        self.assertGreater(float(np.abs(legacy - expected).mean()), 0.5)

    def test_cache_signature_binds_image_weights_and_legacy_sources(self):
        original = {'criterion': {'weights_sha': 'a'}, 'images': [{'sha': 'b'}], 'legacy_sha': 'c'}
        a = json_signature(original)
        self.assertEqual(a, json_signature({'images': [{'sha': 'b'}], 'legacy_sha': 'c', 'criterion': {'weights_sha': 'a'}}))
        for changed in ({'criterion': {'weights_sha': 'z'}, 'images': [{'sha': 'b'}], 'legacy_sha': 'c'},
                        {'criterion': {'weights_sha': 'a'}, 'images': [{'sha': 'z'}], 'legacy_sha': 'c'},
                        {'criterion': {'weights_sha': 'a'}, 'images': [{'sha': 'b'}], 'legacy_sha': 'z'}):
            self.assertNotEqual(a, json_signature(changed))
        result = {'status': 'complete', 'cache_signature': a, 'per_view': {n: 0.2 for n in NAMES},
                  'LPIPS': 0.2, 'self_control': {'status': 'passed', 'value': 0.0}}
        self.assertTrue(cache_valid(result, a))
        self.assertFalse(cache_valid(result, 'changed'))
        result['self_control']['value'] = 0.1
        self.assertFalse(cache_valid(result, a))
        result['self_control']['value'] = 0
        result['LPIPS'] = 0.3
        self.assertFalse(cache_valid(result, a))

    def test_default_requires_all_ablations_but_main_only_is_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = [{'id': ident, 'status': 'prepared', 'comparison_dir': str(root / ident),
                       'common_render_dir': str(root / ident / 'renders'),
                       'common_gt_dir': str(root / ident / 'gt'), 'iteration': 30000}
                      for ident in sorted(MAIN_IDS)]
            (root / 'models.json').write_text(json.dumps({'models': models, 'test_frames': FRAMES}))
            self.assertEqual(len(collect_jobs(root, main_only=True)), 13)
            (root / 'ablations20').mkdir()
            (root / 'ablations20/ablation_plan.json').write_text(json.dumps({'variants': []}))
            (root / 'ablations20/ablation_manifest.json').write_text(json.dumps({'status': 'running'}))
            with self.assertRaises(ValueError):
                collect_jobs(root)


if __name__ == '__main__':
    unittest.main()
