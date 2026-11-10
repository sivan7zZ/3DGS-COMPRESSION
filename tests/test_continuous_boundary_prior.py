from pathlib import Path
import tempfile
import unittest

import numpy as np

from prepare_continuous_boundary import compare_visibility, self_check
from utils.continuous_boundary_prior import (
    accumulate_projected, continuous_maps, load_kernel,
)


class ContinuousBoundaryMapsTests(unittest.TestCase):
    def test_sam_distance_has_true_gaussian_decay_without_hard_dilation(self):
        labels = np.zeros((11, 21), np.int32)
        labels[:, 10:] = 1
        rgb = np.zeros((11, 21, 3), np.float32)
        maps = continuous_maps(rgb, labels, np.ones(labels.shape), np.ones(labels.shape), 2.)
        self.assertEqual(maps.dtype, np.float32)
        np.testing.assert_allclose(maps[0, 5, [9, 10]], 1)
        np.testing.assert_allclose(maps[0, 5, [8, 11]], np.exp(-1 / 8), rtol=1e-6)
        np.testing.assert_allclose(maps[0, 5, [7, 12]], np.exp(-4 / 8), rtol=1e-6)
        self.assertTrue(np.any((maps[0] > 0) & (maps[0] < 1)))
        self.assertTrue(np.all(maps[1:] == 0))

    def test_edge_free_sam_and_invalid_depth_remain_zero(self):
        labels = np.full((9, 13), 4, np.int32)
        depth = np.arange(labels.size, dtype=np.float32).reshape(labels.shape)
        depth[2:4, 2:4] = np.nan
        maps = continuous_maps(np.zeros(labels.shape + (3,), np.float32), labels, depth, np.zeros(labels.shape))
        self.assertTrue(np.all(maps == 0))

    def test_depth_validity_and_rgb_channels_are_independent(self):
        y, x = np.mgrid[:17, :19]
        rgb = np.stack([x / 18., x / 18., x / 18.], -1).astype(np.float32)
        depth = (x + .2 * y).astype(np.float32)
        maps = continuous_maps(rgb, np.ones(x.shape, np.int32), depth, np.ones(x.shape))
        self.assertTrue(np.all(maps[0] == 0))
        self.assertGreater(float(maps[1].mean()), 0)
        self.assertGreater(float(maps[2].mean()), 0)
        maps_invalid = continuous_maps(rgb, np.ones(x.shape, np.int32), depth, np.full(x.shape, .5))
        self.assertTrue(np.all(maps_invalid[1] == 0))
        np.testing.assert_array_equal(maps_invalid[2], maps[2])

    def test_rejects_misaligned_inputs_and_invalid_sigma(self):
        labels = np.ones((3, 4), np.int32)
        for sigma in (0, -1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                continuous_maps(np.zeros((3, 4, 3)), labels, np.ones((3, 4)), np.ones((3, 4)), sigma)
        with self.assertRaises(ValueError):
            continuous_maps(np.zeros((3, 5, 3)), labels, np.ones((3, 4)), np.ones((3, 4)))


class ContinuousProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.kernel = load_kernel(Path(cls.directory.name) / 'build')

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_native_anisotropic_occlusion_matches_independent_reference(self):
        receipt = self_check(Path(self.directory.name), self.kernel)
        self.assertEqual(receipt['status'], 'passed')
        self.assertLess(receipt['max_mass_absolute_error'], 1e-5)

    def test_occluded_back_gaussian_has_lower_mass_and_constant_maps_are_preserved(self):
        projected = np.array([[8, 7, 2, .4, 0, .4, .98, 10, 8, 8],
                              [8, 7, 3, .4, 0, .4, .98, 10, 8, 8]], np.float32)
        maps = np.stack([np.full((16, 18), value, np.float32) for value in (.2, .5, .8)])
        result = accumulate_projected(projected, np.array([0, 2]), np.array([0, 1]), maps, 4, self.kernel)
        self.assertLess(result['total'][2], result['total'][0])
        expected = np.asarray([.2, .5, .8])
        for index in (0, 2):
            np.testing.assert_allclose(result['mass'][index] / result['total'][index], expected, atol=1e-6)
        self.assertEqual(float(result['total'][1]), 0)
        self.assertEqual(float(result['total'][3]), 0)
        self.assertTrue(np.all(result['mass'][[1, 3]] == 0))

    def test_rejects_out_of_range_map_and_wrong_source_rows(self):
        p = np.array([[2, 2, 2, .4, 0, .4, .8, 8, 6, 6]], np.float32)
        maps = np.ones((3, 5, 5), np.float32)
        with self.assertRaises(ValueError):
            accumulate_projected(p, np.array([2]), np.array([0]), maps, 2, self.kernel)
        maps[0, 0, 0] = 2
        with self.assertRaises(ValueError):
            accumulate_projected(p, np.array([0]), np.array([0]), maps, 2, self.kernel)

    def test_projection_audit_rejects_missing_visibility(self):
        with self.assertRaises(ValueError):
            compare_visibility(np.array([0., 1.]), np.array([.1, 1.]))
        self.assertTrue(compare_visibility(np.array([0., 1. + 1e-6]), np.array([0., 1.]))['passed'])


if __name__ == '__main__':
    unittest.main()
