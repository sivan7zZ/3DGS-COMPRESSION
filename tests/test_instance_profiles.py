import unittest

import numpy as np

from prepare_instance_protection import make_profiles, roi_weights


class InstanceProfileTests(unittest.TestCase):
    def fixture(self):
        return (dict(instance_ids=np.array([1, 1, 1, 1, 2, 2]),
                     instance_source=np.array([1, 1, 1, 2, 1, 2]),
                     instance_confidence=np.array([.8] * 6), instance_views=np.array([3] * 6)),
                [dict(instance_id=1, gaussian_count=4), dict(instance_id=2, gaussian_count=2)],
                dict(nodes=[dict(instance_id=iid, frame=f, area=area)
                            for iid, area in [(1, 2), (2, 20)] for f in ["a", "b", "c"]]))

    def test_small_reliable_separate_from_large(self):
        metadata, catalog, tracks = self.fixture()
        profiles = make_profiles(metadata, catalog, tracks, ["a", "b", "c"],
                                 {f: (10, 10) for f in ["a", "b", "c"]},
                                 small_area_fraction=.05, min_direct_count=3)
        self.assertTrue(profiles[0]["small"])
        self.assertTrue(profiles[0]["selected_for_roi"])
        self.assertFalse(profiles[1]["selected_for_roi"])
        self.assertFalse(profiles[1]["small"])

    def test_reject_holdout_tracks(self):
        metadata, catalog, tracks = self.fixture()
        with self.assertRaises(ValueError):
            make_profiles(metadata, catalog, tracks, ["a", "b"], {f: (10, 10) for f in ["a", "b"]})

    def test_each_instance_gets_equal_mass_and_separate_edges(self):
        ids = np.array([[1, 1, 2, 0], [1, 1, 2, 0]], dtype=np.int32)
        weights, edges = roi_weights(ids)
        self.assertAlmostEqual(float(weights[ids == 1].sum()), float(weights[ids == 2].sum()))
        self.assertAlmostEqual(float(weights.mean()), 1.)
        self.assertTrue(np.all(weights[ids == 0] == 0))
        self.assertTrue(np.all(edges[:, 1:3] > 0))
        self.assertTrue(np.all(edges[:, 0] == 0))

    def test_empty_view_has_zero_extra_losses(self):
        weights, edges = roi_weights(np.zeros((3, 4), dtype=np.int32))
        self.assertFalse(weights.any())
        self.assertFalse(edges.any())


if __name__ == "__main__":
    unittest.main()
