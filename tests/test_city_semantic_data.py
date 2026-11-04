import collections
import unittest

import numpy as np

from utils.city_semantic_data import (INNER_TRAIN, INNER_VAL, TEST,
                                      combine_binary_masks, make_registration_maps,
                                      register_labels)


class CitySemanticRegistrationTests(unittest.TestCase):
    def test_disjoint_class_masks_preserve_ignore_zero(self):
        masks = [np.zeros((5, 9), bool) for _ in range(6)]
        for index, mask in enumerate(masks):
            mask[1, index] = True
        result = combine_binary_masks(masks)
        np.testing.assert_array_equal(result[1, :6], np.arange(1, 7))
        self.assertEqual(int(result[0, 0]), 0)

    def test_overlapping_annotation_masks_fail(self):
        masks = [np.ones((2, 3), bool)] * 6
        with self.assertRaises(ValueError):
            combine_binary_masks(masks)

    def test_colmap_registration_preserves_native_identity_pixels(self):
        Camera = collections.namedtuple("Camera", "model width height params")
        old = Camera("OPENCV", 9, 5, np.array([10, 11, 4, 2, 0, 0, 0, 0]))
        new = Camera("PINHOLE", 9, 5, np.array([10, 11, 4, 2]))
        map_x, map_y = make_registration_maps(old, new)
        label = (np.arange(45).reshape(5, 9) % 7).astype(np.uint8)
        np.testing.assert_array_equal(register_labels(label, map_x, map_y), label)

    def test_registered_out_of_bounds_stays_ignore(self):
        labels = np.full((3, 4), 6, np.uint8)
        map_x, map_y = np.meshgrid(np.arange(6, dtype=np.float32)-1,
                                  np.arange(5, dtype=np.float32)-1)
        result = register_labels(labels, map_x, map_y)
        self.assertTrue(np.all(result[0] == 0))
        self.assertTrue(np.all(result[:, -1] == 0))
        self.assertTrue(np.all(result[1:4, 1:5] == 6))

    def test_split_has_no_semantic_test_leak(self):
        self.assertEqual(len(INNER_TRAIN), 6)
        self.assertEqual(len(INNER_VAL), 2)
        self.assertEqual(len(TEST), 2)
        self.assertFalse(set(INNER_TRAIN) & set(INNER_VAL))
        self.assertFalse(set(INNER_TRAIN + INNER_VAL) & set(TEST))
        # Every semantic held-out image must also be held out by 3DGS every-8.
        self.assertTrue(all(int(name.split("_")[-1]) % 8 == 0 for name in TEST))


class CitySemanticNativeInferenceTests(unittest.TestCase):
    def test_tiles_cover_trailing_rows_and_columns_without_resizing(self):
        import torch
        from city_semantic import predict_native

        class PixelClassifier(torch.nn.Module):
            def forward(self, rgb):
                logits = torch.zeros((rgb.shape[0], 6, rgb.shape[2], rgb.shape[3]))
                logits[:, 0] = 4*rgb[:, 0]
                logits[:, 1] = -4*rgb[:, 0]
                return logits

        rgb = np.zeros((53, 79, 3), np.uint8)
        rgb[:, ::2, 0] = 255
        expected = np.where(rgb[..., 0] > 128, 1, 2).astype(np.uint8)
        actual = predict_native(PixelClassifier(), rgb, "cpu", tile=32, stride=24, batch_size=2)
        np.testing.assert_array_equal(actual, expected)

    def test_inference_rejects_uncovered_tile_stride(self):
        from city_semantic import predict_native
        with self.assertRaises(ValueError):
            predict_native(None, np.zeros((10, 10, 3), np.uint8), "cpu", tile=32, stride=40)


if __name__ == "__main__":
    unittest.main()
