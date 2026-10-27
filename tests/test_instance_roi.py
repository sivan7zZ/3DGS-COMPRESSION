"""CPU checks for small-region gradients and train-only ROI loading."""

import os
import contextlib
import io
import tempfile
import types
import unittest

import numpy as np
import torch

from utils.instance_roi import (
    TrainingInstanceROIs, boundary_band, weighted_boundary_l1, weighted_region_l1,
)


class InstanceROITests(unittest.TestCase):
    def test_colmap_filename_with_extension_uses_sam_stem_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            np.savez(os.path.join(directory, "frame_001.npz"), weights=np.ones((2, 3), dtype=np.float32))
            camera = types.SimpleNamespace(image_name="frame_001.png", image_height=2, image_width=3)
            priors = TrainingInstanceROIs(directory, [camera])
            self.assertEqual(list(priors.records), ["frame_001.png"])
            image = torch.zeros((3, 2, 3), requires_grad=True)
            region, edge = priors.losses(camera.image_name, image, torch.ones_like(image))
            self.assertAlmostEqual(region.item(), 1.0)
            self.assertTrue(torch.isfinite(edge))

    def test_region_error_is_normalized_by_roi_pixels_and_channels(self):
        image = torch.zeros((3, 4, 5), requires_grad=True)
        target = torch.zeros_like(image)
        target[:, 1, 2] = 0.8
        weights = torch.zeros((4, 5))
        weights[1, 2] = 7.0
        loss = weighted_region_l1(image, target, weights)
        self.assertAlmostEqual(loss.item(), 0.8, places=6)
        loss.backward()
        expected = torch.zeros_like(image)
        expected[:, 1, 2] = -1.0 / 3.0
        torch.testing.assert_close(image.grad, expected)
        self.assertAlmostEqual(weighted_region_l1(image, target, weights * 30).item(), 0.8, places=6)

    def test_empty_region_and_boundary_remain_differentiable_zero(self):
        image = torch.ones((3, 3, 4), requires_grad=True)
        target = torch.zeros_like(image)
        weights = torch.zeros((3, 4))
        loss = weighted_region_l1(image, target, weights) + weighted_boundary_l1(image, target, weights)
        self.assertEqual(loss.item(), 0.0)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertEqual(torch.count_nonzero(image.grad).item(), 0)

    def test_boundary_supervision_matches_contrast_not_absolute_rgb(self):
        image = torch.zeros((3, 1, 2), requires_grad=True)
        target = torch.zeros_like(image)
        target[:, :, 1] = 1.0
        edge = torch.tensor([[0.0, 2.0]])
        loss = weighted_boundary_l1(image, target, edge)
        self.assertAlmostEqual(loss.item(), 1.0, places=6)
        loss.backward()
        torch.testing.assert_close(image.grad[:, :, 0], torch.full((3, 1), 1.0 / 3.0))
        torch.testing.assert_close(image.grad[:, :, 1], torch.full((3, 1), -1.0 / 3.0))
        self.assertEqual(weighted_boundary_l1(target + 0.7, target, edge).item(), 0.0)

    def test_full_image_region_has_no_artificial_image_border(self):
        self.assertEqual(boundary_band(torch.ones((3, 4))).sum().item(), 0.0)
        self.assertEqual(boundary_band(torch.zeros((3, 4))).sum().item(), 0.0)
        mask = torch.zeros((5, 5))
        mask[2, 2] = 1
        self.assertEqual(boundary_band(mask).sum().item(), 9.0)

    def test_loader_uses_train_views_only_and_resizes_nearest(self):
        with tempfile.TemporaryDirectory() as directory:
            weights = np.array([[1, 0], [0, 0]], dtype=np.float32)
            np.savez(os.path.join(directory, "train.npz"), weights=weights,
                     small_mask=weights > 0, sampling_weight=np.float32(2.5))
            # An unrelated held-out prior must never be read or validated.
            with open(os.path.join(directory, "test.npz"), "w") as stream:
                stream.write("intentionally invalid held-out file")
            cameras = [types.SimpleNamespace(image_name="train", image_height=4, image_width=4)]
            priors = TrainingInstanceROIs(directory, cameras)
            self.assertEqual(list(priors.provenance), ["train"])
            self.assertEqual(priors.sampling_weights(["train"]), [2.5])
            self.assertEqual(tuple(priors.records["train"]["weights"].shape), (4, 4))
            self.assertEqual(priors.records["train"]["weights"].sum().item(), 4.0)
            first_hash = priors.provenance["train"]["sha256"]
            np.savez(os.path.join(directory, "train.npz"), weights=weights * 2)
            changed = TrainingInstanceROIs(directory, cameras)
            self.assertNotEqual(changed.provenance["train"]["sha256"], first_hash)

    def test_loader_rejects_invalid_values_missing_views_and_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            camera = types.SimpleNamespace(image_name="train", image_height=2, image_width=2)
            with self.assertRaises(FileNotFoundError):
                TrainingInstanceROIs(directory, [camera])
            np.savez(os.path.join(directory, "train.npz"), weights=np.array([[np.nan, 0], [0, 1]]))
            with self.assertRaisesRegex(ValueError, "finite nonnegative"):
                TrainingInstanceROIs(directory, [camera])
            camera.image_name = "../outside"
            with self.assertRaisesRegex(ValueError, "escapes"):
                TrainingInstanceROIs(directory, [camera])

    def test_cli_resume_preserves_roi_flags_and_rejects_changes(self):
        from distill_train import parse_arguments
        from utils.distill_utils import atomic_torch_save, sha256_file
        with tempfile.TemporaryDirectory() as directory:
            args, _, _, _, _ = parse_arguments([
                "-s", directory, "-m", directory,
                "--start_checkpoint", os.path.join(directory, "student.pth"),
                "--teacher_model", os.path.join(directory, "teacher.pth"),
                "--instance_roi_dir", directory, "--instance_roi_weight", "0.25",
                "--instance_edge_weight", "0.125", "--instance_balanced_sampling",
            ])
            checkpoint = os.path.join(directory, "chkpnt36000.pth")
            atomic_torch_save((tuple(range(12)), 36000), checkpoint)
            atomic_torch_save({"format": "pwsr-sh-distill-v1", "configuration": vars(args),
                               "checkpoint_sha256": sha256_file(checkpoint)}, checkpoint + ".state.pth")
            resumed, _, _, _, _ = parse_arguments(["--resume_checkpoint", checkpoint])
            self.assertEqual(resumed.instance_roi_dir, directory)
            self.assertEqual(resumed.instance_roi_weight, 0.25)
            self.assertEqual(resumed.instance_edge_weight, 0.125)
            self.assertTrue(resumed.instance_balanced_sampling)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_arguments(["--resume_checkpoint", checkpoint, "--instance_roi_weight", "0.5"])

    def test_distillation_save_preserves_source_lineage_with_current_xyz(self):
        from distill_train import save_model
        from utils.distill_utils import load_torch_file, sha256_file
        from utils.instance_pruning import InstancePruningState, xyz_sha256
        with tempfile.TemporaryDirectory() as directory:
            xyz = np.arange(192, dtype=np.float32).reshape(64, 3)
            priors_path = os.path.join(directory, "source.npz")
            np.savez(priors_path, instance_ids=np.repeat([1, 2], 32),
                     instance_confidence=np.full(64, 0.9, dtype=np.float32),
                     instance_views=np.full(64, 3, dtype=np.uint16),
                     instance_source=np.ones(64, dtype=np.uint8),
                     source_xyz_sha256=np.asarray(xyz_sha256(xyz)))
            state = InstancePruningState.load(priors_path, xyz)
            keep = np.tile(np.r_[np.ones(16, dtype=bool), np.zeros(16, dtype=bool)], 2)
            state.subset_(keep)
            current_xyz = torch.tensor(xyz[keep] + 0.01, requires_grad=True)
            checkpoint = os.path.join(directory, "chkpnt40000.pth")
            test = self
            class Student:
                get_xyz = current_xyz
                def capture(self):
                    return tuple(range(12))
                def save_ply(self, path):
                    test.assertTrue(os.path.isfile(checkpoint))
                    test.assertTrue(os.path.isfile(checkpoint + ".state.pth"))
                    with open(path, "w") as stream:
                        stream.write("ply\nformat ascii 1.0\nend_header\n")
            args = types.SimpleNamespace(model_path=directory, distill_start_iteration=35000)
            scheduler = types.SimpleNamespace(state_dict=lambda: {"last_epoch": 5000})
            with contextlib.redirect_stdout(io.StringIO()):
                _, ply_path = save_model(None, Student(), 40000, args, scheduler, [], ["train"],
                                         {}, "teacher-hash", state, {"train": {"sha256": "roi-hash"}})
            saved = load_torch_file(checkpoint + ".state.pth")
            self.assertEqual(saved["instance_metadata"]["sha256"], sha256_file(checkpoint + ".instances.npz"))
            self.assertEqual(saved["instance_roi_provenance"]["train"]["sha256"], "roi-hash")
            for path in (checkpoint + ".instances.npz", os.path.join(directory, "instance_priors.npz"),
                         os.path.join(os.path.dirname(ply_path), "instance_priors.npz")):
                restored = InstancePruningState.load_resume(path, current_xyz)
                np.testing.assert_array_equal(restored.original_ids, np.flatnonzero(keep))
                self.assertEqual(restored.source_xyz_sha256, xyz_sha256(xyz))
                with np.load(path) as value:
                    self.assertEqual(str(value["current_xyz_sha256"].item()), xyz_sha256(current_xyz))


if __name__ == "__main__":
    unittest.main()
