import unittest
import torch
import numpy as np
from types import SimpleNamespace

from utils.structure_losses import boundary_survival_loss, thin_survival_loss, AnchorTrustRegion


class StructureLossTests(unittest.TestCase):
    def test_erased_boundary_has_gradient_and_intact_boundary_is_preferred(self):
        truth = torch.zeros(3, 12, 12); truth[:, :, 6:] = .6
        weights = torch.ones(12, 12)
        erased = torch.zeros_like(truth, requires_grad=True)
        missing = boundary_survival_loss(erased, truth, weights)
        intact = boundary_survival_loss(truth, truth, weights)
        self.assertGreater(float(missing), float(intact))
        missing.backward(); self.assertGreater(float(erased.grad.abs().sum()), 0)

    def test_thin_line_gap_and_reversed_contrast_are_penalized(self):
        truth = torch.full((3, 32, 48), .5); truth[:, 15:17, 3:45] = .9
        weights = torch.zeros(32, 48); weights[15:17, 3:45] = 1
        gap = truth.clone(); gap[:, :, 18:28] = .5; gap.requires_grad_()
        intact = thin_survival_loss(truth, truth, weights)
        missing = thin_survival_loss(gap, truth, weights)
        inverse = thin_survival_loss(1-truth, truth, weights)
        self.assertGreater(float(missing), float(intact))
        self.assertGreater(float(inverse), float(missing))
        missing.backward(); self.assertGreater(float(gap.grad.abs().sum()), 0)

    def test_empty_structure_is_differentiable_zero(self):
        image = torch.rand(3, 9, 11, requires_grad=True)
        target = torch.rand_like(image); weights = torch.zeros(9, 11)
        loss = boundary_survival_loss(image, target, weights) + thin_survival_loss(image, target, weights)
        self.assertAlmostEqual(float(loss), 0.)
        loss.backward(); self.assertTrue(torch.isfinite(image.grad).all())

    def test_flat_target_has_no_artificial_structure_penalty(self):
        image = torch.rand(3, 9, 11, requires_grad=True)
        target = torch.full_like(image, .5); weights = torch.ones(9, 11)
        self.assertAlmostEqual(float(thin_survival_loss(image, target, weights)), 0.)

    def test_exact_low_contrast_reference_is_not_forced_to_oversharpen(self):
        boundary = torch.full((3, 12, 12), .5); boundary[:, :, 6:] = .51
        self.assertAlmostEqual(float(boundary_survival_loss(boundary, boundary, torch.ones(12, 12))), 0.)
        line = torch.full((3, 24, 24), .5); line[:, 11:13, 2:22] = .53
        mask = torch.zeros(24, 24); mask[11:13, 2:22] = 1
        self.assertAlmostEqual(float(thin_survival_loss(line, line, mask)), 0., places=5)

    def test_anchor_fading_drift_and_shrinking_receive_gradients(self):
        prior = SimpleNamespace(anchors=np.array([1]), anchor_xyz=np.zeros((1, 3), np.float32),
            anchor_log_scaling=np.zeros((1, 3), np.float32), anchor_opacity=np.array([.8], np.float32),
            anchor_features_dc=np.zeros((1, 3), np.float32), anchor_strength=np.ones(1, np.float32))
        class Model:
            get_xyz = torch.tensor([[0., 0., 0.], [3., 0., 0.]], requires_grad=True)
            get_opacity = torch.tensor([[.8], [.001]], requires_grad=True)
            _scaling = torch.tensor([[0., 0., 0.], [-2., -2., -2.]], requires_grad=True)
            _features_dc = torch.ones(2, 1, 3, requires_grad=True)
        model = Model()
        loss = AnchorTrustRegion(prior).loss(model, np.array([0, 1]))
        self.assertGreater(float(loss), 0.)
        loss.backward()
        self.assertGreater(float(model.get_xyz.grad[1].abs().sum()), 0.)
        self.assertLess(float(model.get_opacity.grad[1]), 0.)
        self.assertLess(float(model._scaling.grad[1].sum()), 0.)
        with self.assertRaises(ValueError):
            AnchorTrustRegion(prior).loss(model, np.array([0, 2]))


if __name__ == '__main__':
    unittest.main()
