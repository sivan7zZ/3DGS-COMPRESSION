"""CPU checks of the real standard model and SH checkpoint transformation."""
import argparse
import tempfile
from pathlib import Path
import unittest

import torch
from plyfile import PlyData
from arguments import OptimizationParams
from scene.gaussian_model import GaussianModel
from utils.checkpoint_utils import validate_model_state


def state(n=4):
    tensors = [torch.randn(n,3), torch.randn(n,1,3), torch.randn(n,15,3),
               torch.randn(n,3), torch.randn(n,4), torch.randn(n,1),
               torch.zeros(n), torch.zeros(n,1), torch.zeros(n,1)]
    return (3, *tensors, {"state": {}, "param_groups": []}, 1.0)


class CheckpointModelTests(unittest.TestCase):
    def test_reducing_sh_copies_model_without_aliasing_teacher(self):
        original = state()
        teacher = GaussianModel(3)
        student = GaussianModel(2)
        teacher.load_checkpoint_model(original, device="cpu")
        parser = argparse.ArgumentParser()
        params = OptimizationParams(parser)
        opt = params.extract(parser.parse_args([]))
        student.load_checkpoint_model(original, opt, target_sh_degree=2, device="cpu")
        self.assertEqual(student._features_rest.shape, (4,8,3))
        self.assertEqual(teacher._features_rest.shape, (4,15,3))
        self.assertFalse(teacher._xyz.requires_grad)
        self.assertTrue(student._xyz.requires_grad)
        self.assertNotEqual(teacher._xyz.data_ptr(), student._xyz.data_ptr())
        self.assertTrue(torch.equal(student._features_rest, teacher._features_rest[:,:8]))
        student._features_rest.sum().backward()
        student.optimizer.step()
        snapshot = student.capture()
        self.assertEqual(len(snapshot), 12)
        self.assertTrue(snapshot[10]["state"])
        restored = GaussianModel(2)
        restored.load_checkpoint_model(snapshot, opt, restore_optimizer=True, device="cpu")
        self.assertTrue(torch.equal(restored._features_rest, student._features_rest))
        with tempfile.TemporaryDirectory() as directory:
            ply = Path(directory) / "student.ply"
            student.save_ply(str(ply))
            names = [p.name for p in PlyData.read(str(ply))["vertex"].properties]
            self.assertEqual(len(names), 41)
            self.assertEqual(len([x for x in names if x.startswith("f_rest_")]), 24)
            self.assertEqual(set(names)-set(["x","y","z","nx","ny","nz","opacity"])-
                set("f_dc_%d" % i for i in range(3))-set("f_rest_%d" % i for i in range(24))-
                set("scale_%d" % i for i in range(3))-set("rot_%d" % i for i in range(4)), set())

    def test_invalid_checkpoints_and_sh_upgrade_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_model_state(tuple(range(14)))
        invalid = list(state())
        invalid[3] = torch.zeros(4, 12, 3)
        with self.assertRaises(ValueError):
            validate_model_state(invalid)
        with self.assertRaises(ValueError):
            GaussianModel(4).load_checkpoint_model(state(), target_sh_degree=4, device="cpu")

if __name__ == "__main__":
    unittest.main()
