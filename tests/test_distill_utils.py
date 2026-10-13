"""CPU checks for checkpoint provenance, exact restart state and stage length."""
import os
import random
import tempfile
import unittest

import numpy as np
import torch

from utils.distill_utils import (
    atomic_json_save, atomic_torch_save, atomic_write, capture_rng_state,
    distillation_schedule, read_standard_checkpoint, restore_rng_state,
    resume_configuration, sha256_file, verify_resume_pair,
)


class DistillUtilityTests(unittest.TestCase):
    def test_schedule_counts_updates_independently_from_input_iteration(self):
        end = distillation_schedule(35000, 5000)
        self.assertEqual(end, 40000)
        self.assertEqual(len(range(35001, end + 1)), 5000)
        with self.assertRaises(ValueError):
            distillation_schedule(35000, 5000, 45000)
        with self.assertRaises(ValueError):
            distillation_schedule(35000, 0)

    def test_rng_resume_continues_all_cpu_sources(self):
        state = capture_rng_state()
        expected = (random.random(), np.random.rand(3), torch.rand(3))
        restore_rng_state(state)
        self.assertEqual(random.random(), expected[0])
        np.testing.assert_array_equal(np.random.rand(3), expected[1])
        self.assertTrue(torch.equal(torch.rand(3), expected[2]))

    def test_atomic_failure_preserves_previous_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "state.json")
            atomic_json_save({"old": True}, path)
            before = sha256_file(path)
            def fail(temporary):
                raise RuntimeError("simulated interruption")
            with self.assertRaises(RuntimeError):
                atomic_write(path, fail)
            self.assertEqual(sha256_file(path), before)
            self.assertEqual(os.listdir(directory), ["state.json"])

    def test_resume_pair_rejects_mismatched_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "checkpoint.pth")
            atomic_torch_save((tuple(range(12)), 25199), path)
            metadata = {"format": "pwsr-sh-distill-v1", "checkpoint_sha256": sha256_file(path)}
            verify_resume_pair(path, metadata)
            self.assertEqual(read_standard_checkpoint(path)[1], 25199)
            atomic_torch_save((tuple(range(12)), 25200), path)
            with self.assertRaises(ValueError):
                verify_resume_pair(path, metadata)

    def test_resume_configuration_rejects_changed_training_flags(self):
        saved = {"distill_steps": 5000, "model_path": "/old"}
        requested = {"distill_steps": 1, "model_path": "/new"}
        with self.assertRaises(ValueError):
            resume_configuration(saved, requested, {"distill_steps"})
        actual = resume_configuration(saved, requested, {"model_path"}, {"model_path"})
        self.assertEqual(actual, {"distill_steps": 5000, "model_path": "/new"})

    def test_checkpoint_cli_resume_restores_nondefault_stage_configuration(self):
        from distill_train import parse_arguments
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = os.path.join(directory, "chkpnt35001.pth")
            fresh, _, _, _, _ = parse_arguments([
                "-s", directory, "-m", directory,
                "--start_checkpoint", os.path.join(directory, "initial.pth"),
                "--teacher_model", os.path.join(directory, "teacher.pth"),
                "--distill_steps", "2", "--freeze_covariance", "--no_augmented_view",
            ])
            atomic_torch_save((tuple(range(12)), 35001), checkpoint)
            atomic_torch_save({"format": "pwsr-sh-distill-v1",
                               "configuration": vars(fresh),
                               "checkpoint_sha256": sha256_file(checkpoint)}, checkpoint + ".state.pth")
            resumed, _, _, _, _ = parse_arguments(["--resume_checkpoint", checkpoint])
            self.assertEqual(resumed.distill_steps, 2)
            self.assertEqual(resumed.iterations, 35002)
            self.assertFalse(resumed.enable_covariance)
            self.assertFalse(resumed.augmented_view)
            self.assertEqual(resumed.start_checkpoint, fresh.start_checkpoint)

    def test_adamw_restart_preserves_next_update_and_scheduler(self):
        from distill_train import make_student_optimizer
        from torch.optim.lr_scheduler import ExponentialLR
        class Student:
            pass
        student = Student()
        parameter = torch.nn.Parameter(torch.tensor([2.0, -1.0]))
        student.optimizer = torch.optim.Adam([{"params": [parameter], "lr": 0.02, "name": "f_dc"}])
        make_student_optimizer(student)
        scheduler = ExponentialLR(student.optimizer, gamma=0.9)
        for _ in range(2):
            parameter.square().sum().backward()
            student.optimizer.step()
            student.optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "optimizer.pth")
            atomic_torch_save((parameter.detach(), student.optimizer.state_dict(), scheduler.state_dict()), path)
            from utils.distill_utils import load_torch_file
            values, optimizer_state, scheduler_state = load_torch_file(path)
        resumed = Student()
        other_parameter = torch.nn.Parameter(values.clone())
        resumed.optimizer = torch.optim.Adam([{"params": [other_parameter], "lr": 0.02, "name": "f_dc"}])
        make_student_optimizer(resumed, optimizer_state)
        other_scheduler = ExponentialLR(resumed.optimizer, gamma=0.9)
        other_scheduler.load_state_dict(scheduler_state)
        for active, value, rate in ((student, parameter, scheduler),
                                    (resumed, other_parameter, other_scheduler)):
            value.square().sum().backward()
            active.optimizer.step()
            active.optimizer.zero_grad(set_to_none=True)
            rate.step()
        self.assertTrue(torch.equal(parameter, other_parameter))
        self.assertEqual(scheduler.state_dict(), other_scheduler.state_dict())


if __name__ == "__main__":
    unittest.main()
