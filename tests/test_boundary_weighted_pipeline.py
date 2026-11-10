import copy
import json
from pathlib import Path
import unittest

from run_boundary_weighted_pipeline import commands


class BoundaryWeightedPipelineTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.path = self.root / 'configs/city_boundary_weighted_10x.json'
        self.config = json.loads(self.path.read_text())

    def test_only_compression_is_planned_without_teacher_settings(self):
        initial, output = commands(self.path, self.config)
        self.assertEqual(initial[0], 'train_pup_rir_compression.py')
        self.assertEqual(output.name, 'full_10x')
        self.assertNotIn('teacher_checkpoint', self.config)
        self.assertNotIn('distill_steps', self.config)

    def test_distillation_config_cannot_enter_new_pipeline(self):
        for key, value in [('distill_steps', 0), ('teacher_checkpoint', 'teacher.pth')]:
            changed = copy.deepcopy(self.config)
            changed[key] = value
            with self.assertRaises(ValueError):
                commands(self.path, changed)

    def test_wrong_scoring_or_output_cannot_silently_run(self):
        for key, value in [('pruning_score', 'pup'), ('output_root', '/tmp/outside_repo')]:
            changed = copy.deepcopy(self.config)
            changed[key] = value
            with self.assertRaises(ValueError):
                commands(self.path, changed)


if __name__ == '__main__':
    unittest.main()
