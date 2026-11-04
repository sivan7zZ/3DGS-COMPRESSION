"""Guard scientific interventions and SH3-only queue configuration."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path('/home/zhangys/Prune-Wisely-Reconstruct-Sharply')
spec = importlib.util.spec_from_file_location('ablation_queue', str(ROOT / 'run_comparison_ablations.py'))
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


class ComparisonAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = json.loads(queue.BASE_CONFIG.read_text())

    def test_unique_single_module_interventions(self):
        self.assertEqual(len(queue.VARIANTS), 7)
        self.assertEqual(len({x[0] for x in queue.VARIANTS}), 7)
        self.assertEqual(len({x[1] for x in queue.VARIANTS}), 7)
        for slug, key in queue.VARIANTS:
            self.assertTrue(self.base[key]['enabled'])
            actual = json.loads((queue.CONFIG_DIRECTORY / (slug + '.json')).read_text())
            differences = queue.diff(self.base, actual)
            self.assertEqual({x['key'] for x in differences}, {key + '.enabled', 'output_root'})
            self.assertIs(actual[key]['enabled'], False)

    def test_shared_data_and_exact_budget(self):
        outputs = []
        for slug, key in queue.VARIANTS:
            actual = json.loads((queue.CONFIG_DIRECTORY / (slug + '.json')).read_text())
            self.assertEqual(actual['pruning_target'], 89405)
            self.assertEqual(actual['source_path'], self.base['source_path'])
            self.assertEqual(actual['input_ply'], self.base['input_ply'])
            self.assertEqual(actual['shared_metadata'], self.base['shared_metadata'])
            self.assertTrue(actual['preserve_v3_losses'])
            self.assertEqual(actual['recovery_losses'], self.base['recovery_losses'])
            path = Path(actual['output_root'])
            self.assertIn((ROOT / 'output').resolve(), path.resolve().parents)
            outputs.append(path)
        self.assertEqual(len(set(outputs)), 7)

    def test_no_adaptive_is_fixed_relative_schedule(self):
        from utils.adaptive_schedule import AdaptiveSchedule
        cfg = json.loads((queue.CONFIG_DIRECTORY / 'no_adaptive_schedule.json').read_text())
        schedule = AdaptiveSchedule(cfg['adaptive_compression'])
        current, rounds = 1788093, 0
        while current > 89405:
            current = schedule.round_target(current, 89405)
            schedule.update(1., 2., 1.99)
            self.assertEqual(schedule.ratio, .2)
            rounds += 1
        self.assertEqual(rounds, 14)
        self.assertFalse(schedule.recovery_done([1.] * 199)[0])
        self.assertTrue(schedule.recovery_done([1.] * 200)[0])
        self.assertFalse(schedule.recovery_done([1.] * 1499, final=True)[0])
        self.assertTrue(schedule.recovery_done([1.] * 1500, final=True)[0])


if __name__ == '__main__':
    unittest.main()
