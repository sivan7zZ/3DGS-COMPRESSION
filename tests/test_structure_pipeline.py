import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import run_pipeline


class StructurePipelineTests(unittest.TestCase):
    def test_structure_priors_follow_both_stages_and_control_omits_them(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / 'input.ply').touch(); (root / 'teacher.pth').touch()
            config = dict(source_path=str(root), input_ply=str(root/'input.ply'),
                          teacher_checkpoint=str(root/'teacher.pth'), instance_metadata='labels.npz',
                          instance_roi_dir='roi', structure_priors='structures.npz',
                          structure_thin_weight=.03, structure_score_weight=.35, structure_selection_mode="adaptive")
            with patch.object(run_pipeline, 'ROOT', root):
                _, plan = run_pipeline.build_plan(config)
            self.assertIn('--structure_priors', plan[0]['command'])
            self.assertIn('--structure_score_weight', plan[0]['command'])
            self.assertIn('--structure_priors', plan[1]['command'])
            self.assertIn('--structure_thin_weight', plan[1]['command'])
            for command in (plan[0]['command'], plan[1]['command']):
                self.assertEqual(command[command.index('--structure_selection_mode')+1], 'adaptive')
            config['pre_distill_analysis'] = True
            with patch.object(run_pipeline, 'ROOT', root):
                _, ordered = run_pipeline.build_plan(config)
            self.assertEqual([x['name'] for x in ordered],
                             ['prune', 'render_stage3', 'distill', 'render_sh2', 'metrics'])
            config['disable_instance_protection'] = True
            with patch.object(run_pipeline, 'ROOT', root):
                _, plan = run_pipeline.build_plan(config)
            self.assertNotIn('--structure_priors', plan[0]['command'])
            self.assertNotIn('--instance_roi_dir', plan[0]['command'])
            self.assertNotIn('--structure_priors', plan[1]['command'])


if __name__ == '__main__':
    unittest.main()
