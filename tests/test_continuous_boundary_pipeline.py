"""Check real continuous/binary configurations enter only the SH3 pipeline."""
import contextlib
import copy
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from run_boundary_weighted_pipeline import commands,main


ROOT=Path(__file__).resolve().parents[1]


def load(name):
    path=ROOT/'configs'/name
    return path,json.loads(path.read_text())


class ContinuousBoundaryPipelineTests(unittest.TestCase):
    def test_both_full_budgets_use_continuous_mode_without_distillation(self):
        for ratio,target in ((10,178809),(20,89405)):
            with self.subTest(ratio=ratio):
                path,config=load('city_continuous_boundary_%dx.json'%ratio)
                command,output=commands(path,config)
                self.assertEqual(config['pruning_score'],'pup_continuous_boundary')
                self.assertEqual(config['pruning_target'],target)
                self.assertEqual(command,['train_pup_rir_compression.py','--config',path])
                self.assertEqual(output,ROOT/('output/city2_continuous_boundary_20261109/full_%dx'%ratio))
                self.assertEqual(config['boundary_score'],dict(sam_weight=.4,depth_weight=.4,rgb_edge_weight=.2,gain=5.,fusion='log1p'))
                self.assertEqual(config['boundary_protection']['mode'],'none')
                self.assertNotIn('teacher_checkpoint',config)
                self.assertNotIn('distill_steps',config)

    def test_10x_and_20x_share_every_method_setting_and_prior_except_budget_output(self):
        _,ten=load('city_continuous_boundary_10x.json')
        _,twenty=load('city_continuous_boundary_20x.json')
        for config in (ten,twenty):
            config.pop('output_root')
            config.pop('pruning_target')
        self.assertEqual(ten,twenty)

    def test_smoke_is_separate_and_binary_configuration_still_enters_legacy_mode(self):
        smoke_path,smoke=load('city_continuous_boundary_smoke.json')
        command,output=commands(smoke_path,smoke,smoke_rounds=1)
        self.assertEqual(output.name,'smoke')
        self.assertEqual(command[-2:],['--smoke_rounds',1])
        binary_path,binary=load('city_boundary_weighted_10x.json')
        old_command,old_output=commands(binary_path,binary)
        self.assertEqual(binary['pruning_score'],'pup_boundary_weighted')
        self.assertEqual(old_command,['train_pup_rir_compression.py','--config',binary_path])
        self.assertEqual(old_output,ROOT/'output/city2_boundary_weighted_20261106/full_10x')

    def test_actual_dry_run_advertises_exact_three_sh3_only_stages(self):
        for filename in ('city_continuous_boundary_10x.json','city_continuous_boundary_20x.json','city_boundary_weighted_10x.json'):
            path,_=load(filename)
            buffer=io.StringIO()
            with self.subTest(config=filename),patch('sys.argv',['pipeline','--config',str(path),'--dry_run']),contextlib.redirect_stdout(buffer):
                main()
            planned=json.loads(buffer.getvalue())
            self.assertEqual(planned['stages'],['compress','render_stage3','metrics'])
            self.assertIs(planned['distillation'],False)
            self.assertEqual(planned['compression'][0],'train_pup_rir_compression.py')
            self.assertFalse(any('distill' in argument or 'sh2' in argument for argument in planned['compression']))

    def test_teacher_or_distillation_fields_remain_rejected_for_continuous_mode(self):
        path,config=load('city_continuous_boundary_10x.json')
        for key,value in (('teacher_checkpoint','old.pth'),('distill_steps',0)):
            changed=copy.deepcopy(config)
            changed[key]=value
            with self.subTest(key=key):
                with self.assertRaises(ValueError):commands(path,changed)


if __name__=='__main__':unittest.main()
