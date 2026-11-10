import unittest

from run_boundary_weighted_evaluation import VGGT, worker_environment


class SemanticWorkerEnvironmentTests(unittest.TestCase):
    def test_cross_environment_worker_does_not_load_parent_cuda_libraries(self):
        parent = {'LD_LIBRARY_PATH': '/wrong/torch/lib:/wrong/cudnn',
                  'LD_PRELOAD': '/wrong/library.so', 'PATH': '/normal/bin', 'SAFE_SETTING': 'kept'}
        environment = worker_environment(VGGT, parent)
        self.assertNotIn('LD_LIBRARY_PATH', environment)
        self.assertNotIn('LD_PRELOAD', environment)
        self.assertEqual(environment['SAFE_SETTING'], 'kept')
        self.assertEqual(environment['PATH'], parent['PATH'])
        self.assertIn('LD_LIBRARY_PATH', parent)

    def test_native_renderer_keeps_its_own_runtime_linker_settings(self):
        parent = {'LD_LIBRARY_PATH': '/local/runtime:/prune/torch/lib', 'PATH': '/normal/bin'}
        environment = worker_environment('/home/zhangys/anaconda3/envs/prune_wisely/bin/python', parent)
        self.assertEqual(environment, parent)
        self.assertIsNot(environment, parent)


if __name__ == '__main__':
    unittest.main()
