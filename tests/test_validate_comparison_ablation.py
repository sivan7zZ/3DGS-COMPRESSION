"""Camera-equivalence checks must tolerate PNG suffixes but reject wrong views."""
import unittest
from validate_comparison_ablation import assert_test_frames


class AblationValidationFrameTests(unittest.TestCase):
    def test_png_and_stem_names_are_equivalent(self):
        actual=['frame_%03d.png'%i for i in range(0,70,8)]
        expected=['frame_%03d'%i for i in range(0,70,8)]
        self.assertEqual(assert_test_frames(actual,expected),expected)
        self.assertEqual(assert_test_frames(expected,actual),expected)

    def test_wrong_frame_and_order_rejected(self):
        expected=['frame_000','frame_008','frame_016']
        with self.assertRaises(ValueError):
            assert_test_frames(['frame_000.png','frame_009.png','frame_016.png'],expected)
        with self.assertRaises(ValueError):
            assert_test_frames(['frame_008.png','frame_000.png','frame_016.png'],expected)

    def test_missing_or_extra_views_rejected(self):
        with self.assertRaises(ValueError):
            assert_test_frames(['frame_000.png'],['frame_000','frame_008'])
        with self.assertRaises(ValueError):
            assert_test_frames(['frame_000.png','frame_008.png'],['frame_000'])


if __name__=='__main__':
    unittest.main()
