import unittest

import torch
from utils.pruning_scores import unpack_fisher, fisher_logdet, fixed_budget_mask


class PruningScoreTests(unittest.TestCase):
    def test_logdet_matches_upstream_svd_for_full_fisher_blocks(self):
        torch.manual_seed(7)
        jacobian = torch.randn(11, 12, 6)
        blocks = jacobian.transpose(1, 2) @ jacobian
        rows, cols = torch.triu_indices(6, 6)
        packed = blocks[:, rows, cols]
        self.assertTrue(torch.equal(unpack_fisher(packed), blocks))
        expected = torch.linalg.svdvals(blocks).log().sum(1)
        self.assertTrue(torch.allclose(fisher_logdet(packed, chunk_size=3), expected))

    def test_rank_deficient_fishers_and_equal_scores_keep_exact_budget(self):
        scores = fisher_logdet(torch.zeros(9, 21))
        self.assertTrue(torch.isneginf(scores).all())
        mask = fixed_budget_mask(scores, 5)
        self.assertEqual(mask.sum().item(), 5)
        self.assertTrue(mask[:5].all())
        self.assertFalse(mask[5:].any())

    def test_remove_low_scores_and_reject_nan(self):
        self.assertEqual(fixed_budget_mask(torch.tensor([3., -2., 8., 0.]), 2).tolist(),
                         [False, True, False, True])
        with self.assertRaises(FloatingPointError):
            fixed_budget_mask(torch.tensor([float("nan"), 1.]), 1)
        with self.assertRaises(ValueError):
            fixed_budget_mask(torch.ones(3), 3)


if __name__ == "__main__":
    unittest.main()
