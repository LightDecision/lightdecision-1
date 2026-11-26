import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from torch import nn
    from lightdecision.models.decision.head import DecisionHead
    from lightdecision.models.decision.model import LightDecisionModel
    from lightdecision.models.decision.pooling import group_branch_features

    class TinyCausalAdapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(40, 12)
            self.projection = nn.Linear(12, 12)

        def get_hidden_size(self):
            return 12

        def forward_hidden(self, input_ids, attention_mask, position_ids=None):
            values = self.embedding(input_ids) * attention_mask.unsqueeze(-1)
            count = attention_mask.cumsum(dim=1).clamp_min(1).unsqueeze(-1)
            return self.projection(values.cumsum(dim=1) / count)


@unittest.skipUnless(HAS_TORCH, "torch is not installed in this interpreter")
class DecisionHeadTests(unittest.TestCase):
    def make_head(self, **kwargs):
        return DecisionHead(12, set_dim=16, set_heads=4, ffn_dim=32, **kwargs)

    def test_permutation_equivariance_with_padding(self):
        head = self.make_head().eval()
        q, e = torch.randn(2, 12), torch.randn(2, 7, 12)
        mask = torch.tensor([[True] * 7, [True] * 4 + [False] * 3])
        permutation = torch.tensor([4, 1, 6, 2, 0, 5, 3])
        expected = head(q, e, mask)
        actual = head(q, e[:, permutation], mask[:, permutation])
        torch.testing.assert_close(actual.probabilities, expected.probabilities[:, permutation])
        torch.testing.assert_close(expected.probabilities.sum(-1), torch.ones(2))
        self.assertEqual(expected.probabilities[~mask].abs().sum().item(), 0)

    def test_padding_content_cannot_affect_predictions(self):
        head = self.make_head().eval()
        q, e = torch.randn(1, 12), torch.randn(1, 5, 12)
        mask = torch.tensor([[True, True, True, False, False]])
        expected = head(q, e, mask).probabilities
        e[:, 3:] = torch.nan
        actual = head(q, e, mask).probabilities
        torch.testing.assert_close(actual, expected)

    def test_dynamic_cardinality_shares_weights(self):
        head = self.make_head().eval()
        parameter_count = sum(p.numel() for p in head.parameters())
        for count in (1, 255, 256, 1024):
            out = head(torch.randn(1, 12), torch.randn(1, count, 12), torch.ones(1, count, dtype=torch.bool))
            self.assertEqual(out.probabilities.shape, (1, count))
            torch.testing.assert_close(out.probabilities.sum(-1), torch.ones(1))
            self.assertEqual(sum(p.numel() for p in head.parameters()), parameter_count)

    def test_temperature_and_empty_sets(self):
        head = self.make_head()
        q, e, mask = torch.randn(1, 12), torch.randn(1, 3, 12), torch.ones(1, 3, dtype=torch.bool)
        raw = head(q, e, mask)
        scaled = head(q, e, mask, temperature=2.0)
        torch.testing.assert_close(scaled.logits, raw.logits, atol=0, rtol=0)
        torch.testing.assert_close(scaled.probabilities, (raw.logits / 2).softmax(-1))
        torch.testing.assert_close(scaled.log_probs, (raw.logits / 2).log_softmax(-1))
        with self.assertRaises(ValueError):
            head(q, e, mask, temperature=0)
        with self.assertRaises(ValueError):
            head(q, e, ~mask)

    def test_large_option_set_permutation_with_padding(self):
        head = self.make_head().eval()
        generator = torch.Generator().manual_seed(20261014)
        q = torch.randn(2, 12, generator=generator)
        e = torch.randn(2, 1024, 12, generator=generator)
        mask = torch.arange(1024)[None, :] < torch.tensor([1024, 17])[:, None]
        with torch.inference_mode():
            expected = head(q, e, mask).probabilities
            for _ in range(3):
                order = torch.randperm(1024, generator=generator)
                actual = head(q, e[:, order], mask[:, order]).probabilities
                torch.testing.assert_close(actual, expected[:, order], atol=1e-7, rtol=1e-5)
                self.assertEqual(actual[~mask[:, order]].abs().sum().item(), 0)

    def test_single_valid_option_has_zero_nll_at_any_temperature(self):
        head = self.make_head().eval()
        q, e = torch.randn(1, 12), torch.randn(1, 3, 12)
        mask = torch.tensor([[False, True, False]])
        for temperature in (.1, 1, 20):
            result = head(q, e, mask, temperature=temperature)
            torch.testing.assert_close(result.probabilities, torch.tensor([[0., 1., 0.]]), atol=0, rtol=0)
            self.assertEqual(result.log_probs[0, 1].item(), 0)
            self.assertTrue(bool(torch.isneginf(result.log_probs[~mask]).all()))

    def test_probabilities_remain_fp32_under_autocast(self):
        head = self.make_head(set_layers=0)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            out = head(torch.randn(1, 12), torch.randn(1, 3, 12), torch.ones(1, 3, dtype=torch.bool))
        self.assertEqual(out.logits.dtype, torch.float32)
        self.assertEqual(out.log_probs.dtype, torch.float32)
        self.assertEqual(out.probabilities.dtype, torch.float32)

    def test_bf16_features_with_fp32_head_without_autocast(self):
        head = self.make_head()
        q = torch.randn(2, 12, dtype=torch.bfloat16)
        e = torch.randn(2, 3, 12, dtype=torch.bfloat16)
        out = head(q, e, torch.ones(2, 3, dtype=torch.bool))
        self.assertEqual(head.question_projection.weight.dtype, torch.float32)
        self.assertEqual(out.logits.dtype, torch.float32)
        self.assertEqual(out.log_probs.dtype, torch.float32)
        self.assertEqual(out.probabilities.dtype, torch.float32)

    def test_initialization_preserves_backbone_and_rng(self):
        adapter = TinyCausalAdapter()
        old_weights = {name: tensor.clone() for name, tensor in adapter.state_dict().items()}
        rng_state = torch.get_rng_state().clone()
        head = self.make_head(init_seed=37)
        self.assertTrue(torch.equal(rng_state, torch.get_rng_state()))
        other_head = self.make_head(init_seed=37)
        for first, second in zip(head.parameters(), other_head.parameters()):
            torch.testing.assert_close(first, second)
        LightDecisionModel(adapter, head)
        for name, tensor in adapter.state_dict().items():
            torch.testing.assert_close(tensor, old_weights[name])

    def test_legacy_head_weights_load_strictly_without_rescaling(self):
        legacy = self.make_head(init_seed=19, score_init_gain=1.0).eval()
        legacy_config = dict(legacy.config)
        legacy_config.pop("score_init_gain")
        restored = DecisionHead(**legacy_config).eval()
        incompatible = restored.load_state_dict(legacy.state_dict(), strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        q, e = torch.randn(2, 12), torch.randn(2, 3, 12)
        mask = torch.ones(2, 3, dtype=torch.bool)
        torch.testing.assert_close(restored(q, e, mask).probabilities, legacy(q, e, mask).probabilities, atol=0, rtol=0)

    def test_score_gain_must_keep_nonzero_finite_initial_paths(self):
        for gain in (0.0, -0.1, float("nan"), float("inf"), True):
            with self.subTest(gain=gain), self.assertRaises(ValueError):
                self.make_head(score_init_gain=gain)


    def test_question_pooling_is_mean_with_complete_decisions(self):
        q = torch.tensor([[1., 3.], [3., 5.], [9., 11.]])
        e = q * 2
        features = group_branch_features(q, e, torch.tensor([0, 2, 3]), torch.tensor([[True, True], [True, False]]))
        torch.testing.assert_close(features.question_features, torch.tensor([[2., 4.], [9., 11.]]))


if __name__ == "__main__":
    unittest.main()
