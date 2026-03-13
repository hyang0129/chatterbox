"""
Unit tests for _blend_conditionals — pure-logic blending with no HTTP or model.
"""
import torch

from tests.conftest import _Conditionals, _T3Cond


def _make_fake_conditionals():
    """Build a minimal Conditionals-like object using the test-conftest stubs."""
    t3 = _T3Cond(
        speaker_emb=torch.randn(1, 256),
        cond_prompt_speech_tokens=torch.randint(0, 6000, (1, 375)),
        emotion_adv=torch.tensor([[[0.0]]]),
    )
    gen = {
        "embedding": torch.randn(1, 192),
        "prompt_token": torch.randint(0, 6000, (1, 100)),
        "prompt_token_len": torch.tensor([100]),
        "prompt_feat": torch.randn(1, 200, 80),
        "prompt_feat_len": None,
    }
    return _Conditionals(t3, gen)


class TestBlendConditionals:
    """Pure-logic tests for the blending function."""

    def test_texture_mix_0_equals_voice_a(self) -> None:
        from app.main import _blend_conditionals

        conds_a = _make_fake_conditionals()
        conds_b = _make_fake_conditionals()
        result = _blend_conditionals(conds_a, conds_b, 0)

        expected = conds_a.t3.speaker_emb.float()
        expected = expected / expected.norm(p=2, dim=-1, keepdim=True)
        torch.testing.assert_close(result.t3.speaker_emb.float(), expected)

    def test_texture_mix_100_equals_voice_b(self) -> None:
        from app.main import _blend_conditionals

        conds_a = _make_fake_conditionals()
        conds_b = _make_fake_conditionals()
        result = _blend_conditionals(conds_a, conds_b, 100)

        expected = conds_b.t3.speaker_emb.float()
        expected = expected / expected.norm(p=2, dim=-1, keepdim=True)
        torch.testing.assert_close(result.t3.speaker_emb.float(), expected)

    def test_blended_embeddings_are_l2_normalised(self) -> None:
        from app.main import _blend_conditionals

        conds_a = _make_fake_conditionals()
        conds_b = _make_fake_conditionals()
        result = _blend_conditionals(conds_a, conds_b, 50)

        t3_norm = result.t3.speaker_emb.float().norm(p=2).item()
        assert abs(t3_norm - 1.0) < 1e-5

        xvec_norm = result.gen["embedding"].float().norm(p=2).item()
        assert abs(xvec_norm - 1.0) < 1e-5

    def test_rhythm_always_from_voice_a(self) -> None:
        """Rhythm (speech tokens, prompt_token) always comes from voice A."""
        from app.main import _blend_conditionals

        conds_a = _make_fake_conditionals()
        conds_b = _make_fake_conditionals()
        result = _blend_conditionals(conds_a, conds_b, 50)

        torch.testing.assert_close(
            result.t3.cond_prompt_speech_tokens,
            conds_a.t3.cond_prompt_speech_tokens,
        )
        torch.testing.assert_close(
            result.gen["prompt_token"],
            conds_a.gen["prompt_token"],
        )

    def test_midpoint_is_different_from_both_sources(self) -> None:
        from app.main import _blend_conditionals

        conds_a = _make_fake_conditionals()
        conds_b = _make_fake_conditionals()
        result = _blend_conditionals(conds_a, conds_b, 50)

        norm_a = conds_a.t3.speaker_emb.float()
        norm_a = norm_a / norm_a.norm(p=2, dim=-1, keepdim=True)
        norm_b = conds_b.t3.speaker_emb.float()
        norm_b = norm_b / norm_b.norm(p=2, dim=-1, keepdim=True)
        blended = result.t3.speaker_emb.float()

        assert not torch.allclose(blended, norm_a, atol=1e-4)
        assert not torch.allclose(blended, norm_b, atol=1e-4)
