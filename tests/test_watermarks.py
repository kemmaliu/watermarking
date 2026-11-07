import numpy as np
import pytest
import torch

from wmdistill.generation import detect
from wmdistill.keys import KeyStream
from wmdistill.train import stat_loss
from wmdistill.watermarks import GumbelWatermark, InverseTransformWatermark

V = 50
WMS = [GumbelWatermark, InverseTransformWatermark]


def make(cls, scheme="context"):
    return cls(KeyStream(key=1234, vocab_size=V, scheme=scheme, context_width=2))


def test_uniforms_in_open_interval_and_roughly_uniform():
    k = KeyStream(7, 1000)
    u = k.vocab_uniforms(torch.arange(200))
    assert u.min() > 0 and u.max() < 1
    assert abs(u.mean().item() - 0.5) < 0.01
    assert torch.allclose(k.token_uniforms(torch.arange(200)[:, None], torch.tensor([[3, 999]])),
                          u[:, [3, 999]])


def test_extreme_uniforms_stay_below_one():
    from wmdistill.keys import _to_unit

    u = _to_unit(torch.tensor([0, 0xFFFFFFFF]))
    assert u[0] > 0 and u[1] < 1 and torch.isfinite(-torch.log1p(-u)).all()


def test_generation_and_training_seeds_agree():
    for scheme in ("context", "position"):
        k = KeyStream(7, V, scheme=scheme, context_width=3)
        ids = torch.randint(0, V, (2, 12))
        P = 5
        full = k.seeds_for_sequences(ids, torch.tensor([P, P]))
        for j in range(P, 12):
            assert torch.equal(k.seeds_at_step(ids[:, :j], j - P), full[:, j])


@pytest.mark.parametrize("cls", WMS)
def test_sampling_is_distortion_free(cls):
    """Marginal over keys of the watermarked token equals p."""
    wm = make(cls)
    p = torch.softmax(torch.randn(V) * 2, -1)
    n = 40000
    toks = wm.sample(p.log().expand(n, V), torch.arange(n))
    freq = torch.bincount(toks, minlength=V).float() / n
    assert (freq - p).abs().max() < 0.015


@pytest.mark.parametrize("cls", WMS)
def test_token_scores_match_score_vectors(cls):
    wm = make(cls)
    seeds, toks = torch.arange(30), torch.randint(0, V, (30,))
    assert torch.allclose(wm.token_scores(seeds, toks),
                          wm.score_vectors(seeds).gather(-1, toks[:, None]).squeeze(-1), atol=1e-6)


@pytest.mark.parametrize("cls", WMS)
def test_null_pvalues_uniform_and_watermark_detected(cls):
    # Calibration under the context scheme (with the position scheme every null sample shares one
    # key sequence, so the null is conditional on it -- negligible at real vocab sizes, not at V=50).
    wm = make(cls)
    rng = np.random.default_rng(0)
    null_p, wm_p = [], []
    for _ in range(2000):
        prompt = [int(x) for x in rng.integers(0, V, 3)]
        null_p.append(detect(wm, prompt, rng.integers(0, V, 40).tolist())["pvalue"])
    logits = torch.randn(V) * 0.5  # high entropy -> strong signal
    for _ in range(20):
        prompt, comp = [int(x) for x in rng.integers(0, V, 3)], []
        ids = torch.tensor([prompt])
        for t in range(40):
            tok = wm.sample(logits[None], wm.keys.seeds_at_step(ids, t))
            ids = torch.cat([ids, tok[:, None]], 1)
            comp.append(int(tok))
        wm_p.append(detect(wm, prompt, comp)["pvalue"])
    assert 0.035 < np.mean(np.array(null_p) < 0.05) < 0.065
    assert np.median(wm_p) < 1e-4 and max(wm_p) < 0.05


@pytest.mark.parametrize("cls", WMS)
def test_stat_loss_gradient_pushes_toward_high_score_tokens(cls):
    wm = make(cls)
    seeds = torch.arange(64)
    logits = torch.zeros(64, V, requires_grad=True)
    targets = torch.zeros(64, dtype=torch.long)
    loss, _ = stat_loss(logits, seeds, targets, torch.zeros(64, dtype=torch.long), 1, wm, mode="max", chunk=16)
    loss.backward()
    z = wm.standardized_score_vectors(seeds)
    # descending the loss raises logits of high-score tokens
    corr = torch.corrcoef(torch.stack([(-logits.grad).flatten(), z.flatten()]))[0, 1]
    assert corr > 0.9


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
@pytest.mark.parametrize("cls", WMS)
def test_keys_bit_identical_on_mps(cls):
    """Regression: MPS int64 division runs in fp32 and silently changed the key."""
    wm = cls(KeyStream(15485863, 50257))
    seeds = torch.arange(32) * 7919 + 4_000_000_000
    logits = torch.randn(32, 50257)
    toks = torch.randint(0, 50257, (32,))
    m = torch.device("mps")
    assert torch.equal(wm.keys.vocab_hashes(seeds), wm.keys.vocab_hashes(seeds.to(m)).cpu())
    assert torch.equal(wm.sample(logits, seeds), wm.sample(logits.to(m), seeds.to(m)).cpu())
    assert torch.allclose(wm.score_vectors(seeds), wm.score_vectors(seeds.to(m)).cpu())
    assert torch.allclose(wm.token_scores(seeds, toks), wm.token_scores(seeds.to(m), toks.to(m)).cpu())
