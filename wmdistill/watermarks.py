"""Gumbel-max (Aaronson) and inverse-transform (Kuditipudi et al.) watermarks.

Both are expressed through one interface used by generation, detection and training:

    score_vectors(seeds) -> s[..., w]   per-token score of emitting w under the key
    T(Y) = sum_t s_t[Y_t]               the test statistic
    pvalue(T, n)                        under H0 (text independent of the key)

The training-time stat loss only needs score_vectors, so adding a new watermark means
implementing sample / score_vectors / null moments / pvalue.
"""
import math

import torch

from .keys import KeyStream


class Watermark:
    name: str
    null_mean: float
    null_std: float

    def __init__(self, keys: KeyStream):
        self.keys = keys

    @property
    def vocab_size(self) -> int:
        return self.keys.vocab_size

    def sample(self, logits: torch.Tensor, seeds: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def score_vectors(self, seeds: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def token_scores(self, seeds: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        return self.score_vectors(seeds).gather(-1, tokens[..., None].long()).squeeze(-1)

    def standardized_score_vectors(self, seeds: torch.Tensor) -> torch.Tensor:
        """Per-token scores with mean 0 / std 1 under H0, so loss scales are comparable."""
        return (self.score_vectors(seeds) - self.null_mean) / self.null_std

    def pvalue(self, T: float, n: int) -> float:
        raise NotImplementedError


class GumbelWatermark(Watermark):
    """Y_t = argmax_w U_{t,w}^{1/p_w};  s_t(w) = -log(1 - U_{t,w}) ~ Exp(1) under H0."""

    name = "gumbel"
    null_mean = 1.0
    null_std = 1.0

    def sample(self, logits, seeds):
        u = self.keys.vocab_uniforms(seeds)
        gumbel = -torch.log(-torch.log(u))
        # argmax U^{1/p} == argmax (log p + Gumbel(U)); stable when p underflows to 0.
        return torch.argmax(torch.log_softmax(logits.float(), dim=-1) + gumbel, dim=-1)

    def score_vectors(self, seeds):
        return -torch.log1p(-self.keys.vocab_uniforms(seeds))

    def token_scores(self, seeds, tokens):
        return -torch.log1p(-self.keys.token_uniforms(seeds, tokens))

    def pvalue(self, T, n):
        if n == 0:
            return 1.0
        # Sum of n i.i.d. Exp(1) ~ Gamma(n, 1); P(Gamma >= T) = Q(n, T).
        return float(torch.special.gammaincc(torch.tensor(float(n), dtype=torch.float64),
                                             torch.tensor(float(T), dtype=torch.float64)))


class InverseTransformWatermark(Watermark):
    """Permute the vocab by pi_t, emit the token where the permuted CDF crosses u_t.

    eta_t(w) = rank of w under pi_t, normalized to [0, 1];  s_t(w) = (u_t - 1/2)(eta_t(w) - 1/2).
    Under H0 u and eta are independent, mean 0 and variance (1/12) * (V+1)/(12(V-1)).
    """

    name = "inverse"
    null_mean = 0.0

    def __init__(self, keys):
        super().__init__(keys)
        V = keys.vocab_size
        self.null_std = math.sqrt((1 / 12) * (V + 1) / (12 * (V - 1)))

    def sample(self, logits, seeds):
        perm = torch.argsort(self.keys.vocab_hashes(seeds), dim=-1)
        cdf = torch.cumsum(torch.softmax(logits.float(), dim=-1).gather(-1, perm), dim=-1)
        u = self.keys.scalar_uniform(seeds)
        idx = torch.searchsorted(cdf, u[..., None]).clamp(max=self.vocab_size - 1)
        return perm.gather(-1, idx).squeeze(-1)

    def _eta(self, seeds):
        perm = torch.argsort(self.keys.vocab_hashes(seeds), dim=-1)
        rank = torch.empty_like(perm).scatter_(-1, perm, torch.arange(
            self.vocab_size, device=perm.device).expand_as(perm).contiguous())
        return rank.float() / (self.vocab_size - 1)

    def score_vectors(self, seeds):
        u = self.keys.scalar_uniform(seeds)
        return (u[..., None] - 0.5) * (self._eta(seeds) - 0.5)

    def pvalue(self, T, n):
        if n == 0:
            return 1.0
        z = T / (self.null_std * math.sqrt(n))
        return float(torch.special.ndtr(torch.tensor(-z, dtype=torch.float64)))


WATERMARKS = {"gumbel": GumbelWatermark, "inverse": InverseTransformWatermark}


def make_watermark(name: str, cfg: dict, vocab_size: int) -> Watermark:
    w = cfg["watermark"]
    keys = KeyStream(w["key"], vocab_size, w.get("seed_scheme", "context"), w.get("context_width", 2))
    return WATERMARKS[name](keys)
