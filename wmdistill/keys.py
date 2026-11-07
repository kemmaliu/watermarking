"""Keyed pseudorandomness for watermarking.

Every quantity is a deterministic function of (secret key, seed context), computed with
vectorized 32-bit integer hashing so it is bit-identical on CPU / CUDA / MPS and can be
recomputed at training time (key available) and detection time.

MPS (torch 2.2) has no int64 shifts and does int64 division in float32, which silently
corrupts the hash -- so on MPS the key math runs on CPU (see _exact_device).

Seed schemes
------------
- "context": seed_t = H(key, y_{t-h}, ..., y_{t-1})   (Aaronson / Kirchenbauer-style hashing)
- "position": seed_t = H(key, t), t = index within the completion   (Kuditipudi-style key sequence)
"""
import functools

import torch

MASK = 0xFFFFFFFF


def mix(x: torch.Tensor) -> torch.Tensor:
    """Bijective 32-bit integer hash (inputs/outputs in [0, 2^32), int64 holds products exactly)."""
    x = x & MASK
    for _ in range(2):
        x = ((x >> 16) ^ x) * 0x45D9F3B & MASK
    return (x >> 16) ^ x


def _exact_device(fn):
    """Run on CPU when inputs live on MPS, return results on the caller's device."""

    @functools.wraps(fn)
    def wrapper(self, *args):
        dev = next(a.device for a in args if isinstance(a, torch.Tensor))
        if dev.type != "mps":
            return fn(self, *args)
        out = fn(self, *(a.cpu() if isinstance(a, torch.Tensor) else a for a in args))
        return out.to(dev)

    return wrapper


def _to_unit(h: torch.Tensor) -> torch.Tensor:
    """32-bit hash -> float32 uniform strictly inside (0, 1).

    23-bit resolution so every value, including the largest (1 - 2^-24), is exact in fp32;
    with 24 bits the top value rounds to 1.0 and -log(1 - u) becomes inf.
    """
    return ((h >> 9).float() + 0.5) / float(1 << 23)


class KeyStream:
    def __init__(self, key: int, vocab_size: int, scheme: str = "context", context_width: int = 2):
        if scheme not in ("context", "position"):
            raise ValueError(f"unknown seed scheme {scheme!r}")
        self.key = int(key) & MASK
        self.vocab_size = int(vocab_size)
        self.scheme = scheme
        self.h = int(context_width)

    # ---- seeds -------------------------------------------------------------------------
    def _base(self, like: torch.Tensor) -> torch.Tensor:
        return mix(torch.full_like(like, self.key, dtype=torch.int64))

    @_exact_device
    def seeds_at_step(self, input_ids: torch.Tensor, step: int) -> torch.Tensor:
        """Seeds for the next token during generation. input_ids: [B, L] (prompt + generated)."""
        last = input_ids[:, -1]
        if self.scheme == "position":
            return mix(torch.bitwise_xor(self._base(last), torch.full_like(last, step)))
        s = self._base(last)
        for i in range(1, self.h + 1):
            s = mix(torch.bitwise_xor(s, input_ids[:, -i].long() & MASK))
        return s

    @_exact_device
    def seeds_for_sequences(self, ids: torch.Tensor, prompt_lens: torch.Tensor) -> torch.Tensor:
        """seeds[b, j] = seed used when token ids[b, j] was generated. Valid for j >= prompt_len."""
        ids = ids.long()
        B, L = ids.shape
        if self.scheme == "position":
            t = torch.arange(L, device=ids.device)[None, :] - prompt_lens.to(ids.device)[:, None]
            return mix(torch.bitwise_xor(self._base(ids), t.clamp(min=0)))
        s = self._base(ids)
        for i in range(1, self.h + 1):
            prev = torch.nn.functional.pad(ids, (i, 0))[:, :L]  # prev[:, j] = ids[:, j - i]
            s = mix(torch.bitwise_xor(s, prev & MASK))
        return s

    # ---- randomness derived from seeds ----------------------------------------------------
    @_exact_device
    def vocab_hashes(self, seeds: torch.Tensor) -> torch.Tensor:
        """[..., V] distinct 32-bit hashes, one per vocab item."""
        w = torch.arange(self.vocab_size, device=seeds.device, dtype=torch.int64)
        return mix(torch.bitwise_xor(mix(seeds)[..., None], w))

    @_exact_device
    def vocab_uniforms(self, seeds: torch.Tensor) -> torch.Tensor:
        """[..., V] i.i.d.-looking Unif(0,1) — the Gumbel-max key zeta_t."""
        return _to_unit(self.vocab_hashes(seeds))

    @_exact_device
    def token_uniforms(self, seeds: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """vocab_uniforms(seeds)[..., tokens] without materializing the full vocab."""
        return _to_unit(mix(torch.bitwise_xor(mix(seeds), tokens.long())))

    @_exact_device
    def scalar_uniform(self, seeds: torch.Tensor) -> torch.Tensor:
        """One Unif(0,1) per seed — the inverse-transform key u_t (independent of vocab hashes)."""
        return _to_unit(mix(torch.bitwise_xor(mix(seeds), torch.full_like(seeds, MASK))))
