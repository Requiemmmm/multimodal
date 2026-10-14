#!/usr/bin/env python
"""Shared objectives for hard-token suffix search (script 65) and its E0 audit
(script 71).

The maths here is the inlined behaviour of ``scripts/65_diag_hard_gcg.py``,
moved verbatim into one place so the audit and the search cannot drift apart.
Two implementation choices, both behaviour-preserving:

* the CFG image-conditioned / unconditioned branches depend only on the target
  visual prefix, never on the suffix.  They are constants of the search
  variable, so they are computed once per iteration and reused across every
  candidate evaluation instead of twice per candidate;
* the K per-step image policies are normalised as one ``[K, V]`` batch -- the
  same ``log_softmax`` as the K rank-1 calls it replaces.

The content term follows the official three-branch image CFG
(``chameleon/inference/chameleon.py``)::

    logits = uncond + 1.2 * (imgcond - uncond) + 3.0 * (full - imgcond)

with the teacher-forced forward over ``[prompt + EOT + suffix + BOI + target[:-1]]``
playing the role of the ``full_conditioned`` branch.  The official ``imgcond``
filters the prompt down to ``[BOS, BOI]``; this module prefills ``[BOI]`` only,
so the 1.2 term is a suffix-independent constant rather than an exact
cancellation.  It shifts the objective by a constant (recorded in the audit
output) and does not bias which candidate wins.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


d08 = _load("distill08", str(PROJECT_ROOT / "scripts/08_distill_anole_soft_route.py"))
p12 = _load("ladder12", str(PROJECT_ROOT / "scripts/12_anole_gateway_patch_ladder.py"))
p14 = _load("adam14", str(PROJECT_ROOT / "scripts/14_adam_soft_route.py"))

from morph.decoding.processed_policy import (  # noqa: E402
    apply_repetition_penalty,
    masked_logprobs,
)


def batched_masked_logprobs(raw_logits, legal_mask):
    """Row-wise ``masked_logprobs`` over a ``[N, V]`` batch.

    Identical arithmetic to calling ``processed_policy.masked_logprobs`` once
    per row (which is rank-1 only), just normalised in one kernel.
    """
    import torch

    return torch.log_softmax(
        raw_logits.masked_fill(~legal_mask, float("-inf")), dim=-1
    )


class CFGBranchCache:
    """Suffix-independent CFG branch logits for one fixed visual prefix.

    ``imgcond`` is prefilled with ``[BOI]`` and ``uncond`` with ``[BOS, BOI]``,
    matching the inlined version in script 65.  ``rows[k]`` is the branch's
    logits conditioned on ``prefix[:k]`` -- exactly the tensor the inlined loop
    reads before its ``advance(prefix[k])``.

    ``prefix`` is normally the teacher's ``target_visual``, but may be any
    visual sequence of at least the scored length (e.g. the tokens the model
    itself sampled, for the exposure-bias diagnostic).  The branches never see
    the suffix, which is why one cache serves every candidate.
    """

    def __init__(self, model, transformer, token_sets, prefix, extra_length=8):
        suffix_length = len(prefix)
        self.rows_imgcond = []
        self.rows_uncond = []
        imgcond = p12.Branch(model, transformer, [token_sets.boi_id],
                             suffix_length + extra_length)
        uncond = p12.Branch(model, transformer,
                            [token_sets.bos_id, token_sets.boi_id],
                            suffix_length + extra_length)
        for k in range(suffix_length):
            self.rows_imgcond.append(imgcond.logits)
            self.rows_uncond.append(uncond.logits)
            imgcond.advance(prefix[k])
            uncond.advance(prefix[k])


class HardObjectiveSpace:
    """Route (BOI-CE) and content (forced-BOI visual contrast) objectives."""

    def __init__(self, model, transformer, token_sets, prompt_ids, support,
                 image_mask=None):
        self.model = model
        self.transformer = transformer
        self.token_sets = token_sets
        self.prompt_ids = prompt_ids
        self.support = support
        self.image_mask = (image_mask if image_mask is not None
                           else p12.build_mask(token_sets, token_sets.image_token_ids))
        self.past = None  # lazily built prompt-history tensor

    # ---------------------------------------------------------------- helpers

    def _history(self):
        import torch

        if self.past is None:
            self.past = torch.tensor(self.prompt_ids, dtype=torch.long,
                                     device="cuda")
        return self.past

    def as_suffix(self, suffix):
        """Accept token ids (detached) or a 2-D embedding tensor (live)."""
        import torch

        if isinstance(suffix, torch.Tensor) and suffix.dim() == 2:
            return suffix
        ids = torch.as_tensor(list(suffix), dtype=torch.long, device="cuda")
        return self.model.tok_embeddings(ids).detach().float()

    def _forward(self, suffix_emb, continuation=()):
        embeds = d08.soft_input_embeddings(
            self.model, self.prompt_ids[:-1], self.token_sets.eot_id,
            suffix_emb, continuation)
        logits = p14.feasibility.exact_forward(
            self.model, self.transformer, embeds)
        first_prediction = len(self.prompt_ids) + suffix_emb.shape[0] - 1
        return logits, first_prediction

    def text_policy(self, row):
        return masked_logprobs(
            apply_repetition_penalty(row, self._history(), penalty=1.2),
            self.support)

    # ------------------------------------------------------------ route term

    def boi_ce(self, suffix):
        """Processed BOI cross-entropy at the first free generation position."""
        logits, position = self._forward(self.as_suffix(suffix))
        policy = self.text_policy(logits[position])
        return -policy[self.token_sets.boi_id]

    def boi_breakdown(self, suffix, top_p=0.9):
        """BOI log-prob, rank and top-p sampling share (script 70 semantics)."""
        import torch

        logits, position = self._forward(self.as_suffix(suffix))
        policy = self.text_policy(logits[position])
        boi = self.token_sets.boi_id
        sorted_lp, sorted_idx = policy.sort(descending=True)
        keep = (sorted_lp.exp().cumsum(0) <= top_p)
        keep[0] = True
        share = float(policy[boi].exp() / policy[sorted_idx[keep]].exp().sum())
        rank = int((policy > policy[boi]).sum().item()) + 1
        return {
            "boi_lp": float(policy[boi]),
            "boi_rank": rank,
            "boi_best_lp": float(sorted_lp[0]),
            "boi_share_top_p": share,
            "top_p": top_p,
        }

    # ---------------------------------------------------------- content term

    def content(self, suffix, target_visual, victim_visual, cache=None,
                mode="margin", continuation=None, target_policy=None):
        """Content objective over the forced-BOI prefix.

        ``mode="margin"`` (default) is the original script-65 term: the negative
        mean target-vs-victim log-probability margin.  ``mode="ce"`` drops the
        victim reference and maximises the target tokens' mass directly.

        The distinction matters because of what the campaign already learned on
        the route axis: a converged *margin* there coexisted with a 2.2% sampling
        share and 0/16 behaviour, while the *cross-entropy* objective reached
        43% mass and 51.25% behaviour.  The E0 audit found the same shape on the
        content axis -- the init suffix already carries +2 nats of margin at 7 of
        8 visual positions and still produces no apple -- so margin and mass are
        separated here too.

        Returns ``(loss, per_step)``; ``per_step`` is per-position
        ``log p(target) - log p(victim)`` for "margin" (positive = target
        favoured, the sign convention the CLI reports) and per-position
        ``log p(target)`` for "ce".
        """
        import torch

        K = len(target_visual)
        if K == 0:
            zero = torch.zeros((), device="cuda")
            return zero, []
        # ``continuation`` is the forced context after the suffix; it defaults to
        # the teacher's own target prefix, and can be set to the tokens the model
        # sampled itself (exposure-bias diagnostic).  Position k is always scored
        # against whatever context precedes index k, and the CFG branches are
        # advanced with that same context.
        if continuation is None:
            continuation = [self.token_sets.boi_id] + list(target_visual[:-1])
            branch_prefix = list(target_visual)
        else:
            continuation = list(continuation)
            branch_prefix = continuation[1:]  # drop BOI
        if cache is None:
            cache = CFGBranchCache(self.model, self.transformer, self.token_sets,
                                   branch_prefix)
        if len(cache.rows_imgcond) < K:
            raise ValueError(
                f"CFG branch cache has {len(cache.rows_imgcond)} rows but {K} "
                "positions are scored; the cache prefix must be at least K long"
            )
        logits, first_prediction = self._forward(self.as_suffix(suffix),
                                                 continuation)
        rows = torch.stack(
            [logits[first_prediction + 1 + k] for k in range(K)])
        img = torch.stack(cache.rows_imgcond[:K])
        unc = torch.stack(cache.rows_uncond[:K])
        mixed = unc + 1.2 * (img - unc) + 3.0 * (rows - img)
        policy = batched_masked_logprobs(mixed, self.image_mask)
        targets = torch.tensor(list(target_visual), dtype=torch.long,
                               device="cuda")
        if mode == "distill":
            # Soft cross-entropy against the model's own apple-drawing policy
            # (script 75).  Unlike "margin"/"ce", this never names a target
            # token: it matches the whole distribution, so all five verified
            # first-token variants count equally and filler tokens are weighted
            # by how much mass the apple policy actually puts on them.
            if target_policy is None:
                raise ValueError("mode='distill' requires target_policy")
            tgt = target_policy[:K].to(policy.dtype)
            # ``policy`` is -inf off the image support and ``tgt`` is 0 there, so
            # the raw product is 0 * -inf = NaN.  Zero out the off-support
            # entries first; they contribute nothing either way.
            safe = torch.nan_to_num(policy, neginf=0.0, posinf=0.0)
            agreement = (tgt * safe).sum(-1)  # [K]; == -CE, higher is better
            if not torch.isfinite(agreement).all():
                raise FloatingPointError(
                    "distillation agreement is not finite; check that the "
                    "target policy is normalised over the image support")
            return -agreement.mean(), agreement
        target_lp = policy.gather(1, targets[:, None])[:, 0]
        if mode == "ce":
            return -target_lp.mean(), target_lp
        victims = torch.tensor(list(victim_visual), dtype=torch.long,
                               device="cuda")
        deltas = target_lp - policy.gather(1, victims[:, None])[:, 0]
        return -deltas.mean(), deltas

    # ------------------------------------------------------------ joint term

    def joint(self, suffix, target_visual, victim_visual, content_lambda=1.0,
              cache=None, content_mode="margin"):
        """``boi_ce + lambda * content`` with both parts returned for logging."""
        ce = self.boi_ce(suffix)
        content_loss, deltas = self.content(suffix, target_visual, victim_visual,
                                            cache, mode=content_mode)
        return ce + content_lambda * content_loss, {
            "boi_ce": ce,
            "content_loss": content_loss,
            "content_margins": deltas,
        }
