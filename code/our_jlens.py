#!/usr/bin/env python3
"""Our Jacobian-lens harness, built FROM THE PAPER (Niz build-from-papers rule).

Source of method: "Verbalizable Representations Form a Global Workspace in Language
Models" (Anthropic, transformer-circuits.pub/2026/workspace) + the method spec
Deron extracted into cc-jlens-plan-card C03/card C01 (Step 0a). The cloned
anthropics/jacobian-lens repo is used ONLY as a read-only reference + TEST ORACLE
(our J_l and readouts must agree with it on a small model before we trust ours on
Swift-1.5). We do NOT import or adapt the repo's code — this is our own
implementation of the paper's method.

STATUS: UNVERIFIED. the review host has no torch (no pypi egress), so this was authored but
NOT run here. It must be (a) syntax/py_compiled, (b) oracle-validated on a torch
box against anthropics/jacobian-lens on a tiny HF model (e.g. gpt2), before use.

METHOD (fitted, averaged Jacobian; lens_l(h) = unembed(J_l @ h)):
  J_l = mean over prompts and valid source positions of  d h_final / d h_l,
  where the target cotangent at an output residual dim is injected at EVERY valid
  target position at once, so the grad at source pos p is sum_{p'>=p} dh_final[p']/dh_l[p].
  Valid positions skip the first SKIP_FIRST (attention sinks) and the last (no target).
  Cost: one forward (retain graph) + ceil(d_model/dim_batch) backward passes per prompt.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Sequence

import torch
import torch.nn as nn

SKIP_FIRST_N_POSITIONS = 16


@dataclass
class Layout:
    """Where the residual stack / final norm / unembed live inside an HF model.
    `path` is a dotted attribute path from the top model to the text decoder
    (handles multimodal wrappers via config.get_text_config())."""
    layers_attr: str = "layers"
    norm_attr: str = "norm"
    embed_attr: str = "embed_tokens"


def _resolve_text_decoder(hf_model):
    """Resolve the residual stack BY ROLE, not by attribute name, so the same code
    serves gpt2 (transformer.h / ln_f / wte), Llama/Qwen (model.layers / norm /
    embed_tokens), and the Qwen3_5 multimodal wrapper. Returns
    (decoder_module, layers_modulelist, final_norm, text_config)."""
    text_config = hf_model.config.get_text_config()
    n_layers = int(text_config.num_hidden_layers)
    # (1) the decoder blocks = the ModuleList whose length == num_hidden_layers
    layers_name = layers_ml = None
    for name, mod in hf_model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) == n_layers:
            layers_name, layers_ml = name, mod
            break
    if layers_ml is None:
        raise ValueError(f"no nn.ModuleList of length num_hidden_layers={n_layers} found")
    # (2) the module that OWNS the blocks (its parent) also owns the final norm
    parent_path = layers_name.rsplit(".", 1)[0] if "." in layers_name else ""
    dec = hf_model.get_submodule(parent_path) if parent_path else hf_model
    # (3) final norm = a norm-like direct child of dec that isn't the blocks list
    final_norm = None
    for cn, cm in dec.named_children():
        if cm is layers_ml or isinstance(cm, nn.ModuleList):
            continue
        if ("norm" in cn.lower() or "ln_f" in cn.lower()
                or cm.__class__.__name__.lower().endswith("norm")):
            final_norm = cm  # keep the last match (the final norm, after the blocks)
    if final_norm is None:
        raise ValueError(f"could not locate a final norm on {parent_path or '<root>'}")
    return dec, layers_ml, final_norm, text_config


class OurJacobianLens:
    def __init__(self, hf_model, tokenizer, *, add_bos: bool = True):
        self.model = hf_model
        self.tok = tokenizer
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)   # we differentiate wrt activations, not weights
        # BOS / attention-sink is a METHOD CHOICE and must be EXPLICIT + HERMETIC (Claude P001).
        # The reference repo's from_hf(force_bos=True) MUTATES the shared tokenizer's add_bos_token;
        # if our fit merely read the tokenizer's state, our result would silently depend on whether
        # that ran before us (gpt2: 0.9994 with the inherited BOS, 0.978 without). So we do NOT read
        # or mutate the tokenizer's add_bos_token: we encode with add_special_tokens=False and
        # prepend the BOS ourselves iff `add_bos` AND the tokenizer HAS a bos id. Default True =
        # the paper/repo default (raw-text prompts need an attention-sink BOS). Recorded below.
        self.add_bos = bool(add_bos)
        self.bos_id = getattr(tokenizer, "bos_token_id", None)
        self.bos_applied = self.add_bos and (self.bos_id is not None)  # the recorded, effective choice
        self.decoder, self.layers, self.final_norm, self.text_config = _resolve_text_decoder(hf_model)
        self.n_layers = int(self.text_config.num_hidden_layers)
        self.d_model = int(self.text_config.hidden_size)
        self.embed = hf_model.get_input_embeddings()      # role, not name
        self.lm_head = hf_model.get_output_embeddings()   # unembedding (tied or not)
        if len(self.layers) != self.n_layers:
            raise ValueError(f"n_layers={self.n_layers} but found {len(self.layers)} blocks")

    def _encode(self, prompt: str, max_seq_len: int):
        """Hermetic tokenization: explicit BOS control, independent of tokenizer.add_bos_token
        state (so a prior reference fit that mutated the shared tokenizer cannot change our ids)."""
        budget = max_seq_len - (1 if self.bos_applied else 0)
        enc = self.tok(prompt, return_tensors="pt", truncation=True, max_length=budget,
                       add_special_tokens=False).input_ids.to(self.model.device)
        if self.bos_applied:
            bos = torch.full((enc.shape[0], 1), int(self.bos_id), dtype=enc.dtype, device=enc.device)
            enc = torch.cat([bos, enc], dim=1)
        return enc

    # ---- residual-stream capture via forward hooks (the captured tensor roots autograd) ----
    def _capture_residuals(self, source_layers, target_layer, grad_root_layer=None):
        """Hooks for the VJP. The GRAD ROOT is where backward stops:
        - grad_root_layer is None (full fit): root at the EMBEDDING output, so the whole stack
          is in the backward (bug-2 fix: frozen params + token ids still give grad_fn).
        - grad_root_layer=L (shape-(c) SPLIT fit): root at layer L's OUTPUT (= h_L). Backward
          stops at h_L, so layers below L are NOT in the backward — exactly shape (c), whether
          the layers below are really absent (Swift resident-upper-stack) or merely detached
          here (full-model 0.8B validation). L must be the SHALLOWEST source layer; J_L is then
          wrt the root itself, and deeper source layers wrt their retained outputs.
        Source/target hooks retain_grad + stash each OUTPUT residual. Returns (handles, store)."""
        store: dict[int, torch.Tensor] = {}
        handles = []

        def _root_from(out):
            h = (out[0] if isinstance(out, tuple) else out).detach().requires_grad_(True)
            return h, ((h,) + tuple(out[1:])) if isinstance(out, tuple) else h

        if grad_root_layer is None:
            def embed_hook(module, inp, out):
                h, repl = _root_from(out)
                store["__root__"] = h
                return repl
            handles.append(self.embed.register_forward_hook(embed_hook))
        else:
            def root_hook(module, inp, out):
                h, repl = _root_from(out)
                store["__root__"] = h
                store[grad_root_layer] = h   # the root IS this source layer's residual
                return repl
            handles.append(self.layers[grad_root_layer].register_forward_hook(root_hook))

        def mk(idx):
            def hook(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                h.retain_grad()          # grad wrt this intermediate residual
                store[idx] = h
            return hook

        wanted = set(source_layers) | {target_layer}
        for idx in sorted(wanted):
            if idx == grad_root_layer:
                continue                 # its residual is the root, stashed by root_hook
            handles.append(self.layers[idx].register_forward_hook(mk(idx)))
        return handles, store

    @torch.enable_grad()
    def fit_prompt(self, prompt: str, source_layers: Sequence[int], *,
                   target_layer: int | None = None, dim_batch: int = 8,
                   max_seq_len: int = 128, skip_first: int = SKIP_FIRST_N_POSITIONS,
                   grad_root_layer: int | None = None):
        """Estimate per-layer J_l for ONE prompt. Returns {l: [d_model,d_model]}, n_valid.

        grad_root_layer: None = full fit (embedding grad-root). An int L = shape-(c) SPLIT
        fit: backward is rooted at h_L, so layers below L are excluded (resident-upper-stack).
        L MUST equal the shallowest source layer (every source must be >= the root, or its
        residual is below the root and gets no grad). J_L is read from the root leaf's .grad;
        on 0.8B the split J_l must match the full-fit J_l (same derivative, different root)."""
        if target_layer is None:
            target_layer = self.n_layers - 1
        source_layers = [l for l in source_layers if l < target_layer]
        if grad_root_layer is not None:
            if any(l < grad_root_layer for l in source_layers):
                raise ValueError(f"grad_root_layer={grad_root_layer} must be <= every source "
                                 f"layer; got sources {sorted(source_layers)} (layers below the "
                                 f"root are not in the backward)")
        ids = self._encode(prompt, max_seq_len)   # hermetic, explicit BOS (self.bos_applied)
        seq_len = ids.shape[1]
        if seq_len <= skip_first + 1:
            print(f"[fit] SKIP prompt too short: {seq_len} <= {skip_first+1} (bug-#5 skip-and-log, post P020)", flush=True)
            return {}, 0
        # valid target positions: [skip_first, seq_len-1)
        valid = torch.zeros(seq_len, dtype=torch.bool); valid[skip_first:seq_len-1] = True
        n_valid = int(valid.sum())

        # replicate prompt dim_batch times on batch axis; one forward, retain graph
        batch_ids = ids.expand(dim_batch, -1)
        handles, store = self._capture_residuals(source_layers, target_layer,
                                                 grad_root_layer=grad_root_layer)
        try:
            # input_ids= kwarg (robust for a multimodal ConditionalGeneration wrapper:
            # text-only path, no pixel_values -> the vision tower stays inert)
            self.model(input_ids=batch_ids, use_cache=False)   # forward; hooks fill store
            h_tgt = store[target_layer]               # [dim_batch, seq, d_model]
            Js = {l: torch.zeros(self.d_model, self.d_model) for l in source_layers}
            n_back = math.ceil(self.d_model / dim_batch)
            for blk in range(n_back):
                dims = list(range(blk*dim_batch, min((blk+1)*dim_batch, self.d_model)))
                # cotangent: for batch element b -> one-hot at output-dim dims[b], at every valid target pos
                cot = torch.zeros_like(h_tgt)
                for b, d in enumerate(dims):
                    cot[b, valid, d] = 1.0
                for l in source_layers:
                    if store[l].grad is not None:
                        store[l].grad.zero_()
                torch.autograd.backward(h_tgt, cot, retain_graph=(blk < n_back-1))
                for l in source_layers:
                    g = store[l].grad                 # [dim_batch, seq, d_model]
                    # batch element b carries the cotangent for output residual dim dims[b],
                    # already summed over target positions by the all-valid-pos cotangent.
                    # J_l row for that output dim = mean over valid SOURCE positions of g.
                    gm = g[:, valid, :].mean(dim=1)   # [dim_batch, d_model]
                    for b, d in enumerate(dims):
                        Js[l][d, :] = gm[b].float()
            return Js, n_valid
        finally:
            for hd in handles:
                hd.remove()

    def fit(self, prompts: Sequence[str], source_layers: Sequence[int], **kw):
        """Average J_l over prompts (n_valid-weighted). Returns {l: [d_model,d_model]}."""
        acc = {l: torch.zeros(self.d_model, self.d_model) for l in source_layers}
        tot = 0
        for pr in prompts:
            Js, n_valid = self.fit_prompt(pr, source_layers, **kw)
            for l in source_layers:
                acc[l] += Js[l] * n_valid
            tot += n_valid
        return {l: acc[l] / max(tot, 1) for l in source_layers}

    @torch.no_grad()
    def transport(self, J_l: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """J_l @ h -> final layer-norm -> unembed -> logits. h: [..., d_model]."""
        transported = h.float() @ J_l.float().T      # into final-layer residual basis
        normed = self.final_norm(transported.to(h.dtype))
        return self.lm_head(normed)                   # [..., vocab] verbalizable dist
