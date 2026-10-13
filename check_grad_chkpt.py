#!/usr/bin/env python3
"""
check_grad_ckpt.py (v2) - check the gradient-checkpointing patch for MoE-FFD on your own GPU.

Run it from the MoE-FFD folder, after adding the three edits to ViT_MoE.py:

    python check_grad_ckpt.py

It answers three questions:
  0. Is the checkpointed path actually running? (v1 did not check this, so its PASS
     could be empty: an unpatched loop also gives identical results with the flag on and off.)
  1. Does the patched model give the same loss and gradients as the plain path?
     (compared at batch 8, with the same random seed for the noisy gating)
  2. How much GPU memory and time does one training step need, with the patch on and off?

No dataset and no pretrained weights are needed; inputs are random.
"""

import time

import torch
import torch.nn.functional as F

import ViT_MoE
from ViT_MoE import vit_base_patch16_224_in21k

GB = 2 ** 30
CALLS = {"n": 0}


def count_checkpoint_calls():
    """Wrap the checkpoint function ViT_MoE.py uses, so we can see how often it is called."""
    original = getattr(ViT_MoE, "checkpoint", None)
    if original is None:
        return False

    def counting(*args, **kwargs):
        CALLS["n"] += 1
        return original(*args, **kwargs)

    ViT_MoE.checkpoint = counting
    return True


def step(model, x, y, ckpt, seed=123, keep_grads=False):
    """One forward + backward pass."""
    model.grad_checkpointing = ckpt
    model.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    CALLS["n"] = 0
    t0 = time.time()
    out, moe_loss = model(x)
    loss = F.cross_entropy(out, y) + 1 * moe_loss          # same loss as train.py
    torch.cuda.synchronize()
    t1 = time.time()
    held_after_forward = torch.cuda.memory_allocated() / GB
    loss.backward()
    torch.cuda.synchronize()
    t2 = time.time()
    result = {
        "loss": loss.item(),
        "calls": CALLS["n"],
        "held": held_after_forward,                          # still allocated once forward is done
        "peak": torch.cuda.max_memory_allocated() / GB,      # highest point of the whole step
        "t_fwd": t1 - t0,
        "t_bwd": t2 - t1,
    }
    if keep_grads:
        result["grads"] = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    return result


def grad_diff(a, b):
    """Largest absolute gradient difference, relative to the largest gradient value."""
    if a.keys() != b.keys():
        return float("inf")
    scale = max(float(g.abs().max()) for g in a.values()) + 1e-12
    return max(float((a[n] - b[n]).abs().max()) for n in a) / scale


def batch(n):
    g = torch.Generator().manual_seed(n)
    return torch.randn(n, 3, 224, 224, generator=g).cuda(), torch.randint(0, 2, (n,), generator=g).cuda()


def main():
    assert torch.cuda.is_available(), "this check needs the GPU"
    torch.backends.cudnn.deterministic = True                # as in train.py
    total = torch.cuda.get_device_properties(0).total_memory / GB
    print(f"torch {torch.__version__}, GPU memory {total:.1f} GB")

    has_import = count_checkpoint_calls()
    model = vit_base_patch16_224_in21k(pretrained=False, num_classes=2).cuda()
    has_flag = hasattr(model, "grad_checkpointing")
    model.train()
    n_blocks = len(model.blocks)

    # Move the trainable weights off their initial values so every gate, LoRA and
    # adapter weight receives a real gradient (several start at exactly zero).
    torch.manual_seed(0)
    with torch.no_grad():
        for _, p in model.named_parameters():
            if p.requires_grad:
                p.add_(0.02 * torch.randn_like(p))

    # ---- 0. is the checkpointed path running? ---------------------------------------
    x, y = batch(8)
    step(model, x, y, ckpt=False)                            # warm-up
    off = step(model, x, y, ckpt=False, keep_grads=True)
    on = step(model, x, y, ckpt=True, keep_grads=True)
    print("\n0. Is the patch active?")
    print(f"   edit 1, import in ViT_MoE.py           : {'found' if has_import else 'MISSING'}")
    print(f"   edit 2, grad_checkpointing attribute   : {'found' if has_flag else 'MISSING'}")
    print(f"   edit 3, blocks run through checkpoint  : {on['calls']} with the flag on (expected {n_blocks}), "
          f"{off['calls']} with it off (expected 0)")
    active = has_import and has_flag and on["calls"] == n_blocks and off["calls"] == 0
    if not active:
        print("   RESULT: NOT ACTIVE - the loop in forward_features is not calling checkpoint().")
        print("   It should read exactly:")
        print("           for block in self.blocks:")
        print("               if self.grad_checkpointing and torch.is_grad_enabled():")
        print("                   x,cur_lora_loss,cur_adapter_loss = checkpoint(block, x, use_reentrant=False)")
        print("               else:")
        print("                   x,cur_lora_loss,cur_adapter_loss = block(x)")
        print("   Parts 1 and 2 are only meaningful once this says ACTIVE.")
    else:
        print("   RESULT: ACTIVE")

    # ---- 1. same loss and gradients? ------------------------------------------------
    again = step(model, x, y, ckpt=False, keep_grads=True)   # repeat of the plain path: the noise floor
    other = step(model, x, y, ckpt=False, seed=124)
    floor_loss, floor_grad = abs(off["loss"] - again["loss"]), grad_diff(off["grads"], again["grads"])
    ckpt_loss, ckpt_grad = abs(off["loss"] - on["loss"]), grad_diff(off["grads"], on["grads"])
    print("\n1. Equivalence at batch 8")
    print(f"   loss, plain path            : {off['loss']:.6f}")
    print(f"   loss, checkpointed path     : {on['loss']:.6f}")
    print(f"   loss with a different seed  : {other['loss']:.6f}   (differs, so the gating noise is active)")
    print(f"   parameters with a gradient  : {len(off['grads'])} plain, {len(on['grads'])} checkpointed")
    print(f"   difference, plain vs plain  : loss {floor_loss:.2e}, gradients {floor_grad:.2e}   (noise floor)")
    print(f"   difference, plain vs ckpt   : loss {ckpt_loss:.2e}, gradients {ckpt_grad:.2e}")
    same = (len(off["grads"]) == len(on["grads"])
            and ckpt_loss <= max(1e-5 * abs(off["loss"]), 10 * floor_loss)
            and ckpt_grad <= max(1e-3, 10 * floor_grad))
    if not active:
        print("   RESULT: NOT TESTED - both runs used the plain path")
    else:
        print("   RESULT:", "PASS - same loss and gradients" if same else "FAIL - do not train with the patch")
    del off, on, again, other

    # ---- 2. memory and speed ---------------------------------------------------------
    print("\n2. One training step (GB allocated by PyTorch; seconds)")
    print("   setting                held after forward   peak of step   forward   backward   per image")
    for n, ckpt in ((16, False), (16, True), (32, True), (32, False)):
        label = f"batch {n}, patch {'on ' if ckpt else 'off'}"
        torch.cuda.empty_cache()
        try:
            x, y = batch(n)
            step(model, x, y, ckpt)                          # warm-up
            r = step(model, x, y, ckpt)
            per_image = (r["t_fwd"] + r["t_bwd"]) / n * 1000
            print(f"   {label}    {r['held']:>10.2f}        {r['peak']:>10.2f}    {r['t_fwd']:>6.2f}    {r['t_bwd']:>6.2f}    {per_image:>5.0f} ms")
            del r
        except torch.cuda.OutOfMemoryError:
            print(f"   {label}    out of memory")
        model.zero_grad(set_to_none=True)
        del x, y

    print("\nWith the patch working, 'held after forward' and 'peak of step' should both drop")
    print("sharply compared with patch off, and backward should take longer.")
    print("Real training needs roughly 0.3 GB more than these figures (optimizer state), plus headroom.")


if __name__ == "__main__":
    main()