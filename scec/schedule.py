"""Four-stage causal group schedule."""
import torch

def progressive_masks(groups: int, anchor_period: int, device: torch.device) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Return the initial anchors and dyadic refinement stages.

    For period eight the deterministic order is 0 mod 8, 4 mod 8,
    2 mod 4, and finally 1 mod 2.  Every symbol used as neural context is
    therefore already available to the decoder.
    """
    if anchor_period < 2 or anchor_period & (anchor_period - 1):
        raise ValueError("anchor_period must be a power of two and at least two")
    index = torch.arange(groups, device=device)
    anchor = index.remainder(anchor_period).eq(0)
    targets: list[torch.Tensor] = []
    stride = anchor_period
    while stride > 1:
        targets.append(index.remainder(stride).eq(stride // 2))
        stride //= 2
    covered = anchor.clone()
    for target in targets:
        if bool((covered & target).any()):
            raise RuntimeError("progressive masks overlap")
        covered |= target
    if not bool(covered.all()):
        raise RuntimeError("progressive masks do not cover all vectors")
    return anchor, targets
