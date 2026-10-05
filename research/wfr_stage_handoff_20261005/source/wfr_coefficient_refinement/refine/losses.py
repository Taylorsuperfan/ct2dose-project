"""Raw composite losses. Targets define supervision, never inference features."""
import numpy as np
import torch
from torch.nn import functional as F


def components(logits, magnitude, residual, base, target, scale, cfg):
    """Return per-record losses. All arrays except logits are fixed batch tensors."""
    if logits.shape != magnitude.shape or logits.shape != residual.shape:
        raise ValueError("Factor shapes differ.")
    if magnitude.requires_grad or target.requires_grad or residual.requires_grad:
        raise ValueError("Magnitude and supervised targets must be detached constants.")
    coefficient = torch.tanh(logits / 2)
    error_u = magnitude * coefficient - residual
    dims = tuple(range(1, error_u.ndim))
    reconstruction = error_u.square().mean(dim=dims)
    weight = residual.abs() * (residual.abs() > cfg.sign_threshold)
    bce = F.binary_cross_entropy_with_logits(logits, (residual > 0).to(logits.dtype), reduction="none")
    sign_numerator = (weight * bce).sum(dim=dims)
    sign_denominator = weight.sum(dim=dims)
    sign_per_record = sign_numerator / sign_denominator.clamp_min(1e-12)
    # Preserve the previous amplitude weighting within the selected batch.
    sign_batch = sign_numerator.sum() / sign_denominator.sum().clamp_min(1e-12)

    relative, absolute = [], []
    for b in range(len(logits)):
        d, h, w = target[b].shape
        index = int(target[b].detach().argmax())
        zi = index // (h * w); yi = (index % (h * w)) // w
        target_line = target[b, zi, yi]
        line_error_model = scale * error_u[b, zi, yi]
        peak = target_line.max()
        valid = target_line >= cfg.profile_peak_fraction * torch.maximum(
            peak, torch.as_tensor(cfg.legacy_epsilon_model_units, device=peak.device, dtype=peak.dtype))
        if bool(valid.any()):
            relative.append((line_error_model[valid].abs() /
                             (target_line[valid].abs() + cfg.legacy_epsilon_model_units)).mean())
        else:
            relative.append(line_error_model.sum() * 0)
        # Dimensionless residual units; minimizing this MSE also minimizes its RMSE.
        absolute.append(error_u[b, zi, yi].square().mean())
    return {
        "sign_bce": sign_per_record,
        "sign_batch": sign_batch,
        "sign_numerator": sign_numerator,
        "sign_denominator": sign_denominator,
        "reconstruction_mse": reconstruction,
        "profile_relative": torch.stack(relative),
        "profile_absolute_mse": torch.stack(absolute),
    }


def objective(logits, batch, normalizers, cfg, arm):
    if arm not in ("composition", "profile"):
        raise ValueError("Unknown experimental arm.")
    p = components(logits, batch["magnitude"], batch["residual"],
                   batch["base"], batch["target"], batch["scale"], cfg)
    sign = p["sign_batch"] / normalizers["sign_bce"]
    volume = p["reconstruction_mse"].mean() / normalizers["reconstruction_mse"]
    line = (cfg.profile_relative_share * p["profile_relative"].mean() / normalizers["profile_relative"]
            + (1 - cfg.profile_relative_share) * p["profile_absolute_mse"].mean()
            / normalizers["profile_absolute_mse"])
    total = cfg.sign_weight * sign + cfg.reconstruction_weight * volume
    if arm == "profile":
        total = total + cfg.profile_weight * line
    return total, {
        "total": float(total.detach()), "sign_scaled": float(sign.detach()),
        "reconstruction_scaled": float(volume.detach()), "profile_scaled": float(line.detach()),
    }


def make_batch(records, device, scale):
    keys = ("condition", "magnitude", "residual", "base", "target")
    batch = {k: torch.as_tensor(np.stack([r[k] for r in records]), dtype=torch.float32, device=device)
             for k in keys}
    batch["scale"] = float(scale)
    return batch
