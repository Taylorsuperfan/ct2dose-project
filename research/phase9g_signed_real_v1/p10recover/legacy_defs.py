"""Recovered definitions from the supplied legacy notebook.

Definitions below are copied verbatim, not inferred from checkpoint shapes.
Top-level training, file writes, plotting, test evaluation and flexible loaders
were NOT imported. See records/source_provenance.json for cell/segment hashes.
The original inconsistent axis conventions are intentionally preserved.
Global configs are bound in a private module namespace by LegacyPipeline.
"""
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

CONFIG = {}
PHASE10C_CONFIG = {}
PHASE10D_CONFIG = {}


class CTDoseNpyDataset(Dataset):
    def __init__(self, json_path, dose_scale=1000.0):
        self.json_path = Path(json_path)
        self.dose_scale = float(dose_scale)

        with open(self.json_path, "r") as f:
            self.records = json.load(f)

        print(f"Loaded {len(self.records)} records from {self.json_path}")
        print("Dataset dose_scale:", self.dose_scale)

    def __len__(self):
        return len(self.records)

    def normalize_ct(self, ct):
        """
        Match the previous training/evaluation CT scale.

        If CT is raw HU, roughly [-1000, 1000], convert to [0, 1]:
            ct_norm = clip((ct + 1024) / 2048, 0, 1)

        If CT is already normalized, keep it unchanged.
        """
        ct = ct.astype(np.float32)

        if ct.min() < -10 or ct.max() > 10:
            ct = (ct + 1024.0) / 2048.0
            ct = np.clip(ct, 0.0, 1.0)

        return ct.astype(np.float32)

    def scale_dose_if_needed(self, dose):
        """
        Match model dose scale.

        Raw stored dose may be around 0.003.
        Training/evaluation dose is around 3.0.
        """
        dose = dose.astype(np.float32)

        if dose.max() < 0.1:
            dose = dose * self.dose_scale

        return dose.astype(np.float32)

    def __getitem__(self, idx):
        rec = self.records[idx]

        ct = np.load(rec["input_path"]).astype(np.float32)
        dose = np.load(rec["output_path"]).astype(np.float32)

        ct = self.normalize_ct(ct)
        dose = self.scale_dose_if_needed(dose)

        if ct.ndim == 3:
            ct = ct[None, ...]
        if dose.ndim == 3:
            dose = dose[None, ...]

        return {
            "ct": torch.from_numpy(ct),
            "dose": torch.from_numpy(dose),
            "case_id": rec.get("case_id", "unknown"),
            "input_path": rec["input_path"],
            "output_path": rec["output_path"],
        }

class DoubleConv3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class ConditionalUNetFlow3D(nn.Module):
    """
    Rectified-flow velocity model:
        v_theta(xt, x0, t)

    Input channels:
        xt: current noisy/interpolated dose
        x0: CT
        t_map: scalar time expanded to volume
    """
    def __init__(self, in_ch=3, out_ch=1, base_ch=24):
        super().__init__()

        self.enc1 = DoubleConv3D(in_ch, base_ch)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = DoubleConv3D(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool3d(2)

        self.bottleneck = DoubleConv3D(base_ch * 2, base_ch * 4)

        self.up2 = nn.ConvTranspose3d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv3D(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose3d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = DoubleConv3D(base_ch * 2, base_ch)

        # IMPORTANT: checkpoint uses the name out_conv
        self.out_conv = nn.Conv3d(base_ch, out_ch, kernel_size=1)

    def forward(self, xt, x0, t):
        # t can be [B], [B,1], or [B,1,1,1,1]
        if t.ndim == 1:
            t = t.view(-1, 1, 1, 1, 1)
        elif t.ndim == 2:
            t = t.view(-1, 1, 1, 1, 1)

        t_map = t.expand(-1, 1, xt.shape[2], xt.shape[3], xt.shape[4])
        inp = torch.cat([xt, x0, t_map], dim=1)

        e1 = self.enc1(inp)
        e2 = self.enc2(self.pool1(e1))

        b = self.bottleneck(self.pool2(e2))

        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out_conv(d1)

class MultiplicativeAdditiveRefineHead3D(nn.Module):
    def __init__(
        self,
        in_ch=7,
        base_ch=16,
        log_scale_bound=0.15,
        additive_scale=0.015,
    ):
        super().__init__()

        self.log_scale_bound = float(log_scale_bound)
        self.additive_scale = float(additive_scale)

        self.trunk = nn.Sequential(
            nn.Conv3d(in_ch, base_ch, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=base_ch),
            nn.SiLU(),

            nn.Conv3d(base_ch, base_ch, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=base_ch),
            nn.SiLU(),

            nn.Conv3d(base_ch, base_ch, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=base_ch),
            nn.SiLU(),

            nn.Conv3d(base_ch, base_ch, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=base_ch),
            nn.SiLU(),
        )

        self.log_scale_head = nn.Conv3d(base_ch, 1, kernel_size=1)
        self.additive_head = nn.Conv3d(base_ch, 1, kernel_size=1)

    def forward(self, ct, base_pred):
        feat = make_structure_features(ct, base_pred)
        h = self.trunk(feat)

        log_scale = self.log_scale_bound * torch.tanh(self.log_scale_head(h))
        additive = self.additive_scale * torch.tanh(self.additive_head(h))

        refined = base_pred * torch.exp(log_scale) + additive
        refined = torch.clamp(refined, min=0.0)

        return refined, {
            "log_scale": log_scale,
            "additive": additive,
        }

def spatial_gradient_magnitude_3d(x):
    """
    x: [B,1,D,H,W]
    return: [B,1,D,H,W]
    """
    dz = torch.zeros_like(x)
    dy = torch.zeros_like(x)
    dx = torch.zeros_like(x)

    dz[:, :, 1:, :, :] = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
    dy[:, :, :, 1:, :] = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
    dx[:, :, :, :, 1:] = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]

    grad = torch.sqrt(dx * dx + dy * dy + dz * dz + 1e-8)
    return grad

def make_structure_features(ct, base_pred):
    """
    All features are deployable:
    only use CT and base prediction, not GT.

    returns feature tensor [B, 7, D, H, W]
    """
    peak = torch.amax(base_pred, dim=(2, 3, 4), keepdim=True).clamp_min(1e-8)
    rel = base_pred / peak

    core = torch.sigmoid((rel - 0.70) / 0.05)
    shoulder = torch.sigmoid((rel - 0.25) / 0.05) * (1.0 - torch.sigmoid((rel - 0.70) / 0.05))
    tail = torch.sigmoid((0.25 - rel) / 0.05)

    grad = spatial_gradient_magnitude_3d(base_pred)
    grad_norm = grad / (torch.amax(grad, dim=(2, 3, 4), keepdim=True).clamp_min(1e-8))

    features = torch.cat(
        [
            ct,
            base_pred,
            rel,
            core,
            shoulder,
            tail,
            grad_norm,
        ],
        dim=1,
    )

    return features

@torch.no_grad()
def euler_sample_ct2dose(model, ct, steps=10):
    """
    Correct sampler for this CT-to-dose rectified-flow model.

    Training convention is effectively:
        x_t = (1 - t) * CT + t * dose

    Therefore inference should start from CT, not from zero:
        x_0 = CT
        x_{t+dt} = x_t + dt * v_theta(x_t, CT, t)

    Midpoint time t=(i+0.5)/steps gave the best sanity-check result.
    """
    model.eval()

    # IMPORTANT: start from CT, not zeros
    x = ct.clone()

    dt = 1.0 / steps

    for i in range(steps):
        t = torch.full(
            (ct.shape[0],),
            (i + 0.5) / steps,
            device=ct.device,
            dtype=ct.dtype,
        )

        v = model(x, ct, t)
        x = x + dt * v
        x = torch.clamp(x, min=0.0)

    return x

def apply_core_scalar_calibration_deployable(
    pred,
    core_scale=0.990,
    core_thr=0.70,
    tau=0.04,
):
    """
    pred: [B,1,D,H,W]

    core mask is computed from pred itself:
        rel = pred / pred.max()

    final = pred * (1 + core_mask * (core_scale - 1))
    """
    peak = torch.amax(pred, dim=(2, 3, 4), keepdim=True).clamp_min(1e-8)
    rel = pred / peak

    core_mask = torch.sigmoid((rel - core_thr) / tau)
    scale_field = 1.0 + core_mask * (core_scale - 1.0)

    calibrated = pred * scale_field
    calibrated = torch.clamp(calibrated, min=0.0)

    aux = {
        "core_mask_mean": float(core_mask.mean().detach().cpu()),
        "core_mask_max": float(core_mask.max().detach().cpu()),
        "scale_field_min": float(scale_field.min().detach().cpu()),
        "scale_field_max": float(scale_field.max().detach().cpu()),
    }

    return calibrated, aux

@torch.no_grad()
def predict_phase9g(ct, steps=30):
    base_pred = euler_sample_ct2dose(base_model, ct, steps=steps)

    phase9d_pred, phase9d_aux = model_9d_plus(ct, base_pred)

    phase9g_pred, phase9g_aux = apply_core_scalar_calibration_deployable(
        phase9d_pred,
        core_scale=CONFIG["core_scale"],
        core_thr=CONFIG["core_thr"],
        tau=CONFIG["tau"],
    )

    return {
        "base_pred": base_pred,
        "phase9d_pred": phase9d_pred,
        "phase9g_pred": phase9g_pred,
        "phase9d_aux": phase9d_aux,
        "phase9g_aux": phase9g_aux,
    }

def to_numpy_3d(x):
    """
    Convert [1,D,H,W] or [D,H,W] tensor to numpy [D,H,W].
    """
    if torch.is_tensor(x):
        x = x.detach().cpu().float().numpy()

    x = np.asarray(x)

    if x.ndim == 4:
        x = x[0]

    return x

def extract_axis_profiles(pred_tensor, gt_tensor):
    """
    Extract central line profiles along x/y/z through GT peak.

    pred_tensor, gt_tensor:
        [1,D,H,W] tensors

    Returns:
        dict with axis -> {"pred": profile, "gt": profile}
    """
    pred = to_numpy_3d(pred_tensor)
    gt = to_numpy_3d(gt_tensor)

    # shape [D,H,W]
    peak_idx = np.unravel_index(np.argmax(gt), gt.shape)
    zc, yc, xc = peak_idx

    profiles = {
        "x": {
            "pred": pred[zc, yc, :],
            "gt": gt[zc, yc, :],
        },
        "y": {
            "pred": pred[zc, :, xc],
            "gt": gt[zc, :, xc],
        },
        "z": {
            "pred": pred[:, yc, xc],
            "gt": gt[:, yc, xc],
        },
    }

    return profiles

def percentage_error_curve(pred_profile, gt_profile, eps=1e-8, threshold_frac=0.01):
    pred = np.asarray(pred_profile).astype(np.float64)
    gt = np.asarray(gt_profile).astype(np.float64)

    gt_peak = max(float(np.max(gt)), eps)
    valid = gt >= threshold_frac * gt_peak

    pct = np.full_like(gt, np.nan, dtype=np.float64)
    pct[valid] = 100.0 * np.abs(pred[valid] - gt[valid]) / (np.abs(gt[valid]) + eps)

    mean_pct = float(np.nanmean(pct))
    max_pct = float(np.nanmax(pct))
    valid_count = int(np.sum(valid))
    threshold = gt_peak * threshold_frac

    return pct, mean_pct, max_pct, valid_count, threshold

def profile_error_metrics_np(pred_profile, gt_profile, threshold_frac=0.01):
    """
    Simple professor-style profile metric:
    valid = gt >= threshold_frac * gt_peak
    """
    pct, mean_pct, max_pct, valid_count, threshold = percentage_error_curve(
        pred_profile,
        gt_profile,
        threshold_frac=threshold_frac,
    )

    abs_err = np.abs(np.asarray(pred_profile) - np.asarray(gt_profile))
    gt = np.asarray(gt_profile)
    gt_peak = max(float(np.max(gt)), 1e-8)
    valid = gt >= threshold_frac * gt_peak

    mean_abs = float(np.mean(abs_err[valid])) if np.sum(valid) > 0 else np.nan

    return {
        "mean_pct": mean_pct,
        "max_pct": max_pct,
        "mean_abs": mean_abs,
        "valid_count": valid_count,
        "threshold": threshold,
    }

def make_coord_channels_like(x):
    """
    x shape: [B, 1, D, H, W]
    returns:
        x_depth: [B,1,D,H,W]
        radial:  [B,1,D,H,W]
    """
    B, C, D, H, W = x.shape
    device = x.device
    dtype = x.dtype

    xd = torch.linspace(0.0, 1.0, D, device=device, dtype=dtype).view(1, 1, D, 1, 1)
    yy = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype).view(1, 1, 1, H, 1)
    zz = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype).view(1, 1, 1, 1, W)

    x_depth = xd.expand(B, 1, D, H, W)

    radial = torch.sqrt(yy ** 2 + zz ** 2)
    radial = radial / torch.clamp(radial.max(), min=1e-6)
    radial = radial.expand(B, 1, D, H, W)

    return x_depth, radial

def gradient_x(v):
    """
    Finite difference along D / x direction.
    v: [B,1,D,H,W]
    """
    gx = torch.zeros_like(v)

    if v.shape[2] > 2:
        gx[:, :, 1:-1] = 0.5 * (v[:, :, 2:] - v[:, :, :-2])
        gx[:, :, 0] = v[:, :, 1] - v[:, :, 0]
        gx[:, :, -1] = v[:, :, -1] - v[:, :, -2]

    return gx

def gradient_mag_3d(v):
    gx = torch.zeros_like(v)
    gy = torch.zeros_like(v)
    gz = torch.zeros_like(v)

    if v.shape[2] > 2:
        gx[:, :, 1:-1] = 0.5 * (v[:, :, 2:] - v[:, :, :-2])

    if v.shape[3] > 2:
        gy[:, :, :, 1:-1] = 0.5 * (v[:, :, :, 2:] - v[:, :, :, :-2])

    if v.shape[4] > 2:
        gz[:, :, :, :, 1:-1] = 0.5 * (v[:, :, :, :, 2:] - v[:, :, :, :, :-2])

    return torch.sqrt(gx ** 2 + gy ** 2 + gz ** 2 + 1e-12)

def make_dose_region_features(pred, eps=1e-6):
    """
    pred: [B,1,D,H,W]
    returns:
        relative dose, main/core/shoulder/tail masks
    """
    B = pred.shape[0]
    maxv = pred.flatten(1).max(dim=1).values.view(B, 1, 1, 1, 1)
    rel = pred / torch.clamp(maxv, min=eps)

    main_mask = (rel > PHASE10C_CONFIG["main_frac"]).float()
    core_mask = (rel > PHASE10C_CONFIG["core_frac"]).float()

    shoulder_mask = (
        (rel > PHASE10C_CONFIG["shoulder_low"])
        & (rel <= PHASE10C_CONFIG["shoulder_high"])
    ).float()

    tail_mask = (
        (rel > 0.01)
        & (rel <= PHASE10C_CONFIG["main_frac"])
    ).float()

    return rel, main_mask, core_mask, shoulder_mask, tail_mask

def make_phase10c_features(ct, phase9g_pred):
    """
    Feature channels:
      0  ct
      1  phase9g_pred
      2  relative dose
      3  x_depth
      4  radial
      5  main_mask
      6  core_mask
      7  shoulder_mask
      8  tail_mask
      9  grad_x_pred
      10 grad_mag_pred
    """
    x_depth, radial = make_coord_channels_like(phase9g_pred)
    rel, main_mask, core_mask, shoulder_mask, tail_mask = make_dose_region_features(phase9g_pred)

    gx = gradient_x(phase9g_pred)
    gmag = gradient_mag_3d(phase9g_pred)

    feat = torch.cat(
        [
            ct,
            phase9g_pred,
            rel,
            x_depth,
            radial,
            main_mask,
            core_mask,
            shoulder_mask,
            tail_mask,
            gx,
            gmag,
        ],
        dim=1,
    )

    return feat

class FalloffAwareBoundedRefineHead3D(nn.Module):
    def __init__(self, in_ch=11, base_ch=16, delta_scale=0.008):
        super().__init__()

        self.delta_scale = float(delta_scale)

        self.trunk = nn.Sequential(
            nn.Conv3d(in_ch, base_ch, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=base_ch),
            nn.SiLU(),

            nn.Conv3d(base_ch, base_ch, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=base_ch),
            nn.SiLU(),

            nn.Conv3d(base_ch, base_ch, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=base_ch),
            nn.SiLU(),

            nn.Conv3d(base_ch, base_ch, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=base_ch),
            nn.SiLU(),
        )

        self.delta_head = nn.Conv3d(base_ch, 1, kernel_size=1)

    def forward(self, ct, phase9g_pred):
        feat = make_phase10c_features(ct, phase9g_pred)
        h = self.trunk(feat)

        raw_delta = self.delta_head(h)
        delta = self.delta_scale * torch.tanh(raw_delta)

        rel, main_mask, core_mask, shoulder_mask, tail_mask = make_dose_region_features(phase9g_pred)

        correction_mask = torch.ones_like(phase9g_pred)

        # Stronger core protection than Phase10C
        correction_mask = correction_mask * (
            1.0 - PHASE10D_CONFIG["core_protect_strength"] * core_mask
        )

        # Tail can move slightly, but not too freely
        correction_mask = correction_mask * (
            1.0 - PHASE10D_CONFIG["tail_protect_strength"] * tail_mask
        )

        # Allow correction in main + shoulder + controlled tail
        meaningful_region = torch.clamp(
            main_mask + 0.55 * shoulder_mask + 0.25 * tail_mask,
            min=0.0,
            max=1.0,
        )

        correction_mask = correction_mask * meaningful_region

        bounded_delta = correction_mask * delta

        refined = phase9g_pred + bounded_delta
        refined = torch.clamp(refined, min=0.0)

        return refined, {
            "delta": bounded_delta,
            "raw_delta": raw_delta,
            "correction_mask": correction_mask,
        }

def central_profiles_torch(v):
    B, C, D, H, W = v.shape
    cx, cy, cz = D // 2, H // 2, W // 2

    px = v[:, 0, :, cy, cz]
    py = v[:, 0, cx, :, cz]
    pz = v[:, 0, cx, cy, :]

    return px, py, pz

def masked_profile_l1(pred_p, gt_p, threshold_frac=0.01):
    maxv = gt_p.max(dim=1, keepdim=True).values
    thr = threshold_frac * maxv
    mask = (gt_p > thr).float()

    err = torch.abs(pred_p - gt_p) * mask
    denom = torch.clamp(mask.sum(dim=1), min=1.0)

    return (err.sum(dim=1) / denom).mean()

def profile_slope_l1(pred_p, gt_p, threshold_frac=0.01):
    dp = pred_p[:, 1:] - pred_p[:, :-1]
    dg = gt_p[:, 1:] - gt_p[:, :-1]

    maxv = gt_p.max(dim=1, keepdim=True).values
    thr = threshold_frac * maxv
    mask = ((gt_p[:, 1:] > thr) | (gt_p[:, :-1] > thr)).float()

    err = torch.abs(dp - dg) * mask
    denom = torch.clamp(mask.sum(dim=1), min=1.0)

    return (err.sum(dim=1) / denom).mean()

def no_worse_profile_loss(new_p, old_p, gt_p, margin_abs=0.0004, threshold_frac=0.01):
    maxv = gt_p.max(dim=1, keepdim=True).values
    thr = threshold_frac * maxv
    mask = (gt_p > thr).float()

    old_err = torch.abs(old_p - gt_p)
    new_err = torch.abs(new_p - gt_p)

    penalty = F.relu(new_err - old_err - margin_abs) * mask
    denom = torch.clamp(mask.sum(dim=1), min=1.0)

    return (penalty.sum(dim=1) / denom).mean()

def log_profile_l1(pred_p, gt_p, threshold_frac=0.01, eps=1e-4):
    """
    Relative-error-like loss for profile.
    More suitable for falloff region than plain L1.
    """
    maxv = gt_p.max(dim=1, keepdim=True).values
    thr = threshold_frac * maxv
    mask = (gt_p > thr).float()

    lp = torch.log(torch.clamp(pred_p, min=eps))
    lg = torch.log(torch.clamp(gt_p, min=eps))

    err = torch.abs(lp - lg) * mask
    denom = torch.clamp(mask.sum(dim=1), min=1.0)

    return (err.sum(dim=1) / denom).mean()

def falloff_mask_from_gt_profile(gt_p):
    """
    Falloff region on x-profile:
    gt between low_frac and high_frac of peak.
    """
    maxv = gt_p.max(dim=1, keepdim=True).values
    rel = gt_p / torch.clamp(maxv, min=1e-8)

    mask = (
        (rel >= PHASE10D_CONFIG["falloff_low_frac"])
        & (rel <= PHASE10D_CONFIG["falloff_high_frac"])
    ).float()

    return mask

def falloff_log_loss(pred_x, gt_x, eps=1e-4):
    mask = falloff_mask_from_gt_profile(gt_x)

    lp = torch.log(torch.clamp(pred_x, min=eps))
    lg = torch.log(torch.clamp(gt_x, min=eps))

    err = torch.abs(lp - lg) * mask
    denom = torch.clamp(mask.sum(dim=1), min=1.0)

    return (err.sum(dim=1) / denom).mean()

def falloff_no_worse_log_loss(new_x, old_x, gt_x, eps=1e-4):
    """
    Penalize if new log-profile error is worse than Phase9G
    specifically in falloff region.
    """
    mask = falloff_mask_from_gt_profile(gt_x)

    ln = torch.log(torch.clamp(new_x, min=eps))
    lo = torch.log(torch.clamp(old_x, min=eps))
    lg = torch.log(torch.clamp(gt_x, min=eps))

    new_err = torch.abs(ln - lg)
    old_err = torch.abs(lo - lg)

    penalty = F.relu(
        new_err - old_err - PHASE10D_CONFIG["falloff_margin_log"]
    ) * mask

    denom = torch.clamp(mask.sum(dim=1), min=1.0)

    return (penalty.sum(dim=1) / denom).mean()

def smoothness_loss_3d(delta):
    gx = torch.abs(delta[:, :, 1:] - delta[:, :, :-1]).mean()
    gy = torch.abs(delta[:, :, :, 1:] - delta[:, :, :, :-1]).mean()
    gz = torch.abs(delta[:, :, :, :, 1:] - delta[:, :, :, :, :-1]).mean()
    return gx + gy + gz

def phase10d_loss(refined, phase9g_pred, gt, aux):
    voxel_l1 = torch.mean(torch.abs(refined - gt))

    new_x, new_y, new_z = central_profiles_torch(refined)
    old_x, old_y, old_z = central_profiles_torch(phase9g_pred)
    gt_x, gt_y, gt_z = central_profiles_torch(gt)

    along_loss = masked_profile_l1(
        new_x,
        gt_x,
        threshold_frac=PHASE10D_CONFIG["threshold_frac"],
    )

    along_slope = profile_slope_l1(
        new_x,
        gt_x,
        threshold_frac=PHASE10D_CONFIG["threshold_frac"],
    )

    along_log = log_profile_l1(
        new_x,
        gt_x,
        threshold_frac=PHASE10D_CONFIG["threshold_frac"],
    )

    falloff_log = falloff_log_loss(new_x, gt_x)

    falloff_noworse = falloff_no_worse_log_loss(
        new_x,
        old_x,
        gt_x,
    )

    perp_noworse_y = no_worse_profile_loss(
        new_y,
        old_y,
        gt_y,
        margin_abs=PHASE10D_CONFIG["perp_margin_abs"],
        threshold_frac=PHASE10D_CONFIG["threshold_frac"],
    )

    perp_noworse_z = no_worse_profile_loss(
        new_z,
        old_z,
        gt_z,
        margin_abs=PHASE10D_CONFIG["perp_margin_abs"],
        threshold_frac=PHASE10D_CONFIG["threshold_frac"],
    )

    perp_noworse = 0.5 * (perp_noworse_y + perp_noworse_z)

    rel_gt, main_mask, core_mask, shoulder_mask, tail_mask = make_dose_region_features(gt)

    old_core_err = torch.abs(phase9g_pred - gt)
    new_core_err = torch.abs(refined - gt)

    core_penalty = F.relu(
        new_core_err - old_core_err - PHASE10D_CONFIG["core_margin_abs"]
    ) * core_mask

    core_noworse = core_penalty.sum() / torch.clamp(core_mask.sum(), min=1.0)

    residual_l1 = torch.mean(torch.abs(aux["delta"]))
    smooth = smoothness_loss_3d(aux["delta"])

    total = (
        PHASE10D_CONFIG["lambda_voxel"] * voxel_l1
        + PHASE10D_CONFIG["lambda_along"] * along_loss
        + PHASE10D_CONFIG["lambda_along_slope"] * along_slope
        + PHASE10D_CONFIG["lambda_along_log"] * along_log
        + PHASE10D_CONFIG["lambda_falloff_log"] * falloff_log
        + PHASE10D_CONFIG["lambda_falloff_no_worse"] * falloff_noworse
        + PHASE10D_CONFIG["lambda_perp_no_worse"] * perp_noworse
        + PHASE10D_CONFIG["lambda_core_no_worse"] * core_noworse
        + PHASE10D_CONFIG["lambda_residual_l1"] * residual_l1
        + PHASE10D_CONFIG["lambda_smooth"] * smooth
    )

    return total, {
        "total": total.detach().item(),
        "voxel_l1": voxel_l1.detach().item(),
        "along_loss": along_loss.detach().item(),
        "along_slope": along_slope.detach().item(),
        "along_log": along_log.detach().item(),
        "falloff_log": falloff_log.detach().item(),
        "falloff_noworse": falloff_noworse.detach().item(),
        "perp_noworse": perp_noworse.detach().item(),
        "core_noworse": core_noworse.detach().item(),
        "residual_l1": residual_l1.detach().item(),
        "smooth": smooth.detach().item(),
    }

@torch.no_grad()
def predict_phase10d(ct, steps=None):
    if steps is None:
        steps = PHASE10D_CONFIG["euler_steps"]

    base_model.eval()
    model_9d_plus.eval()
    phase10d_model.eval()

    preds9 = predict_phase9g(ct, steps=steps)

    phase9g_pred = preds9["phase9g_pred"]

    refined, aux = phase10d_model(
        ct,
        phase9g_pred,
    )

    preds9["phase10d_pred"] = refined
    preds9["phase10d_aux"] = aux

    return preds9
