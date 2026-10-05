"""The old sign architecture, now used as a bounded signed coefficient."""
import copy
import numpy as np
import torch
from torch import nn
from cwfr.model import Encoder


class CoefficientHead(nn.Module):
    def __init__(self, channels=16):
        super().__init__()
        self.encoder = Encoder(channels)
        self.out = nn.Conv3d(channels, 1, 1)

    @classmethod
    def from_parent(cls, parent):
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.encoder = copy.deepcopy(parent.sign_encoder)
        obj.out = copy.deepcopy(parent.sign_out)
        return obj

    def forward(self, condition):
        if condition.ndim != 5 or condition.shape[1] != 2:
            raise ValueError("Expected a batch of two-channel 3-D conditions.")
        return self.out(self.encoder(condition)[0])[:, 0]

    @torch.no_grad()
    def coefficient(self, condition):
        self.eval()
        return torch.tanh(self(condition) / 2)


@torch.no_grad()
def predict_new_record(parent_model, head, condition, base_model_units,
                       residual_scale, dose_scale_factor, steps=16, query_chunk=8192):
    """Target-free deployment interface. No target, sign label, or target total is accepted."""
    if residual_scale <= 0 or dose_scale_factor <= 0:
        raise ValueError("Positive saved scales are required.")
    result = parent_model.predict(condition, steps, query_chunk)
    coefficient = head.coefficient(condition)[0].cpu().numpy().astype(np.float64)
    magnitude = result["magnitude"]
    base = np.asarray(base_model_units, dtype=np.float64)
    if base.shape != magnitude.shape:
        raise ValueError("Base prediction and reconstructed magnitude differ in shape.")
    raw = (base + residual_scale * magnitude * coefficient) / dose_scale_factor
    if not np.isfinite(raw).all():
        raise FloatingPointError("Nonfinite composed prediction.")
    return {"prediction_stored": np.maximum(raw, 0).astype(np.float32),
            "raw_prediction_stored": raw.astype(np.float32),
            "magnitude_normalized": magnitude.astype(np.float32),
            "coefficient": coefficient.astype(np.float32),
            "target_used_at_inference": False}
