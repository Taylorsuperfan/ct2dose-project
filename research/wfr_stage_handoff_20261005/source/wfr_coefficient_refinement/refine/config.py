"""One prospective A/B experiment. Defaults are engineering choices, not fitted results."""
from dataclasses import dataclass, asdict
import math

ARMS = ("composition", "profile")
GUARDS = (
    "rmse_stored_units",
    "x_profile_rmse_stored_units",
    "y_mean_pct", "z_mean_pct",
    "y_profile_rmse_stored_units", "z_profile_rmse_stored_units",
)

@dataclass(frozen=True)
class RefinementConfig:
    seed: int = 29
    updates: int = 384
    batch_size: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip: float = 5.0
    monitor_every: int = 32
    save_every: int = 32
    sign_threshold: float = 0.01
    sign_weight: float = 0.1
    reconstruction_weight: float = 1.0
    profile_weight: float = 1.0
    profile_relative_share: float = 0.5
    profile_peak_fraction: float = 0.01
    legacy_epsilon_model_units: float = 1e-8
    guard_relative_tolerance: float = 0.01
    numerical_relative_slack: float = 1e-6
    min_x_improvement_percentage_points: float = 1e-6
    normalization_floor: float = 1e-8
    uncertainty_oracle_used_for_training: bool = False

    def validate(self):
        for name in ("seed", "updates", "batch_size", "monitor_every", "save_every"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise ValueError("Invalid integer setting: " + name)
        for name in ("learning_rate", "gradient_clip", "normalization_floor", "legacy_epsilon_model_units"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError("Invalid positive setting: " + name)
        for name in ("weight_decay", "sign_threshold", "sign_weight", "reconstruction_weight",
                     "profile_weight", "guard_relative_tolerance", "numerical_relative_slack",
                     "min_x_improvement_percentage_points"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError("Invalid nonnegative setting: " + name)
        if not 0 < self.profile_peak_fraction <= 1 or not 0 <= self.profile_relative_share <= 1:
            raise ValueError("Invalid profile fractions.")
        if self.reconstruction_weight <= 0 or self.profile_weight <= 0:
            raise ValueError("This two-arm experiment requires reconstruction and profile supervision.")
        if self.uncertainty_oracle_used_for_training is not False:
            raise ValueError("Monitor oracle labels must not be used for training.")
        return self

    def to_dict(self):
        return asdict(self)
