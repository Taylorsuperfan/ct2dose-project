"""Declared engineering settings; not hyperparameters prescribed by the supervisor."""
from dataclasses import dataclass, asdict
import math

@dataclass(frozen=True)
class Config:
    stage: str = 'pilot'
    seed: int = 17
    updates: int = 384
    batch_size: int = 2
    learning_rate: float = 0.0003
    weight_decay: float = 0.00001
    gradient_clip: float = 5.0
    channels: int = 16
    point_hidden: int = 64
    delta: float = 1.0
    sign_loss_weight: float = 1.0
    sign_active_threshold: float = 0.01
    source_points_per_bank: int = 256
    target_draws_per_bank: int = 256
    banks_per_record: int = 2
    pairs_per_record: int = 1024
    coupling_iterations: int = 1000
    coupling_relative_tolerance: float = 0.00001
    coupling_seed: int = 2401
    ode_steps: int = 16
    query_chunk: int = 8192
    monitor_every: int = 32
    save_every: int = 32
    monitor_records_per_case: int = 20
    monitor_seed: int = 732
    inference_reconstruction: str = 'trilinear_grid_sign_last'
    checkpoint_selection: str = 'monitor_case_mean_clipped_dose_rmse'
    measure_convention: str = 'voxel_sum_common_divisor_N_for_coupling'
    zero_target_policy: str = 'stop_nonzero_pilot_scope_no_epsilon_replacement'
    def validate(self):
        if self.stage not in ('one','six','pilot'): raise ValueError('Stage must be one, six, or pilot.')
        ints = ('updates','batch_size','channels','point_hidden','source_points_per_bank','target_draws_per_bank',
                'banks_per_record','pairs_per_record','coupling_iterations','ode_steps','query_chunk','monitor_every','save_every','monitor_records_per_case')
        for k in ints:
            if type(getattr(self,k)) is not int or getattr(self,k)<1: raise ValueError('Invalid positive integer: '+k)
        for k in ('learning_rate','gradient_clip','delta','coupling_relative_tolerance'):
            if not math.isfinite(getattr(self,k)) or getattr(self,k)<=0: raise ValueError('Invalid parameter: '+k)
        for k in ('weight_decay','sign_loss_weight','sign_active_threshold'):
            if not math.isfinite(getattr(self,k)) or getattr(self,k)<0: raise ValueError('Invalid parameter: '+k)
        if math.pi*self.delta <= math.sqrt(3): raise ValueError('This unit-cube implementation requires all pair distances below the WFR travel cutoff.')
        if self.source_points_per_bank>32768: raise ValueError('Source support exceeds available voxel centers.')
        if self.source_points_per_bank*self.target_draws_per_bank>1048576: raise ValueError('Dense coupling matrix exceeds this bounded pilot design.')
        if self.inference_reconstruction!='trilinear_grid_sign_last' or self.measure_convention!='voxel_sum_common_divisor_N_for_coupling': raise ValueError('Unsupported scientific convention.')
        if self.checkpoint_selection!='monitor_case_mean_clipped_dose_rmse' or self.zero_target_policy!='stop_nonzero_pilot_scope_no_epsilon_replacement': raise ValueError('Unsupported policy.')
        return self
    def to_dict(self): return asdict(self)


def recipe(stage):
    return Config(stage=stage,updates=256 if stage=='one' else 384,batch_size=1 if stage=='one' else 2).validate()
