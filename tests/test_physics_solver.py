import pytest

from ct2dose.physics.attenuation_diffusion import (
    AttenuationDiffusionConfig,
    solve_attenuation_diffusion,
)


def test_placeholder_requires_formulation() -> None:
    config = AttenuationDiffusionConfig(
        attenuation_coefficient=0.1,
        transverse_diffusion_coefficient=0.01,
        depth_step=1.0,
    )
    with pytest.raises(NotImplementedError):
        solve_attenuation_diffusion(source=None, density=None, config=config)
