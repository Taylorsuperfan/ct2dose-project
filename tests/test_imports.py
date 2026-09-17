def test_package_imports() -> None:
    import ct2dose  # noqa: F401
    from ct2dose.data import splits  # noqa: F401
    from ct2dose.geometry import beam_coordinates  # noqa: F401
    from ct2dose.physics import hu_to_density, wed_attenuation  # noqa: F401
