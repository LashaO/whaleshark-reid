def test_package_importable():
    import whaleshark_reid

    assert whaleshark_reid.__version__ == "0.1.0"


def test_core_submodule_importable():
    from whaleshark_reid import core  # noqa: F401


def test_storage_submodule_importable():
    from whaleshark_reid import storage  # noqa: F401
