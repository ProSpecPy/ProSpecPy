def test_package_import():
    from prospecpy import __version__
    
    assert __version__ is not None, "Version should not be None"
