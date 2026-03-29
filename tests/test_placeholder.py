def test_project_structure():
    """Verify core project folders exist."""
    from pathlib import Path

    assert Path("data/raw").exists()
    assert Path("data/external").exists()
    assert Path("notebooks").exists()
    assert Path("models").exists()
    assert Path("api").exists()
    assert Path("dashboard").exists()
    assert Path("scripts").exists()


def test_config_loads():
    """Verify config loads without errors."""
    from config import settings

    assert settings.app_name == "CreditIQ"
    assert settings.jwt_algorithm == "HS256"
