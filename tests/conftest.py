"""Root conftest — registers custom pytest markers for the whole suite."""
from __future__ import annotations


def pytest_configure(config: object) -> None:
    """Register project-level markers so pytest doesn't warn about unknown marks."""
    config.addinivalue_line("markers", "unit: fast unit tests, no external deps")
    config.addinivalue_line("markers", "integration: requires docker-compose")
    config.addinivalue_line("markers", "load: locust load tests")
    config.addinivalue_line("markers", "gpu: requires GPU")
