"""Unit tests for pipeline/pool/model_pool.py.

No real GPU or torch installation is required — model loading is always mocked.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

import pipeline.pool.model_pool as pool_module
from pipeline.pool.model_pool import (
    ModelEntry,
    ModelPool,
    _ModelContext,
    get_model_pool,
)

# ── Helpers ────────────────────────────────────────────────────────────────────


def _mock_model() -> MagicMock:
    """Return a MagicMock whose .to() / .half() / .eval() all return self."""
    m = MagicMock()
    m.to.return_value = m
    m.half.return_value = m
    m.eval.return_value = m
    return m


def _fake_settings(
    max_vram_gb: float = 8.0,
    use_fp16: bool = False,
    device: str = "cpu",
) -> MagicMock:
    s = MagicMock()
    s.max_vram_gb = max_vram_gb
    s.use_fp16 = use_fp16
    s.device = device
    return s


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def settings():
    return _fake_settings()


@pytest.fixture
def pool():
    return ModelPool()


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Ensure the module-level singleton is reset around every test."""
    original = pool_module._pool_instance
    yield
    pool_module._pool_instance = original


# ── Pool initialisation ────────────────────────────────────────────────────────


@pytest.mark.unit
class TestPoolInit:
    def test_pool_starts_empty(self, pool, settings):
        with patch("pipeline.pool.model_pool.get_settings", return_value=settings):
            assert pool.stats()["pool_size"] == 0

    def test_loaded_models_empty_on_init(self, pool, settings):
        with patch("pipeline.pool.model_pool.get_settings", return_value=settings):
            assert pool.stats()["loaded_models"] == []

    def test_total_vram_zero_on_init(self, pool, settings):
        with patch("pipeline.pool.model_pool.get_settings", return_value=settings):
            assert pool.stats()["total_vram_gb"] == 0.0


# ── register_loader ────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestRegisterLoader:
    def test_stores_loader(self, pool):
        fn = lambda: None  # noqa: E731
        pool.register_loader("mymodel", fn)
        assert pool._loaders["mymodel"] is fn

    def test_overrides_vram_estimate(self, pool):
        pool.register_loader("mymodel", lambda: None, vram_gb=5.0)
        assert pool._vram_estimates["mymodel"] == 5.0

    def test_does_not_change_vram_when_none(self, pool):
        default = pool._vram_estimates.get("sam2")
        pool.register_loader("sam2", lambda: None, vram_gb=None)
        assert pool._vram_estimates.get("sam2") == default


# ── get() ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestGet:
    def test_loads_model_on_first_call(self, pool, settings):
        model = _mock_model()
        loader = MagicMock(return_value=model)
        pool.register_loader("sam2", loader, vram_gb=1.0)

        with patch("pipeline.pool.model_pool.get_settings", return_value=settings):
            result = pool.get("sam2")

        loader.assert_called_once()
        assert result is model.to.return_value  # model.to(device) was called

    def test_returns_same_model_on_second_call(self, pool, settings):
        model = _mock_model()
        pool.register_loader("sam2", MagicMock(return_value=model), vram_gb=1.0)

        with patch("pipeline.pool.model_pool.get_settings", return_value=settings):
            first = pool.get("sam2")
            second = pool.get("sam2")

        assert first is second

    def test_loader_called_only_once(self, pool, settings):
        loader = MagicMock(return_value=_mock_model())
        pool.register_loader("sam2", loader, vram_gb=1.0)

        with patch("pipeline.pool.model_pool.get_settings", return_value=settings):
            pool.get("sam2")
            pool.get("sam2")
            pool.get("sam2")

        loader.assert_called_once()

    def test_updates_last_used_on_repeated_get(self, pool, settings):
        pool.register_loader("sam2", MagicMock(return_value=_mock_model()), vram_gb=1.0)

        with patch("pipeline.pool.model_pool.get_settings", return_value=settings):
            pool.get("sam2")
            t_after_first = pool._pool["sam2"].last_used

            time.sleep(0.01)  # small sleep so timestamp advances
            pool.get("sam2")
            t_after_second = pool._pool["sam2"].last_used

        assert t_after_second > t_after_first

    def test_raises_key_error_for_unknown_model(self, pool, settings):
        with patch("pipeline.pool.model_pool.get_settings", return_value=settings):
            with pytest.raises(KeyError, match="no_such_model"):
                pool.get("no_such_model")

    def test_increments_use_count(self, pool, settings):
        pool.register_loader("sam2", MagicMock(return_value=_mock_model()), vram_gb=1.0)

        with patch("pipeline.pool.model_pool.get_settings", return_value=settings):
            pool.get("sam2")  # loads (use_count stays 0 on load)
            pool.get("sam2")  # first cached access
            pool.get("sam2")  # second cached access

        assert pool._pool["sam2"].use_count == 2

    def test_model_moved_to_device_on_load(self, pool, settings):
        model = _mock_model()
        pool.register_loader("sam2", MagicMock(return_value=model), vram_gb=1.0)

        with patch("pipeline.pool.model_pool.get_settings", return_value=settings):
            pool.get("sam2")

        model.to.assert_called_with("cpu")

    def test_model_eval_called_on_load(self, pool, settings):
        model = _mock_model()
        pool.register_loader("sam2", MagicMock(return_value=model), vram_gb=1.0)

        with patch("pipeline.pool.model_pool.get_settings", return_value=settings):
            pool.get("sam2")

        model.eval.assert_called_once()

    def test_model_half_called_when_fp16_cuda(self, pool):
        fp16_settings = _fake_settings(use_fp16=True, device="cuda")
        model = _mock_model()
        pool.register_loader("sam2", MagicMock(return_value=model), vram_gb=1.0)

        with patch("pipeline.pool.model_pool.get_settings", return_value=fp16_settings):
            pool.get("sam2")

        model.half.assert_called_once()

    def test_model_half_not_called_when_cpu(self, pool, settings):
        model = _mock_model()
        pool.register_loader("sam2", MagicMock(return_value=model), vram_gb=1.0)

        with patch("pipeline.pool.model_pool.get_settings", return_value=settings):
            pool.get("sam2")

        model.half.assert_not_called()


# ── _current_vram_gb ───────────────────────────────────────────────────────────


@pytest.mark.unit
class TestCurrentVramGb:
    def test_zero_when_empty(self, pool):
        assert pool._current_vram_gb() == 0.0

    def test_sums_all_entries(self, pool, settings):
        pool.register_loader("sam2", MagicMock(return_value=_mock_model()), vram_gb=2.4)
        pool.register_loader("xmem", MagicMock(return_value=_mock_model()), vram_gb=1.2)

        with patch("pipeline.pool.model_pool.get_settings", return_value=settings):
            pool.get("sam2")
            pool.get("xmem")

        assert abs(pool._current_vram_gb() - 3.6) < 1e-9

    def test_reflects_pool_contents(self, pool, settings):
        pool.register_loader("raft", MagicMock(return_value=_mock_model()), vram_gb=0.3)

        with patch("pipeline.pool.model_pool.get_settings", return_value=settings):
            pool.get("raft")

        assert pool._current_vram_gb() == 0.3


# ── LRU eviction ──────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestLruEviction:
    def test_evicts_least_recently_used(self):
        """Load A then B; touch A to make B the LRU; load C which needs eviction → B evicted."""
        tight = _fake_settings(max_vram_gb=2.5)
        pool = ModelPool()
        pool.register_loader("model_a", MagicMock(return_value=_mock_model()), vram_gb=1.0)
        pool.register_loader("model_b", MagicMock(return_value=_mock_model()), vram_gb=1.0)
        pool.register_loader("model_c", MagicMock(return_value=_mock_model()), vram_gb=1.5)

        with patch("pipeline.pool.model_pool.get_settings", return_value=tight):
            pool.get("model_a")   # pool: [a=1.0]  total=1.0
            pool.get("model_b")   # pool: [a,b=1.0 each] total=2.0
            pool.get("model_a")   # touch a → a is MRU; b is LRU
            # Loading c: 2.0+1.5=3.5 > 2.5 → evict b (LRU)
            pool.get("model_c")

        assert "model_b" not in pool._pool
        assert "model_a" in pool._pool
        assert "model_c" in pool._pool

    def test_vram_budget_respected_after_eviction(self):
        """After eviction the total VRAM must not exceed max_vram_gb."""
        tight = _fake_settings(max_vram_gb=2.5)
        pool = ModelPool()
        pool.register_loader("model_a", MagicMock(return_value=_mock_model()), vram_gb=1.0)
        pool.register_loader("model_b", MagicMock(return_value=_mock_model()), vram_gb=1.5)

        with patch("pipeline.pool.model_pool.get_settings", return_value=tight):
            pool.get("model_a")  # total=1.0
            pool.get("model_b")  # 1.0+1.5=2.5 → fits; load b

        assert pool._current_vram_gb() <= tight.max_vram_gb

    def test_multiple_evictions_when_needed(self):
        """Evict several models if a single eviction is not enough."""
        tight = _fake_settings(max_vram_gb=3.0)
        pool = ModelPool()
        pool.register_loader("a", MagicMock(return_value=_mock_model()), vram_gb=1.0)
        pool.register_loader("b", MagicMock(return_value=_mock_model()), vram_gb=1.0)
        pool.register_loader("big", MagicMock(return_value=_mock_model()), vram_gb=2.8)

        with patch("pipeline.pool.model_pool.get_settings", return_value=tight):
            pool.get("a")    # total=1.0
            pool.get("b")    # total=2.0
            # Loading big: 2.0+2.8=4.8 > 3.0 → evict a (LRU), still 1.0+2.8=3.8 > 3.0
            # → evict b (now LRU), 0+2.8=2.8 <= 3.0, load big
            pool.get("big")

        assert "a" not in pool._pool
        assert "b" not in pool._pool
        assert "big" in pool._pool


# ── stats() ───────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestStats:
    def test_pool_size_matches_loaded_models(self, pool, settings):
        pool.register_loader("sam2", MagicMock(return_value=_mock_model()), vram_gb=1.0)
        pool.register_loader("raft", MagicMock(return_value=_mock_model()), vram_gb=0.3)

        with patch("pipeline.pool.model_pool.get_settings", return_value=settings):
            pool.get("sam2")
            pool.get("raft")
            s = pool.stats()

        assert s["pool_size"] == 2
        assert set(s["loaded_models"]) == {"sam2", "raft"}

    def test_total_vram_matches_sum(self, pool, settings):
        pool.register_loader("sam2", MagicMock(return_value=_mock_model()), vram_gb=2.4)

        with patch("pipeline.pool.model_pool.get_settings", return_value=settings):
            pool.get("sam2")
            s = pool.stats()

        assert abs(s["total_vram_gb"] - 2.4) < 1e-9

    def test_max_vram_comes_from_settings(self, pool, settings):
        settings.max_vram_gb = 6.0
        with patch("pipeline.pool.model_pool.get_settings", return_value=settings):
            assert pool.stats()["max_vram_gb"] == 6.0


# ── acquire() context manager ─────────────────────────────────────────────────


@pytest.mark.unit
class TestAcquire:
    def test_yields_loaded_model(self, pool, settings):
        model = _mock_model()
        pool.register_loader("sam2", MagicMock(return_value=model), vram_gb=1.0)

        with patch("pipeline.pool.model_pool.get_settings", return_value=settings):
            with pool.acquire("sam2") as m:
                assert m is model.to.return_value

    def test_model_stays_in_pool_after_exit(self, pool, settings):
        pool.register_loader("sam2", MagicMock(return_value=_mock_model()), vram_gb=1.0)

        with patch("pipeline.pool.model_pool.get_settings", return_value=settings):
            with pool.acquire("sam2"):
                pass
            assert "sam2" in pool._pool

    def test_returns_model_context_instance(self, pool):
        pool.register_loader("raft", lambda: _mock_model(), vram_gb=0.3)
        ctx = pool.acquire("raft")
        assert isinstance(ctx, _ModelContext)

    def test_second_acquire_returns_same_model(self, pool, settings):
        loader = MagicMock(return_value=_mock_model())
        pool.register_loader("xmem", loader, vram_gb=1.0)

        with patch("pipeline.pool.model_pool.get_settings", return_value=settings):
            with pool.acquire("xmem") as m1:
                pass
            with pool.acquire("xmem") as m2:
                pass

        assert m1 is m2
        loader.assert_called_once()


# ── get_model_pool() singleton ─────────────────────────────────────────────────


@pytest.mark.unit
class TestGetModelPool:
    def test_returns_model_pool_instance(self):
        pool_module._pool_instance = None
        result = get_model_pool()
        assert isinstance(result, ModelPool)

    def test_same_instance_on_repeated_calls(self):
        pool_module._pool_instance = None
        p1 = get_model_pool()
        p2 = get_model_pool()
        assert p1 is p2

    def test_stub_loaders_registered_for_all_models(self):
        pool_module._pool_instance = None
        pool = get_model_pool()
        for name in ("sam2", "xmem", "propainter", "raft", "boundary_fusion"):
            assert name in pool._loaders

    def test_stub_loader_raises_not_implemented(self):
        pool_module._pool_instance = None
        tight = _fake_settings(max_vram_gb=99.0)
        pool = get_model_pool()
        with patch("pipeline.pool.model_pool.get_settings", return_value=tight):
            with pytest.raises(NotImplementedError, match="sam2"):
                pool.get("sam2")
