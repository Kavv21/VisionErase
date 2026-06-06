from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable

import structlog

from api.core.config import get_settings
from api.core.metrics import MODEL_LOAD_TIME, MODEL_POOL_SIZE, VRAM_USAGE

log = structlog.get_logger(__name__)


@dataclass
class ModelEntry:
    model:     Any
    name:      str
    vram_gb:   float
    loaded_at: float
    last_used: float
    use_count: int = field(default=0)


class _ModelContext:
    """Context manager returned by ModelPool.acquire().

    Yields the loaded model on __enter__; model stays warm in the pool on exit.
    """

    def __init__(self, pool: "ModelPool", name: str) -> None:
        self._pool = pool
        self._name = name

    def __enter__(self) -> Any:
        return self._pool.get(self._name)

    def __exit__(self, *_: object) -> None:
        pass  # model remains warm; no teardown needed


class ModelPool:
    """LRU memory pool for CV models.

    Keeps models resident in VRAM/RAM between inference calls.
    Evicts the least-recently-used model when the VRAM budget would be exceeded.
    Thread-safe via an internal Lock.
    """

    _VRAM_ESTIMATES: dict[str, float] = {
        "sam2":            2.4,
        "xmem":            1.2,
        "propainter":      3.1,
        "raft":            0.3,
        "boundary_fusion": 0.8,
    }

    def __init__(self) -> None:
        self._pool: OrderedDict[str, ModelEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._loaders: dict[str, Callable] = {}
        self._vram_estimates: dict[str, float] = dict(self._VRAM_ESTIMATES)

    # ── Public API ─────────────────────────────────────────────────────────────

    def register_loader(
        self,
        name: str,
        loader_fn: Callable,
        vram_gb: float | None = None,
    ) -> None:
        """Register a callable that loads and returns a ready model."""
        self._loaders[name] = loader_fn
        if vram_gb is not None:
            self._vram_estimates[name] = vram_gb

    def acquire(self, name: str) -> _ModelContext:
        """Return a context manager that yields the loaded model.

        Usage::

            with pool.acquire("sam2") as model:
                result = model.predict(frame)
        """
        return _ModelContext(self, name)

    def get(self, name: str) -> Any:
        """Return a ready model, loading it on the first call.

        Updates last_used and moves the entry to the MRU position on every call.
        """
        with self._lock:
            if name in self._pool:
                entry = self._pool[name]
                entry.last_used = time.time()
                entry.use_count += 1
                self._pool.move_to_end(name)
                return entry.model
            return self._load_model(name).model

    def stats(self) -> dict:
        """Return a snapshot of pool state for debugging and monitoring."""
        settings = get_settings()
        with self._lock:
            return {
                "loaded_models": list(self._pool.keys()),
                "total_vram_gb": self._current_vram_gb(),
                "max_vram_gb":   settings.max_vram_gb,
                "pool_size":     len(self._pool),
            }

    # ── Internal ───────────────────────────────────────────────────────────────

    def _load_model(self, name: str) -> ModelEntry:
        """Load a model, evicting LRU entries if necessary to stay within budget.

        Must be called while _lock is held.
        """
        if name not in self._loaders:
            raise KeyError(f"No loader registered for model '{name}'")

        settings = get_settings()
        vram_needed = self._vram_estimates.get(name, 2.0)

        if self._current_vram_gb() + vram_needed > settings.max_vram_gb:
            self._evict_lru(vram_needed)

        start = time.time()
        model = self._loaders[name]()

        if settings.use_fp16 and settings.device == "cuda":
            model = model.half()

        model = model.to(settings.device)
        model.eval()

        elapsed = time.time() - start
        now = time.time()

        entry = ModelEntry(
            model=model,
            name=name,
            vram_gb=vram_needed,
            loaded_at=now,
            last_used=now,
        )
        self._pool[name] = entry
        self._pool.move_to_end(name)

        log.info(
            "model_loaded",
            model=name,
            load_time_sec=round(elapsed, 3),
            vram_gb=vram_needed,
        )
        MODEL_LOAD_TIME.labels(model=name).observe(elapsed)
        MODEL_POOL_SIZE.inc()
        _update_vram_gauge()

        return entry

    def _evict_lru(self, needed_gb: float) -> None:
        """Pop LRU models until the pool has room for needed_gb.

        Must be called while _lock is held.
        """
        settings = get_settings()
        while self._pool and self._current_vram_gb() + needed_gb > settings.max_vram_gb:
            evicted_name, entry = self._pool.popitem(last=False)
            try:
                entry.model.to("cpu")
            except Exception:
                pass
            del entry
            _empty_cuda_cache()
            log.info("model_evicted", model=evicted_name)
            MODEL_POOL_SIZE.dec()

    def _current_vram_gb(self) -> float:
        return sum(e.vram_gb for e in self._pool.values())


# ── CUDA helpers ───────────────────────────────────────────────────────────────

def _empty_cuda_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _update_vram_gauge() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            VRAM_USAGE.set(torch.cuda.memory_allocated() / 1e9)
    except ImportError:
        pass


# ── Singleton ──────────────────────────────────────────────────────────────────

_pool_instance: ModelPool | None = None
_pool_lock = threading.Lock()


def get_model_pool() -> ModelPool:
    """Return the process-wide singleton ModelPool.

    Stub loaders are registered for all five standard model names so that
    callers get a clear error message (not KeyError) if weights are missing.
    """
    global _pool_instance
    if _pool_instance is not None:
        return _pool_instance
    with _pool_lock:
        if _pool_instance is None:
            pool = ModelPool()
            _register_stub_loaders(pool)
            _pool_instance = pool
    return _pool_instance


def _register_stub_loaders(pool: ModelPool) -> None:
    for name in ("sam2", "xmem", "propainter", "raft", "boundary_fusion"):
        def _stub(n: str = name) -> Any:
            raise NotImplementedError(f"Load {n} model weights first")
        pool.register_loader(name, _stub)
