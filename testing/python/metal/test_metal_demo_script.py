import importlib.util
from pathlib import Path

import pytest


DEMO_PATH = Path(__file__).resolve().parents[3] / "examples" / "metal" / "demo_mps_matmul_profile.py"


def load_demo_module():
    spec = importlib.util.spec_from_file_location("demo_mps_matmul_profile", DEMO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load demo module from {DEMO_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_mps_event_defaults_to_synchronized_wall_clock():
    module = load_demo_module()

    calls = []
    times = iter([10.0, 10.9])

    class FakeEvent:
        def __init__(self, enable_timing=True):
            raise AssertionError("default benchmark path should not create MPS events")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(module.torch.mps, "Event", FakeEvent, raising=False)
    monkeypatch.setattr(module.torch.mps, "synchronize", lambda: calls.append("mps_sync"))
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(times))
    try:
        latency = module.benchmark_mps_event(lambda: calls.append("fn"), warmup=2, repeat=3)
    finally:
        monkeypatch.undo()

    assert latency == pytest.approx(300.0)
    assert calls.count("fn") == 5
    assert calls.count("mps_sync") == 2


def test_benchmark_mps_event_can_use_mps_events_when_requested():
    module = load_demo_module()

    calls = []

    class FakeEvent:
        def __init__(self, enable_timing=True):
            self.enable_timing = enable_timing

        def record(self):
            calls.append("record")

        def synchronize(self):
            calls.append("sync")

        def elapsed_time(self, other):
            return 12.0

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(module.torch.mps, "Event", FakeEvent, raising=False)
    monkeypatch.setattr(module.torch.mps, "synchronize", lambda: calls.append("mps_sync"))
    try:
        latency = module.benchmark_mps_event(lambda: calls.append("fn"), warmup=2, repeat=3, use_events=True)
    finally:
        monkeypatch.undo()

    assert latency == pytest.approx(4.0)
    assert calls.count("fn") == 5
    assert calls.count("record") == 2
