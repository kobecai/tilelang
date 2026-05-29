# Metal MPS Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and verify a TileLang demo that runs on a new conda environment on Apple Silicon using Metal/MPS.

**Architecture:** Create one focused demo script under `examples/metal/` that compiles a small TileLang matmul kernel for `target="metal"` with `execution_backend="torch"`, validates against PyTorch MPS matmul, and reports latency. Add a lightweight test that validates the demo helper API without requiring a full kernel compile.

**Tech Stack:** Python 3.11, conda, PyTorch MPS, TileLang Metal target, pytest.

---

### Task 1: Demo Structure Test

**Files:**
- Create: `testing/python/metal/test_metal_demo_script.py`
- Create later: `examples/metal/demo_mps_matmul_profile.py`

- [ ] **Step 1: Write the failing test**

```python
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


def test_benchmark_mps_event_returns_average_milliseconds():
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

    fake_torch = pytest.MonkeyPatch()
    fake_torch.setattr(module.torch.mps, "Event", FakeEvent)
    fake_torch.setattr(module.torch.mps, "synchronize", lambda: calls.append("mps_sync"))
    try:
        latency = module.benchmark_mps_event(lambda: calls.append("fn"), warmup=2, repeat=3)
    finally:
        fake_torch.undo()

    assert latency == pytest.approx(4.0)
    assert calls.count("fn") == 5
    assert calls.count("record") == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest testing/python/metal/test_metal_demo_script.py -q`

Expected: FAIL because `examples/metal/demo_mps_matmul_profile.py` does not exist.

### Task 2: Metal Demo Script

**Files:**
- Create: `examples/metal/demo_mps_matmul_profile.py`

- [ ] **Step 1: Implement the demo script**

Create a script with:
- `benchmark_mps_event(fn, warmup, repeat)` using `torch.mps.Event`.
- `matmul(...)` returning a TileLang prim func with scalar inner loop, matching existing Metal tests.
- `main()` that checks MPS availability, compiles with `target="metal"` and `execution_backend="torch"`, validates output, prints generated shader length and latency.

- [ ] **Step 2: Run the structure test**

Run: `python -m pytest testing/python/metal/test_metal_demo_script.py -q`

Expected: PASS.

### Task 3: Conda Environment and Runtime Verification

**Files:**
- Modify installed environment only; no repo file changes.

- [ ] **Step 1: Create environment**

Run: `conda create -y -n tilelang-mps python=3.11`

Expected: environment appears in `conda env list`.

- [ ] **Step 2: Install project**

Run: `conda run -n tilelang-mps python -m pip install -U pip setuptools wheel cmake ninja`

Run: `conda run -n tilelang-mps python -m pip install . -v`

Expected: `import tilelang` works from the environment.

- [ ] **Step 3: Run Metal tests and demo**

Run: `conda run -n tilelang-mps python -m pytest testing/python/metal/test_metal_demo_script.py -q`

Run: `conda run -n tilelang-mps python examples/metal/demo_mps_matmul_profile.py`

Expected: correctness validation passes and latency is printed in milliseconds.
