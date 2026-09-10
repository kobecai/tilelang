import sys
from pathlib import Path

import torch

import tilelang.testing


_EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "deepseek_v32"
sys.path.insert(0, str(_EXAMPLE_DIR))

from topk_selector import _assert_selected_values_match, get_10bit_data, tl_topk  # noqa: E402


@tilelang.testing.requires_cuda
def test_issue_1351_topk_selector_near_identical_values():
    """Threshold-bin occupancy can exceed the 4K shared staging buffer.

    Repro from https://github.com/tile-ai/tilelang/issues/1351: scores that
    only differ in the last 10 bits all fall into one stage-1 radix bin.
    """
    batch, seq_len, topk = 2, 32 * 1024, 2048
    values = get_10bit_data(batch, seq_len)
    starts = torch.tensor([0, 257], dtype=torch.int32, device="cuda")
    ends = torch.tensor([seq_len, seq_len - 137], dtype=torch.int32, device="cuda")

    indexes = tl_topk(values, starts, ends, topk)
    _assert_selected_values_match(values, indexes, topk, starts, ends)


if __name__ == "__main__":
    tilelang.testing.main()
