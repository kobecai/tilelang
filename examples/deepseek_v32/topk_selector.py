import torch
import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
}
_RADIX = 1 << 8


def convert_to_uint16(x):
    hval = T.cast(x, T.float16)
    bits_uint = T.reinterpret(hval, T.uint16)
    bits_uint = T.if_then_else(x < 0, ~bits_uint & (0xFFFF), bits_uint | (0x8000))
    return bits_uint >> 8


def convert_to_uint32(x):
    bits_uint = T.reinterpret(x, T.uint32)
    bits_uint = T.if_then_else(
        x < 0,
        ~bits_uint & T.cast((0xFFFFFFFF), T.uint32),
        bits_uint | T.cast((0x80000000), T.uint32),
    )
    return bits_uint


def convert_to_radix_bin(x, round):
    return T.cast((convert_to_uint32(x) >> (24 - round * 8)) & 0xFF, T.int32)


@tilelang.jit(pass_configs=pass_configs)
def tl_topk_impl(input, index, starts, ends, threads=1024, in_dtype=T.float32, out_dtype=T.int32):
    topk = T.const("topk")
    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    RADIX = _RADIX
    histogram_size = RADIX * 2 if threads == RADIX else RADIX + 1
    # Fast-path capacity for the stage-1 threshold bucket. Larger buckets
    # (issue #1351) skip shared staging and rescan global input instead.
    SMEM_INPUT_SIZE = 4096

    input: T.Tensor[(batch, seq_len), in_dtype]
    index: T.Tensor[(batch, topk), out_dtype]
    starts: T.Tensor[(batch), out_dtype]
    ends: T.Tensor[(batch), out_dtype]

    with T.Kernel(batch, threads=threads) as (bx):
        tx = T.get_thread_binding()

        s_threshold_bin_id = T.alloc_shared([1], T.int32)
        # Pad the 257 logical entries when using 256 threads so T.fill has a
        # bijective two-element-per-thread layout on the ROCm launch.
        s_histogram = T.alloc_shared([histogram_size], T.int32)
        s_num_input = T.alloc_shared([2], T.int32)
        s_input_idx = T.alloc_shared([2, SMEM_INPUT_SIZE], T.int32)

        l_threshold_bin_id = T.alloc_var(T.int32)
        l_new_topk = T.alloc_var(T.int32)
        l_num_input = T.alloc_var(T.int32)
        l_num_threshold = T.alloc_var(T.int32)
        l_stage1_bin = T.alloc_var(T.int32)
        l_prefix = T.alloc_var(T.int32)
        l_matched = T.alloc_var(T.int32)
        l_bin_id32 = T.alloc_var(T.int32)
        l_val = T.alloc_var(T.int32)
        l_start_pos = T.alloc_var(T.int32)
        l_start_idx = T.alloc_var(T.int32)
        l_end_idx = T.alloc_var(T.int32)
        l_out_pos = T.alloc_var(T.int32)

        pos = T.alloc_var(T.int32)

        l_new_topk = topk
        l_start_idx = starts[bx]
        l_end_idx = ends[bx]
        l_prefix = 0

        # stage 1: use 8bit to do quick topk
        T.fill(s_histogram, 0)
        T.fill(s_num_input[0], 0)

        # Default to bin 0 in case no threshold crossing is found, e.g. when
        # the valid input range has no more than topk elements.
        T.fill(s_threshold_bin_id, 0)

        T.sync_threads()
        for s in T.serial(T.ceildiv(seq_len, threads * 4)):
            input_base = s * threads * 4 + tx * 4
            for j in T.serial(4):
                input_idx = input_base + j
                if input_idx < l_end_idx and input_idx >= l_start_idx and input_idx < seq_len:
                    inval_int16 = convert_to_uint16(input[bx, input_idx])
                    T.atomic_add(s_histogram[inval_int16], 1)
        T.sync_threads()

        # cumsum
        if tx < RADIX:
            for i in T.serial(8):
                offset = 1 << i
                T.sync_threads(3, RADIX)
                if tx < RADIX - offset:
                    l_val = s_histogram[tx] + s_histogram[tx + offset]
                T.sync_threads(3, RADIX)
                if tx < RADIX - offset:
                    s_histogram[tx] = l_val

            # find threshold bin id
            T.sync_threads(3, RADIX)
            # s_histogram[tx] is the suffix count for bins >= tx. Use >=/< to
            # also catch the exact-boundary case where that count equals topk.
            if s_histogram[tx] >= l_new_topk and s_histogram[tx + 1] < l_new_topk:
                s_threshold_bin_id[0] = tx
        T.sync_threads()
        l_threshold_bin_id = s_threshold_bin_id[0]
        l_stage1_bin = l_threshold_bin_id
        # histogram[k] is the suffix count of bins >= k, so the threshold bin
        # occupancy is histogram[T] - histogram[T + 1].
        l_num_threshold = s_histogram[l_threshold_bin_id] - s_histogram[l_threshold_bin_id + 1]
        l_new_topk = l_new_topk - s_histogram[l_threshold_bin_id + 1]
        T.sync_threads()

        # Emit bins strictly above the coarse threshold. Compact the threshold
        # bin into shared memory only when it fits; otherwise stage 2 rescans.
        for s in T.serial(T.ceildiv(seq_len, threads * 4)):
            T.sync_threads()
            input_base = s * threads * 4 + tx * 4
            for j in T.serial(4):
                input_idx = input_base + j
                if input_idx < l_end_idx and input_idx >= l_start_idx and input_idx < seq_len:
                    bin_id = convert_to_uint16(input[bx, input_idx])
                    l_bin_id32 = T.cast(bin_id, T.int32)
                    if l_bin_id32 > l_threshold_bin_id:
                        pos = T.atomic_add(s_histogram[l_bin_id32 + 1], 1, return_prev=True)
                        index[bx, pos] = input_idx
                    elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                        if l_num_threshold <= SMEM_INPUT_SIZE:
                            pos = T.atomic_add(s_num_input[0], 1, return_prev=True)
                            s_input_idx[0, pos] = input_idx

        # stage 2: refine the coarse threshold bin
        if l_num_threshold <= SMEM_INPUT_SIZE:
            for round in T.serial(4):
                if l_new_topk <= 0:
                    break

                r_idx = round % 2
                l_start_pos = topk - l_new_topk

                T.sync_threads()
                T.fill(s_histogram, 0)
                if tx == 0:
                    s_num_input[r_idx ^ 1] = 0
                T.sync_threads()

                l_num_input = s_num_input[r_idx]
                for s in T.serial(T.ceildiv(l_num_input, threads)):
                    if s * threads + tx < l_num_input:
                        l_bin_id32 = convert_to_radix_bin(input[bx, s_input_idx[r_idx, s * threads + tx]], round)
                        T.atomic_add(s_histogram[l_bin_id32], 1)
                T.sync_threads()
                # cumsum
                if tx < RADIX:
                    for i in T.serial(8):
                        offset = 1 << i
                        T.sync_threads(3, RADIX)
                        if tx < RADIX - offset:
                            l_val = s_histogram[tx] + s_histogram[tx + offset]
                        T.sync_threads(3, RADIX)
                        if tx < RADIX - offset:
                            s_histogram[tx] = l_val

                    # find threshold bin id
                    T.sync_threads(3, RADIX)
                    # s_histogram[tx] is the suffix count for bins >= tx. Use >=/< to
                    # also catch the exact-boundary case where that count equals topk.
                    if s_histogram[tx] >= l_new_topk and s_histogram[tx + 1] < l_new_topk:
                        s_threshold_bin_id[0] = tx
                T.sync_threads()

                l_threshold_bin_id = s_threshold_bin_id[0]
                l_new_topk = l_new_topk - s_histogram[l_threshold_bin_id + 1]
                T.sync_threads()

                for s in T.serial(T.ceildiv(l_num_input, threads)):
                    T.sync_threads()
                    if s * threads + tx < l_num_input:
                        l_bin_id32 = convert_to_radix_bin(input[bx, s_input_idx[r_idx, s * threads + tx]], round)
                        if l_bin_id32 > l_threshold_bin_id:
                            pos = T.atomic_add(s_histogram[l_bin_id32 + 1], 1, return_prev=True) + l_start_pos
                            index[bx, pos] = s_input_idx[r_idx, s * threads + tx]
                        elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                            if round == 3:
                                l_out_pos = T.atomic_add(s_histogram[l_bin_id32 + 1], 1, return_prev=True) + l_start_pos
                                if l_out_pos < topk:
                                    index[bx, l_out_pos] = s_input_idx[r_idx, s * threads + tx]
                            else:
                                pos = T.atomic_add(s_num_input[r_idx ^ 1], 1, return_prev=True)
                                s_input_idx[r_idx ^ 1, pos] = s_input_idx[r_idx, s * threads + tx]
        else:
            # Threshold bucket does not fit in shared memory (issue #1351).
            # Keep the remaining set implicitly via (stage-1 bin, uint32 prefix)
            # and rescan the original input each radix round.
            for round in T.serial(4):
                if l_new_topk <= 0:
                    break

                l_start_pos = topk - l_new_topk

                T.sync_threads()
                T.fill(s_histogram, 0)
                T.fill(s_threshold_bin_id, 0)
                T.sync_threads()

                for s in T.serial(T.ceildiv(seq_len, threads * 4)):
                    input_base = s * threads * 4 + tx * 4
                    for j in T.serial(4):
                        input_idx = input_base + j
                        if input_idx < l_end_idx and input_idx >= l_start_idx and input_idx < seq_len:
                            l_matched = 1
                            if T.cast(convert_to_uint16(input[bx, input_idx]), T.int32) != l_stage1_bin:
                                l_matched = 0
                            # Nested IfStmt is required: TIR `And` would still
                            # evaluate `u32 >> 32` on round 0 (undefined shift).
                            if round > 0:  # noqa: SIM102
                                if T.cast(convert_to_uint32(input[bx, input_idx]) >> (32 - round * 8), T.int32) != l_prefix:
                                    l_matched = 0
                            if l_matched != 0:
                                l_bin_id32 = convert_to_radix_bin(input[bx, input_idx], round)
                                T.atomic_add(s_histogram[l_bin_id32], 1)
                T.sync_threads()
                # cumsum
                if tx < RADIX:
                    for i in T.serial(8):
                        offset = 1 << i
                        T.sync_threads(3, RADIX)
                        if tx < RADIX - offset:
                            l_val = s_histogram[tx] + s_histogram[tx + offset]
                        T.sync_threads(3, RADIX)
                        if tx < RADIX - offset:
                            s_histogram[tx] = l_val

                    T.sync_threads(3, RADIX)
                    if s_histogram[tx] >= l_new_topk and s_histogram[tx + 1] < l_new_topk:
                        s_threshold_bin_id[0] = tx
                T.sync_threads()

                l_threshold_bin_id = s_threshold_bin_id[0]
                l_new_topk = l_new_topk - s_histogram[l_threshold_bin_id + 1]
                T.sync_threads()

                for s in T.serial(T.ceildiv(seq_len, threads * 4)):
                    T.sync_threads()
                    input_base = s * threads * 4 + tx * 4
                    for j in T.serial(4):
                        input_idx = input_base + j
                        if input_idx < l_end_idx and input_idx >= l_start_idx and input_idx < seq_len:
                            l_matched = 1
                            if T.cast(convert_to_uint16(input[bx, input_idx]), T.int32) != l_stage1_bin:
                                l_matched = 0
                            if round > 0:  # noqa: SIM102
                                if T.cast(convert_to_uint32(input[bx, input_idx]) >> (32 - round * 8), T.int32) != l_prefix:
                                    l_matched = 0
                            if l_matched != 0:
                                l_bin_id32 = convert_to_radix_bin(input[bx, input_idx], round)
                                if l_bin_id32 > l_threshold_bin_id:
                                    pos = T.atomic_add(s_histogram[l_bin_id32 + 1], 1, return_prev=True) + l_start_pos
                                    index[bx, pos] = input_idx
                                elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                                    if round == 3:
                                        l_out_pos = T.atomic_add(s_histogram[l_bin_id32 + 1], 1, return_prev=True) + l_start_pos
                                        if l_out_pos < topk:
                                            index[bx, l_out_pos] = input_idx
                if round < 3:
                    l_prefix = l_prefix * RADIX + l_threshold_bin_id


def get_10bit_data(bs: int, seq_len: int, device=None) -> torch.Tensor:
    """Floats that share the top 22 bits and only differ in the lowest 10.

    Stage 1 only looks at 8 bits of float16, so this distribution collapses into
    a single coarse bin whose occupancy is the full sequence (issue #1351).
    """
    if device is None:
        device = "cuda"
    torch.manual_seed(42)
    top_22_bits_mask = 0xFFFFFC00
    last_10_bits_mask = 0x000003FF
    fixed_top_22_bits = 0x3F900000
    random_bottom_bits = torch.randint(0, 2**10, (bs, seq_len), dtype=torch.int32, device=device)
    score_bits = (fixed_top_22_bits & top_22_bits_mask) | (random_bottom_bits & last_10_bits_mask)
    return score_bits.view(torch.float32)


def _assert_selected_values_match(values, indexes, topk, starts, ends):
    selected_are_valid = (indexes >= starts.unsqueeze(1)) & (indexes < ends.unsqueeze(1))
    assert torch.all(selected_are_valid), "top-k returned an index outside the valid range"
    for row, row_indexes in enumerate(indexes):
        assert torch.unique(row_indexes).numel() == topk, f"top-k returned duplicate indices in row {row}"

    positions = torch.arange(values.shape[1], device=values.device).unsqueeze(0)
    valid = (positions >= starts.unsqueeze(1)) & (positions < ends.unsqueeze(1))
    ref_vals = torch.topk(values.masked_fill(~valid, float("-inf")), topk, dim=-1).values
    tl_vals = torch.gather(values, 1, indexes.to(torch.int64))
    torch.testing.assert_close(
        torch.sort(tl_vals, dim=-1, descending=True).values,
        ref_vals,
        rtol=0,
        atol=0,
    )


def tl_topk(input, starts, ends, topk):
    batch, seq_len = input.shape
    indexes = torch.zeros(batch, topk, dtype=torch.int32, device=input.device)
    # CUDA supports named barriers for the 256-thread radix subgroup. HIP
    # currently lowers that synchronization to a full workgroup barrier, so
    # launch exactly the radix participants there to keep every barrier
    # convergent without changing the tuned CUDA configuration.
    threads = _RADIX if torch.version.hip is not None else 1024
    tl_topk_impl(input, indexes, starts, ends, threads=threads)
    return indexes


def test_topk_selector(batch=64, seq_len=32 * 1024, topk=2048):
    torch.manual_seed(1)
    input = torch.randn(batch, seq_len, dtype=torch.float32).cuda()
    starts = torch.zeros(batch, dtype=torch.int32).cuda()
    ends = torch.ones(batch, dtype=torch.int32).cuda() * seq_len

    indexes = tl_topk(input, starts, ends, topk)
    print(indexes)

    indexes_ref = torch.topk(input, topk, dim=-1)[1]
    print(indexes_ref)

    # indexes_ref = fast_topk(input, topk)
    # print(indexes_ref)

    # Calculate intersection of out_ref and out_trt
    for i in range(batch):
        ref_np = indexes_ref[i].cpu().to(torch.int32).numpy()
        trt_np = indexes[i].cpu().to(torch.int32).numpy()

        set_ref = set(ref_np)
        set_trt = set(trt_np)
        intersection = set_ref & set_trt
        print("selected/all:", len(intersection), "/", len(set_ref), "=", len(intersection) / len(set_ref))

    # Performance test with CUDA events

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(5):
        _ = tl_topk(input, starts, ends, topk)
    torch.cuda.synchronize()

    n_iters = 20
    start_event.record()
    for _ in range(n_iters):
        _ = tl_topk(input, starts, ends, topk)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Average tl_topk time: {elapsed_time_ms / n_iters:.3f} ms")

    # Torch topk time
    start_event.record()
    for _ in range(n_iters):
        _ = torch.topk(input, topk, dim=-1)[1]
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Average torch.topk time: {elapsed_time_ms / n_iters:.3f} ms")


def test_topk_selector_close_values(batch=2, seq_len=32 * 1024, topk=2048):
    input = get_10bit_data(batch, seq_len)
    starts = torch.zeros(batch, dtype=torch.int32, device=input.device)
    ends = torch.full((batch,), seq_len, dtype=torch.int32, device=input.device)
    indexes = tl_topk(input, starts, ends, topk)
    _assert_selected_values_match(input, indexes, topk, starts, ends)


def run_regression_perf(batch=64, seq_len=32 * 1024, topk=2048):
    torch.manual_seed(1)
    input = torch.randn(batch, seq_len, dtype=torch.float32).cuda()
    starts = torch.zeros(batch, dtype=torch.int32).cuda()
    ends = torch.ones(batch, dtype=torch.int32).cuda() * seq_len

    from tilelang.profiler import do_bench

    def run_kernel_only():
        tl_topk(input, starts, ends, topk)

    return do_bench(run_kernel_only, backend="cupti")


if __name__ == "__main__":
    test_topk_selector()
    test_topk_selector_close_values()
