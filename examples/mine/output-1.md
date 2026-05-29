# python 1.py
/usr/local/lib/python3.11/site-packages/torch/cuda/__init__.py:65: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
# from tvm.script import tir as T

@T.prim_func
def gemm_128x128(A_handle: T.handle, B_handle: T.handle, C_handle: T.handle):
    A = T.match_buffer(A_handle, (128, 128), "float16", strides=(128, 1))
    B = T.match_buffer(B_handle, (128, 128), "float16", strides=(128, 1))
    C = T.match_buffer(C_handle, (128, 128), "float16", strides=(128, 1))
    # with T.block("root"):
    bx = T.launch_thread("blockIdx.x", 1)
    tx = T.launch_thread("threadIdx.x", 128)
    ty = T.launch_thread("threadIdx.y", 1)
    tz = T.launch_thread("threadIdx.z", 1)
    with T.block("tilelang_root"):
        T.reads(A[0, 0], B[0, 0], C[0, 0])
        T.writes()
        A_s = T.alloc_buffer((128, 128), "float16", scope="shared.dyn")
        B_s = T.alloc_buffer((128, 128), "float16", scope="shared.dyn")
        C_l = T.alloc_buffer((128, 128), "float16", scope="local.fragment")
        T.copy(T.region(A[0, 0], 1, 128, 128), T.region(A_s[0, 0], 2, 128, 128))
        T.copy(T.region(B[0, 0], 1, 128, 128), T.region(B_s[0, 0], 2, 128, 128))
        T.fill(T.region(C_l[0, 0], 2, 128, 128), 0)
        T.gemm(T.region(A_s[0, 0], 1, 128, 128), T.region(B_s[0, 0], 1, 128, 128), T.region(C_l[0, 0], 3, 128, 128), T.bool(False), T.bool(False), 128, 128, 128, 0, T.bool(False), 128, 128, 0, 0, 1, 0, 0, 0, 0)
        T.copy(T.region(C_l[0, 0], 1, 128, 128), T.region(C[0, 0], 2, 128, 128))
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def gemm_128x128(A_handle: T.handle, B_handle: T.handle, C_handle: T.handle):
        T.func_attr({"dyn_shared_memory_buf": 65536, "target": T.target({"arch": "sm_80", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}), "thread_extent": {"blockIdx.x": 1, "threadIdx.x": 128, "threadIdx.y": 1, "threadIdx.z": 1}, "tir.is_entry_func": True, "tl.has_tma": T.bool(False), "tl.readonly_param_indices": [0, 1], "tma_descriptor_args": {}})
        A = T.match_buffer(A_handle, (128, 128), "float16", strides=(128, 1))
        B = T.match_buffer(B_handle, (128, 128), "float16", strides=(128, 1))
        C = T.match_buffer(C_handle, (128, 128), "float16", strides=(128, 1))
        bx = T.launch_thread("blockIdx.x", 1)
        buf_dyn_shmem = T.allocate([65536], "uint8", "shared.dyn")
        C_l = T.allocate([128], "float16", "local")
        thread_binding = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        for i in T.unroll(16):
            A_s = T.Buffer((16384,), "float16", data=buf_dyn_shmem, scope="shared.dyn")
            A_1 = T.Buffer((16384,), "float16", data=A.data)
            A_s[thread_binding % 16 // 8 * 8192 + i * 512 + thread_binding // 16 * 64 + (thread_binding // 64 + thread_binding % 8 // 4) % 2 * 32 + (thread_binding % 64 // 32 + thread_binding % 4 // 2) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 2) % 2 * 8:thread_binding % 16 // 8 * 8192 + i * 512 + thread_binding // 16 * 64 + (thread_binding // 64 + thread_binding % 8 // 4) % 2 * 32 + (thread_binding % 64 // 32 + thread_binding % 4 // 2) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 2) % 2 * 8 + 8] = A_1[i * 1024 + thread_binding * 8:i * 1024 + thread_binding * 8 + 8]
        for i in T.unroll(16):
            B_s = T.Buffer((16384,), "float16", data=buf_dyn_shmem, scope="shared.dyn")
            B_1 = T.Buffer((16384,), "float16", data=B.data)
            B_s[thread_binding % 16 // 8 * 8192 + i * 512 + thread_binding // 16 * 64 + (thread_binding // 64 + thread_binding % 8 // 4) % 2 * 32 + (thread_binding % 64 // 32 + thread_binding % 4 // 2) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 2) % 2 * 8 + 16384:thread_binding % 16 // 8 * 8192 + i * 512 + thread_binding // 16 * 64 + (thread_binding // 64 + thread_binding % 8 // 4) % 2 * 32 + (thread_binding % 64 // 32 + thread_binding % 4 // 2) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 2) % 2 * 8 + 16384 + 8] = B_1[i * 1024 + thread_binding * 8:i * 1024 + thread_binding * 8 + 8]
        C_l_1 = T.Buffer((128,), "float16", data=C_l, scope="local")
        for i in T.unroll(32):
            C_l_1[i * 4:i * 4 + 4] = T.Broadcast(T.float16(0.0), 4)
        with T.attr(0, "lexical_alloc_scope", 1):
            A_local = T.allocate([32], "float16", "local")
            B_local = T.allocate([32], "float16", "local")
            T.tvm_storage_sync("shared.dyn")
            for ki in range(8):
                for i in range(4):
                    T.ptx_ldmatrix(T.bool(False), 4, T.tvm_access_ptr(T.type_annotation("float16"), buf_dyn_shmem, ki // 4 * 8192 + thread_binding % 64 // 32 * 4096 + i * 1024 + thread_binding % 16 // 8 * 512 + (thread_binding % 16 * 64 + (thread_binding % 8 // 4 + ki % 4 // 2) % 2 * 32 + (thread_binding % 4 // 2 + ki % 2) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 2) % 2 * 8) % 512, 8, 1), T.tvm_access_ptr(T.type_annotation("float16"), A_local, i * 8, 8, 2))
                for i in range(4):
                    T.ptx_ldmatrix(T.bool(True), 4, T.tvm_access_ptr(T.type_annotation("float16"), buf_dyn_shmem, thread_binding // 64 * 8192 + ki * 1024 + thread_binding % 16 // 8 * 512 + (thread_binding % 16 * 64 + (thread_binding % 8 // 4 + i // 2) % 2 * 32 + (thread_binding % 4 // 2 + i % 2) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 2) % 2 * 8) % 512 + 16384, 8, 1), T.tvm_access_ptr(T.type_annotation("float16"), B_local, i * 8, 8, 2))
                for i, j in T.grid(4, 4):
                    T.ptx_mma("float16", "m16n8k16", "row", "col", "fp16", "fp16", "fp16", A_local, i * 8, B_local, j * 8, C_l, i * 32 + j * 8, T.bool(False))
                    T.ptx_mma("float16", "m16n8k16", "row", "col", "fp16", "fp16", "fp16", A_local, i * 8, B_local, j * 8 + 4, C_l, i * 32 + j * 8 + 4, T.bool(False))
        for i in T.unroll(64):
            C_1 = T.Buffer((16384,), "float16", data=C.data)
            C_1[thread_binding % 64 // 32 * 8192 + i // 16 * 2048 + i % 2 * 1024 + thread_binding % 32 // 4 * 128 + thread_binding // 64 * 64 + i % 16 // 2 * 8 + thread_binding % 4 * 2:thread_binding % 64 // 32 * 8192 + i // 16 * 2048 + i % 2 * 1024 + thread_binding % 32 // 4 * 128 + thread_binding // 64 * 64 + i % 16 // 2 * 8 + thread_binding % 4 * 2 + 2] = C_l_1[i * 2:i * 2 + 2]