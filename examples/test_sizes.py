#!/usr/bin/env python3
"""
Multi-size correctness sweep for Helion NKI examples.

Runs each kernel across a range of input sizes (small → large) and checks
allclose against the reference. Designed to catch size-dependent bugs like
tile boundary wrap-around or off-by-one errors that only show up at scale.

Usage:
    cd /home/ubuntu/helion_nki
    source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
    HELION_BACKEND=nki python examples/test_sizes.py

Skipped examples (require CUDA or special hardware not available on Neuron):
    blackwell_attention, flex_attention, fp8_attention, fp8_gemm, aot_example

Clean up compiled kernels after run:
    rm -rf /var/tmp/neuron-compile-cache
"""
from __future__ import annotations

import functools
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Callable

import torch

sys.path.insert(0, "/home/ubuntu/helion_nki")

from helion._testing import DEVICE

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check(name: str, got: torch.Tensor, ref: torch.Tensor, atol: float = 1e-3, rtol: float = 1e-3) -> bool:
    try:
        torch.testing.assert_close(got.float(), ref.float(), atol=atol, rtol=rtol)
        return True
    except AssertionError as e:
        lines = str(e).splitlines()
        print(f"    MISMATCH: {lines[0]}")
        return False


def _make_jagged(num_rows: int, max_cols: int, feat_dim: int | None = None, device: torch.device = DEVICE):
    lengths = torch.randint(1, max_cols + 1, (num_rows,))
    x_offsets = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(lengths, 0)])
    nnz = int(x_offsets[-1])
    if feat_dim is not None:
        x_data = torch.randn(nnz, feat_dim, dtype=torch.float32, device=device)
    else:
        x_data = torch.randn(nnz, dtype=torch.float32, device=device)
    return x_data, x_offsets.to(torch.int32).to(device)


# ---------------------------------------------------------------------------
# Test registry
# ---------------------------------------------------------------------------

@dataclass
class SizeTest:
    name: str
    sizes: list[tuple[Any, ...]]   # (label, *args)
    run_fn: Callable[..., bool]
    skip_reason: str = ""


TESTS: list[SizeTest] = []


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------
from examples.add import add as add_kernel

def _test_add(m: int, n: int) -> bool:
    x = torch.randn([m, n], device=DEVICE, dtype=torch.float16)
    y = torch.randn([m, n], device=DEVICE, dtype=torch.float16)
    return _check(f"add({m},{n})", add_kernel(x, y), torch.add(x, y))

TESTS.append(SizeTest("add", [
    ("128x128",    128,   128),
    ("1024x1024",  1024,  1024),
    ("4096x4096",  4096,  4096),
    ("10240x10240",10240, 10240),
], _test_add))


# ---------------------------------------------------------------------------
# batch_softmax
# ---------------------------------------------------------------------------
from examples.batch_softmax import batch_softmax

def _test_batch_softmax(b: int, m: int, n: int) -> bool:
    x = torch.randn([b, m, n], device=DEVICE, dtype=torch.float16)
    ref = torch.nn.functional.softmax(x, dim=-1)
    return _check(f"batch_softmax({b},{m},{n})", batch_softmax(x), ref, atol=1e-2, rtol=1e-2)

TESTS.append(SizeTest("batch_softmax", [
    ("4x128x256",   4,  128,  256),
    ("8x256x512",   8,  256,  512),
    ("16x512x1024", 16, 512,  1024),
    ("32x512x2048", 32, 512,  2048),
], _test_batch_softmax))


# ---------------------------------------------------------------------------
# bf16xint16_gemm
# ---------------------------------------------------------------------------
from examples.bf16xint16_gemm import bf16xint16_gemm, reference_bf16xint16_pytorch

def _test_bf16xint16_gemm(m: int, k: int, n: int) -> bool:
    x = torch.randn([m, k], device=DEVICE, dtype=torch.bfloat16)
    w = torch.randint(-(2**15), 2**15 - 1, (k, n), device=DEVICE, dtype=torch.int16)
    got = bf16xint16_gemm(x, w)
    ref = reference_bf16xint16_pytorch(x, w)
    return _check(f"bf16xint16_gemm({m},{k},{n})", got, ref, atol=2.0, rtol=1e-2)

TESTS.append(SizeTest("bf16xint16_gemm", [
    ("128x128x128",  128, 128, 128),
    ("256x256x256",  256, 256, 256),
    ("512x512x512",  512, 512, 512),
    ("1024x1024x1024", 1024, 1024, 1024),
], _test_bf16xint16_gemm))


# ---------------------------------------------------------------------------
# bmm
# ---------------------------------------------------------------------------
from examples.bmm import bmm as bmm_kernel

def _test_bmm(b: int, m: int, k: int, n: int) -> bool:
    x = torch.randn([b, m, k], device=DEVICE, dtype=torch.float16)
    y = torch.randn([b, k, n], device=DEVICE, dtype=torch.float16)
    return _check(f"bmm({b},{m},{k},{n})", bmm_kernel(x, y), torch.bmm(x, y), atol=1.0, rtol=1e-2)

TESTS.append(SizeTest("bmm", [
    ("2x128x128x128",   2,  128, 128, 128),
    ("4x256x256x256",   4,  256, 256, 256),
    ("8x512x512x512",   8,  512, 512, 512),
    ("16x512x768x1024", 16, 512, 768, 1024),
], _test_bmm))


# ---------------------------------------------------------------------------
# broadcast_matmul
# ---------------------------------------------------------------------------
from examples.broadcast_matmul import broadcast_matmul

def _test_broadcast_matmul(b: int, m: int, k: int, n: int) -> bool:
    x = torch.randn([b, m, k], device=DEVICE, dtype=torch.float16)
    w = torch.randn([k, n], device=DEVICE, dtype=torch.float16)
    return _check(f"broadcast_matmul({b},{m},{k},{n})", broadcast_matmul(x, w), torch.matmul(x, w), atol=1.0, rtol=1e-2)

TESTS.append(SizeTest("broadcast_matmul", [
    ("4x128x256x128",   4,  128, 256, 128),
    ("8x256x512x256",   8,  256, 512, 256),
    ("16x512x768x1024", 16, 512, 768, 1024),
    ("32x512x768x1024", 32, 512, 768, 1024),
], _test_broadcast_matmul))


# ---------------------------------------------------------------------------
# concatenate
# ---------------------------------------------------------------------------
from examples.concatenate import concat2d_dim1

def _test_concatenate(rows: int, c1: int, c2: int) -> bool:
    x = torch.randn([rows, c1], device=DEVICE)
    y = torch.randn([rows, c2], device=DEVICE)
    ref = torch.cat([x, y], dim=1)
    return _check(f"concat2d_dim1({rows},{c1},{c2})", concat2d_dim1(x, y), ref)

TESTS.append(SizeTest("concatenate", [
    ("256x128x128",   256,  128,  128),
    ("1024x256x256",  1024, 256,  256),
    ("1500x400x600",  1500, 400,  600),
    ("4096x512x512",  4096, 512,  512),
], _test_concatenate))


# ---------------------------------------------------------------------------
# cross_entropy
# ---------------------------------------------------------------------------
from examples.cross_entropy import cross_entropy as cross_entropy_kernel

def _test_cross_entropy(n: int, vocab: int) -> bool:
    logits = torch.randn(n, vocab, device=DEVICE, dtype=torch.float32)
    labels = torch.randint(0, vocab, (n,), device=DEVICE, dtype=torch.long)
    got = cross_entropy_kernel(logits, labels)
    ref = torch.nn.functional.cross_entropy(logits, labels, reduction="mean")
    return _check(f"cross_entropy({n},{vocab})", got, ref)

TESTS.append(SizeTest("cross_entropy", [
    ("128x1024",    128,  1024),
    ("512x8192",    512,  8192),
    ("1024x32768",  1024, 32768),
    ("1024x131072", 1024, 131072),
], _test_cross_entropy))


# ---------------------------------------------------------------------------
# embedding
# ---------------------------------------------------------------------------
from examples.embedding import embedding as embedding_kernel

def _test_embedding(num_emb: int, emb_dim: int, batch: int, seq: int) -> bool:
    x = torch.randint(0, num_emb, [batch, seq], device=DEVICE, dtype=torch.int32)
    w = torch.randn([num_emb, emb_dim], device=DEVICE)
    got = embedding_kernel(x, w)
    ref = torch.nn.functional.embedding(x, w)
    return _check(f"embedding({num_emb},{emb_dim},{batch},{seq})", got, ref, atol=0.0, rtol=0.0)

TESTS.append(SizeTest("embedding", [
    ("16x64_256x32",   16,  64,  256, 32),
    ("64x128_512x32",  64,  128, 512, 32),
    ("256x256_1024x64",256, 256, 1024, 64),
    ("1024x512_2048x64",1024,512, 2048, 64),
], _test_embedding))


# ---------------------------------------------------------------------------
# exp
# ---------------------------------------------------------------------------
from examples.exp import exp as exp_kernel

def _test_exp(n: int) -> bool:
    x = torch.randn(n, device=DEVICE, dtype=torch.float32)
    return _check(f"exp({n})", exp_kernel(x), torch.exp(x))

TESTS.append(SizeTest("exp", [
    ("4096",   4096),
    ("16384",  16384),
    ("65536",  65536),
    ("262144", 262144),
], _test_exp))


# ---------------------------------------------------------------------------
# fused_linear_jsd
# ---------------------------------------------------------------------------
from examples.fused_linear_jsd import fused_linear_jsd_fwd, fused_linear_jsd_pytorch

def _test_fused_linear_jsd(m: int, n: int, k: int) -> bool:
    student_input  = torch.rand([m, n], device=DEVICE, dtype=torch.float)
    teacher_input  = torch.rand([m, n], device=DEVICE, dtype=torch.float)
    student_weight = torch.rand([k, n], device=DEVICE, dtype=torch.float)
    teacher_weight = torch.rand([k, n], device=DEVICE, dtype=torch.float)
    args = (0.5, -100, 1.0, student_weight, teacher_weight, student_input, teacher_input)
    got = fused_linear_jsd_fwd(*args)
    ref = fused_linear_jsd_pytorch(*args)
    return _check(f"fused_linear_jsd({m},{n},{k})", got, ref, atol=1e-2, rtol=1e-2)

TESTS.append(SizeTest("fused_linear_jsd", [
    ("32x64x4096",   32,  64,  4096),
    ("64x128x8192",  64,  128, 8192),
    ("64x128x16384", 64,  128, 16384),
    ("128x256x16384",128, 256, 16384),
], _test_fused_linear_jsd))


# ---------------------------------------------------------------------------
# fused_nki_ops
# ---------------------------------------------------------------------------
from examples.fused_nki_ops import relu_and_sum, bias_scale_add

def _test_fused_nki_ops(m: int, n: int) -> bool:
    x = torch.randn([m, n], device=DEVICE, dtype=torch.float32)
    got_r = relu_and_sum(x)                           # returns [m, 1]
    ref_r = torch.relu(x).sum(dim=-1, keepdim=True)  # match [m, 1]
    if not _check(f"relu_and_sum({m},{n})", got_r, ref_r):
        return False
    bias_val = 0.5
    scale = 2.0
    y = torch.randn([m, n], device=DEVICE, dtype=torch.float32)
    got_b = bias_scale_add(x, bias_val, scale, y)    # bias is a float scalar
    ref_b = (x + bias_val) * scale + y
    return _check(f"bias_scale_add({m},{n})", got_b, ref_b)

TESTS.append(SizeTest("fused_nki_ops", [
    ("64x128",   64,  128),
    ("128x256",  128, 256),
    ("512x512",  512, 512),
    ("1024x1024",1024,1024),
], _test_fused_nki_ops))


# ---------------------------------------------------------------------------
# gather_gemv
# ---------------------------------------------------------------------------
from examples.gather_gemv import gather_gemv

def _ref_gather_gemv(w: torch.Tensor, idx: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.stack([w[i.item()].to(x.dtype) @ x for i in idx], dim=0)

def _test_gather_gemv(B: int, S: int, N: int) -> bool:
    w   = torch.randn((B, S, S), device=DEVICE, dtype=torch.float16)
    idx = torch.randint(0, B, [N], device=DEVICE, dtype=torch.int32)
    x   = torch.randn((S,), device=DEVICE, dtype=torch.float16)
    got = gather_gemv(w, idx, x)
    ref = _ref_gather_gemv(w, idx, x)
    return _check(f"gather_gemv(B={B},S={S},N={N})", got, ref, atol=1e-1, rtol=1e-2)

TESTS.append(SizeTest("gather_gemv", [
    ("B4_S512_N2",   4,  512,  2),
    ("B8_S1024_N2",  8,  1024, 2),
    ("B8_S2048_N2",  8,  2048, 2),
    ("B8_S4096_N2",  8,  4096, 2),
], _test_gather_gemv))


# ---------------------------------------------------------------------------
# geglu
# ---------------------------------------------------------------------------
from examples.geglu import geglu as geglu_kernel
import torch.nn as nn

def _test_geglu(shape: tuple) -> bool:
    a = torch.randn(shape, device=DEVICE, dtype=torch.float16)
    b = torch.randn(shape, device=DEVICE, dtype=torch.float16)
    got = geglu_kernel(a, b)
    ref = nn.functional.gelu(a, approximate="tanh").to(b.dtype) * b
    return _check(f"geglu{shape}", got, ref, atol=1e-2, rtol=1e-2)

TESTS.append(SizeTest("geglu", [
    ("64x512",   (64,  512)),
    ("128x1024", (128, 1024)),
    ("2x2048x1024", (2, 2048, 1024)),
    ("4x2048x2048", (4, 2048, 2048)),
], _test_geglu))


# ---------------------------------------------------------------------------
# grouped_gemm
# ---------------------------------------------------------------------------
from examples.grouped_gemm import (
    grouped_gemm_jagged_example,
    _reference_grouped_gemm,
)

def _test_grouped_gemm(G: int, K: int, N: int) -> bool:
    dtype  = torch.bfloat16
    group_A = [torch.randn(64 * (i + 1), K, device=DEVICE, dtype=dtype).contiguous() for i in range(G)]
    group_B = [torch.randn(K, N, device=DEVICE, dtype=dtype).contiguous()] * G
    got = grouped_gemm_jagged_example(group_A, group_B)
    ref = _reference_grouped_gemm(group_A, group_B)
    ok = True
    for i, (g, r) in enumerate(zip(got, ref)):
        if not _check(f"grouped_gemm(G={G},K={K},N={N}) group{i}", g, r, atol=1e-1, rtol=1e-2):
            ok = False
    return ok

TESTS.append(SizeTest("grouped_gemm", [
    ("G4_K256_N128", 4, 256, 128),
    ("G4_K512_N256", 4, 512, 256),
    ("G8_K512_N256", 8, 512, 256),
    ("G8_K1024_N512",8, 1024, 512),
], _test_grouped_gemm))


# ---------------------------------------------------------------------------
# grpo_loss
# ---------------------------------------------------------------------------
from examples.grpo_loss import helion_grpo_loss, torch_grpo_loss

def _test_grpo_loss(B: int, L: int, V: int) -> bool:
    torch.manual_seed(42)
    logits1 = torch.randn(B, L + 1, V, device=DEVICE, dtype=torch.bfloat16, requires_grad=False)
    logits_ref = logits1.detach().clone().float()
    completion_ids = torch.randint(0, V - 1, (B, L), dtype=torch.int64, device=DEVICE)
    completion_mask = torch.ones_like(completion_ids, dtype=torch.float32)
    ref_logp  = torch.randn(B, L, device=DEVICE, dtype=torch.float32)
    old_logp  = torch.randn(B, L, device=DEVICE, dtype=torch.float32)
    advantages = torch.randn(B, device=DEVICE, dtype=torch.float32)
    kw = dict(temperature=0.9, beta=0.2, eps_low=0.2, eps_high=0.4)
    loss_h, kl_h, _ = helion_grpo_loss(logits1, old_logp, ref_logp, completion_ids, advantages, completion_mask, **kw)
    loss_r, kl_r, _ = torch_grpo_loss(logits_ref, old_logp, ref_logp, completion_ids, advantages, completion_mask, **kw)
    ok = _check(f"grpo_loss({B},{L},{V}) loss", loss_h, loss_r, atol=1e-1, rtol=1e-1)
    ok = _check(f"grpo_loss({B},{L},{V}) kl",   kl_h,   kl_r,   atol=1e-1, rtol=1e-1) and ok
    return ok

TESTS.append(SizeTest("grpo_loss", [
    ("B4_L128_V1024",  4, 128, 1024),
    ("B8_L128_V12800", 8, 128, 12800),
    ("B8_L512_V12800", 8, 512, 12800),
    ("B8_L1024_V12800",8, 1024,12800),
], _test_grpo_loss))


# ---------------------------------------------------------------------------
# jagged_dense_add
# ---------------------------------------------------------------------------
from examples.jagged_dense_add import (
    jagged_dense_add_2d,
    jagged_dense_add_2d_reference,
    random_jagged_2d,
)

def _test_jagged_dense_add(num_rows: int, max_cols: int) -> bool:
    torch.manual_seed(42)
    xd, xo = random_jagged_2d(num_rows, max_cols, device=DEVICE)
    y = torch.randn([num_rows, max_cols], device=DEVICE)
    got = jagged_dense_add_2d(xd, xo, y)
    torch.manual_seed(42)
    xd2, xo2 = random_jagged_2d(num_rows, max_cols, device=DEVICE)
    y2 = torch.randn([num_rows, max_cols], device=DEVICE)
    ref = jagged_dense_add_2d_reference(xd2, xo2, y2)
    return _check(f"jagged_dense_add({num_rows},{max_cols})", got, ref)

TESTS.append(SizeTest("jagged_dense_add", [
    ("64x128",   64,   128),
    ("128x128",  128,  128),
    ("128x5000", 128,  5000),
    ("256x5000", 256,  5000),
    ("512x5000", 512,  5000),
], _test_jagged_dense_add))


# ---------------------------------------------------------------------------
# jagged_layer_norm
# ---------------------------------------------------------------------------
from examples.jagged_layer_norm import (
    jagged_layer_norm_kernel,
    reference_jagged_layer_norm_pytorch,
)

def _test_jagged_layer_norm(B: int, max_seqlen: int, M: int) -> bool:
    torch.manual_seed(42)
    seq_lengths = torch.randint(1, max_seqlen + 1, (B,))
    x_offsets = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(seq_lengths, 0)])
    x_data = torch.randn(int(x_offsets[-1]), M, dtype=torch.float32, device=DEVICE)
    xo = x_offsets.to(torch.int32).to(DEVICE)
    got = jagged_layer_norm_kernel(x_data, xo)
    ref = reference_jagged_layer_norm_pytorch(x_data, xo)
    return _check(f"jagged_layer_norm(B={B},seq={max_seqlen},M={M})", got, ref, atol=1e-3, rtol=1e-3)

TESTS.append(SizeTest("jagged_layer_norm", [
    ("B32_seq64_M64",   32,  64,  64),
    ("B64_seq128_M128", 64,  128, 128),
    ("B128_seq256_M256",128, 256, 256),
    ("B256_seq512_M512",256, 512, 512),
], _test_jagged_layer_norm))


# ---------------------------------------------------------------------------
# jagged_mean
# ---------------------------------------------------------------------------
from examples.jagged_mean import (
    jagged_mean_kernel,
    reference_jagged_mean_kernel_pytorch,
)

def _test_jagged_mean(num_rows: int, max_cols: int, M: int) -> bool:
    torch.manual_seed(42)
    xd, xo = _make_jagged(num_rows, max_cols, M)
    fc = torch.randint(1, M + 1, (num_rows,), dtype=torch.int32, device=DEVICE)
    got = jagged_mean_kernel(xd, xo, fc, M)
    ref = reference_jagged_mean_kernel_pytorch(xd, xo, fc, M)
    return _check(f"jagged_mean({num_rows},{max_cols},M={M})", got, ref)

TESTS.append(SizeTest("jagged_mean", [
    ("32x64_M64",   32,  64,  64),
    ("64x64_M64",   64,  64,  64),
    ("128x128_M64", 128, 128, 64),
    ("256x256_M128",256, 256, 128),
], _test_jagged_mean))


# ---------------------------------------------------------------------------
# jagged_softmax
# ---------------------------------------------------------------------------
from examples.jagged_softmax import (
    jagged_softmax_kernel,
    reference_jagged_softmax_pytorch,
)

def _test_jagged_softmax(num_rows: int, max_cols: int, M: int) -> bool:
    torch.manual_seed(42)
    xd, xo = _make_jagged(num_rows, max_cols, M)
    got = jagged_softmax_kernel(xd, xo)
    ref = reference_jagged_softmax_pytorch(xd, xo)
    return _check(f"jagged_softmax({num_rows},{max_cols},M={M})", got, ref, atol=1e-4, rtol=1e-4)

TESTS.append(SizeTest("jagged_softmax", [
    ("32x32_M64",    32,  32,  64),   # matches run_nki_examples default
    ("64x64_M64",    64,  64,  64),
    ("256x64_M64",   256, 64,  64),
    ("512x128_M128", 512, 128, 128),
], _test_jagged_softmax))


# ---------------------------------------------------------------------------
# jagged_sum
# ---------------------------------------------------------------------------
from examples.jagged_sum import (
    jagged_sum_kernel,
    reference_jagged_sum_kernel_pytorch,
)

def _test_jagged_sum(num_rows: int, max_cols: int, M: int) -> bool:
    torch.manual_seed(42)
    xd, xo = _make_jagged(num_rows, max_cols, M)
    got = jagged_sum_kernel(xd, xo)
    ref = reference_jagged_sum_kernel_pytorch(xd, xo)
    return _check(f"jagged_sum({num_rows},{max_cols},M={M})", got, ref)

TESTS.append(SizeTest("jagged_sum", [
    ("32x64_M64",   32,  64,  64),
    ("64x64_M64",   64,  64,  64),
    ("128x64_M64",  128, 64,  64),
    ("256x128_M128",256, 128, 128),
], _test_jagged_sum))


# ---------------------------------------------------------------------------
# jsd
# ---------------------------------------------------------------------------
from examples.jsd import jsd_forward, HelionJSD, TorchJSDBaseline

def _test_jsd(BT: int, V: int) -> bool:
    torch.manual_seed(42)
    log_q = torch.randn(BT, V, device=DEVICE).log_softmax(dim=-1).requires_grad_(True)
    log_p = torch.randn(BT, V, device=DEVICE).log_softmax(dim=-1)
    shift_labels = torch.randint(0, V, (BT,), device=DEVICE)
    beta, ignore_index = 0.5, -100
    got = HelionJSD(beta=beta, ignore_index=ignore_index)(log_q, log_p, shift_labels)
    ref = TorchJSDBaseline(beta=beta, ignore_index=ignore_index)(log_q.detach(), log_p, shift_labels)
    return _check(f"jsd(BT={BT},V={V})", got, ref, atol=1e-2, rtol=1e-2)

TESTS.append(SizeTest("jsd", [
    ("BT256_V1024",   256,  1024),
    ("BT1024_V8192",  1024, 8192),
    ("BT2048_V32768", 2048, 32768),
    ("BT4096_V65536", 4096, 65536),
], _test_jsd))


# ---------------------------------------------------------------------------
# kl_div
# ---------------------------------------------------------------------------
from examples.kl_div import kl_div_forward, HelionKLDivLoss

def _test_kl_div(BT: int, V: int) -> bool:
    torch.manual_seed(42)
    inp = torch.randn(BT, V, device=DEVICE).log_softmax(dim=-1).requires_grad_(True)
    tgt = torch.randn(BT, V, device=DEVICE).softmax(dim=-1)

    helion_loss = HelionKLDivLoss(reduction="batchmean", log_target=False)
    ref_loss_fn = torch.nn.KLDivLoss(reduction="batchmean", log_target=False)

    got = helion_loss(inp, tgt)
    ref = ref_loss_fn(inp.detach(), tgt)
    return _check(f"kl_div(BT={BT},V={V})", got, ref, atol=1e-2, rtol=1e-2)

TESTS.append(SizeTest("kl_div", [
    ("BT256_V1024",   256,  1024),
    ("BT1024_V8192",  1024, 8192),
    ("BT2048_V32768", 2048, 32768),
    ("BT4096_V65536", 4096, 65536),
], _test_kl_div))


# ---------------------------------------------------------------------------
# layer_norm
# ---------------------------------------------------------------------------
from examples.layer_norm import layer_norm as layer_norm_kernel

def _test_layer_norm(batch: int, dim: int) -> bool:
    x = torch.randn([batch, dim], device=DEVICE, dtype=torch.float16)
    weight = torch.randn([dim], device=DEVICE, dtype=torch.float16)
    bias = torch.randn([dim], device=DEVICE, dtype=torch.float16)
    eps = 1e-4
    got = layer_norm_kernel(x, [dim], weight, bias, eps)  # returns single tensor via autograd Function
    ref = torch.nn.functional.layer_norm(x, [dim], weight, bias, eps)
    return _check(f"layer_norm({batch},{dim})", got, ref, atol=1e-2, rtol=1e-2)

TESTS.append(SizeTest("layer_norm", [
    ("128x512",   128,  512),
    ("512x1024",  512,  1024),
    ("1024x4096", 1024, 4096),
    ("4096x8192", 4096, 8192),
], _test_layer_norm))


# ---------------------------------------------------------------------------
# layer_norm_f32
# ---------------------------------------------------------------------------
from examples.layer_norm_f32 import layer_norm as layer_norm_f32_kernel

def _test_layer_norm_f32(batch: int, dim: int) -> bool:
    x = torch.randn([batch, dim], device=DEVICE, dtype=torch.float32)
    weight = torch.randn([dim], device=DEVICE, dtype=torch.float32)
    bias = torch.randn([dim], device=DEVICE, dtype=torch.float32)
    eps = 1e-5
    got = layer_norm_f32_kernel(x, [dim], weight, bias, eps)  # autograd Function, returns single tensor
    ref = torch.nn.functional.layer_norm(x, [dim], weight, bias, eps)
    return _check(f"layer_norm_f32({batch},{dim})", got, ref, atol=5e-4, rtol=5e-4)

TESTS.append(SizeTest("layer_norm_f32", [
    ("128x512",   128,  512),
    ("512x1024",  512,  1024),
    ("1024x1024", 1024, 1024),
    ("2048x2048", 2048, 2048),
], _test_layer_norm_f32))


# ---------------------------------------------------------------------------
# long_sum
# ---------------------------------------------------------------------------
from examples.long_sum import longsum, baseline_sum

def _test_long_sum(m: int, n: int) -> bool:
    x = torch.randn([m, n], device=DEVICE, dtype=torch.float32)
    return _check(f"long_sum({m},{n})", longsum(x), baseline_sum(x), atol=1e-2, rtol=1e-2)

TESTS.append(SizeTest("long_sum", [
    ("4x16384",  4, 16384),
    ("4x32768",  4, 32768),
    ("4x131072", 4, 131072),
    ("8x131072", 8, 131072),
], _test_long_sum))


# ---------------------------------------------------------------------------
# low_mem_dropout
# ---------------------------------------------------------------------------
from examples.low_mem_dropout import low_mem_dropout, low_mem_dropout_bwd

def _test_low_mem_dropout(size: int) -> bool:
    x = torch.randn(size=(size,), device=DEVICE)
    p, seed = 0.25, 123
    out = low_mem_dropout(p, x, seed)
    grad_y = torch.ones_like(x)
    grad_x = low_mem_dropout_bwd(p, grad_y, seed)
    mask_fwd = out != 0
    mask_bwd = grad_x != 0
    if not torch.equal(mask_fwd, mask_bwd):
        print(f"    MISMATCH: fwd/bwd masks differ for size={size}")
        return False
    return True

TESTS.append(SizeTest("low_mem_dropout", [
    ("2048",   2048),
    ("8192",   8192),
    ("32768",  32768),
    ("131072", 131072),
], _test_low_mem_dropout))


# ---------------------------------------------------------------------------
# mamba2_chunk_scan
# ---------------------------------------------------------------------------
from examples.mamba2_chunk_scan import helion_mamba2_chunk_scan_kernel, ref_chunk_scan

def _test_mamba2_chunk_scan(batch: int, nheads: int, seqlen: int, chunk_size: int, headdim: int, dstate: int) -> bool:
    from examples.mamba2_chunk_scan import test as mamba2_scan_test
    # Use the example's own test() which builds numerically valid inputs via init strings
    try:
        mamba2_scan_test("zrzzzzr", batch, nheads, 1, seqlen, chunk_size, headdim, dstate)
        return True
    except AssertionError as e:
        print(f"    MISMATCH: {e}")
        return False

TESTS.append(SizeTest("mamba2_chunk_scan", [
    ("b2_h4_s256_hd4_ds4",    2, 4, 256,  128, 4,  4),
    ("b2_h4_s512_hd64_ds64",  2, 4, 512,  128, 64, 64),
    ("b4_h8_s1024_hd64_ds64", 4, 8, 1024, 128, 64, 64),
    ("b4_h8_s2048_hd64_ds64", 4, 8, 2048, 128, 64, 64),
], _test_mamba2_chunk_scan))


# ---------------------------------------------------------------------------
# mamba2_chunk_state
# ---------------------------------------------------------------------------
from examples.mamba2_chunk_state import helion_mamba2_chunk_state_kernel, ref_chunk_state

def _test_mamba2_chunk_state(batch: int, nheads: int, seqlen: int, chunk_size: int, headdim: int, dstate: int) -> bool:
    from examples.mamba2_chunk_state import test as mamba2_state_test
    try:
        mamba2_state_test("uuuu", batch, nheads, 1, seqlen, chunk_size, headdim, dstate)
        return True
    except AssertionError as e:
        print(f"    MISMATCH: {e}")
        return False

TESTS.append(SizeTest("mamba2_chunk_state", [
    ("b2_h4_s256_hd32_ds32",  2, 4, 256,  128, 32,  32),
    ("b2_h4_s512_hd64_ds64",  2, 4, 512,  128, 64,  64),
    ("b4_h8_s1024_hd64_ds64", 4, 8, 1024, 128, 64,  64),
    ("b4_h8_s2048_hd64_ds64", 4, 8, 2048, 128, 64,  64),
], _test_mamba2_chunk_state))


# ---------------------------------------------------------------------------
# matmul
# ---------------------------------------------------------------------------
from examples.matmul import matmul as matmul_kernel

def _test_matmul(m: int, k: int, n: int) -> bool:
    x = torch.randn([m, k], device=DEVICE, dtype=torch.float16)
    y = torch.randn([k, n], device=DEVICE, dtype=torch.float16)
    return _check(f"matmul({m},{k},{n})", matmul_kernel(x, y), torch.matmul(x, y), atol=1.0, rtol=1e-2)

TESTS.append(SizeTest("matmul", [
    ("128x128x128",    128,  128,  128),
    ("512x512x512",    512,  512,  512),
    ("1024x1024x1024", 1024, 1024, 1024),
    ("2048x2048x2048", 2048, 2048, 2048),
], _test_matmul))


# ---------------------------------------------------------------------------
# matmul_layernorm
# ---------------------------------------------------------------------------
from examples.matmul_layernorm import matmul_layernorm, matmul_layernorm_pytorch

def _test_matmul_layernorm(m: int, k: int, n: int) -> bool:
    x      = torch.randn([m, k], device=DEVICE, dtype=torch.float16)
    y      = torch.randn([k, n], device=DEVICE, dtype=torch.float16)
    weight = torch.randn([n], device=DEVICE, dtype=torch.float16)
    bias   = torch.randn([n], device=DEVICE, dtype=torch.float16)
    got = matmul_layernorm(x, y, weight, bias)
    ref = matmul_layernorm_pytorch(x, y, weight, bias)
    return _check(f"matmul_layernorm({m},{k},{n})", got, ref, atol=1e-1, rtol=1e-1)

TESTS.append(SizeTest("matmul_layernorm", [
    ("64x64x128",   64,  64,  128),
    ("128x128x256", 128, 128, 256),
    ("256x256x512", 256, 256, 512),
    ("512x512x1024",512, 512, 1024),
], _test_matmul_layernorm))


# ---------------------------------------------------------------------------
# moe_matmul_ogs
# ---------------------------------------------------------------------------
from examples.moe_matmul_ogs import (
    moe_matmul_ogs,
    moe_matmul_ogs_helion_kernel_args_gen,
    moe_matmul_ogs_reference,
)

def _test_moe_matmul_ogs(T: int, K: int, N: int, E: int) -> bool:
    dtype = torch.float16
    A = torch.randn(T, K, device=DEVICE, dtype=dtype)
    W = torch.randn(E, K, N, device=DEVICE, dtype=dtype)
    top1 = torch.randint(E, (T,), device=DEVICE)
    kernel_args = moe_matmul_ogs_helion_kernel_args_gen(A, W, top1)
    got = moe_matmul_ogs(*kernel_args)
    ref = moe_matmul_ogs_reference(A, W, top1)
    return _check(f"moe_matmul_ogs(T={T},K={K},N={N},E={E})", got, ref, atol=1e-1, rtol=1e-1)

TESTS.append(SizeTest("moe_matmul_ogs", [
    ("T256_K128_N64_E8",    256,  128, 64,  8),
    ("T512_K256_N128_E16",  512,  256, 128, 16),
    ("T1024_K512_N256_E30", 1024, 512, 256, 30),
    ("T2048_K512_N256_E30", 2048, 512, 256, 30),
], _test_moe_matmul_ogs))


# ---------------------------------------------------------------------------
# psum_reuse_minimal
# ---------------------------------------------------------------------------
from examples.psum_reuse_minimal import mm_relu_kernel

def _test_psum_reuse_minimal(m: int, n: int) -> bool:
    x = torch.randn([m, n], device=DEVICE, dtype=torch.float32)
    y = torch.randn([m, n], device=DEVICE, dtype=torch.float32)
    got = mm_relu_kernel(x, y)
    ref = torch.relu(x @ y)
    return _check(f"mm_relu_kernel({m},{n})", got, ref, atol=1.0, rtol=1e-2)

TESTS.append(SizeTest("psum_reuse_minimal", [
    ("128x128", 128, 128),  # kernel has static_shapes=True fixed to 128x128
], _test_psum_reuse_minimal))


# ---------------------------------------------------------------------------
# rms_norm
# ---------------------------------------------------------------------------
from examples.rms_norm import rms_norm as rms_norm_kernel

def _test_rms_norm(m: int, n: int) -> bool:
    x = torch.randn([m, n], device=DEVICE, dtype=torch.float16)
    weight = torch.randn([n], device=DEVICE, dtype=torch.float16)
    got = rms_norm_kernel(x, weight)
    ref = torch.nn.functional.rms_norm(x, [n], weight)
    return _check(f"rms_norm({m},{n})", got, ref, atol=1e-2, rtol=1e-2)

TESTS.append(SizeTest("rms_norm", [
    ("128x512",   128,  512),
    ("1024x2048", 1024, 2048),
    ("2048x4096", 2048, 4096),
    ("2048x8192", 2048, 8192),
], _test_rms_norm))


# ---------------------------------------------------------------------------
# segment_reduction
# ---------------------------------------------------------------------------
from examples.segment_reduction import segmented_reduction_helion, segmented_reduction_pytorch

def _test_segment_reduction(num_nodes: int, num_edges: int, num_features: int) -> bool:
    indices = torch.randint(0, num_nodes, (num_edges,), device=DEVICE).sort()[0]
    data = torch.randn(num_edges, num_features, device=DEVICE, dtype=torch.float32)
    got = segmented_reduction_helion(indices, data, num_nodes)
    ref = segmented_reduction_pytorch(indices, data, num_nodes)
    return _check(f"segment_reduction(nodes={num_nodes},edges={num_edges},feat={num_features})", got, ref)

TESTS.append(SizeTest("segment_reduction", [
    ("n64_e512_f64",    64,  512,  64),
    ("n100_e2000_f128", 100, 2000, 128),
    ("n256_e8192_f256", 256, 8192, 256),
    ("n512_e16384_f256",512, 16384,256),
], _test_segment_reduction))


# ---------------------------------------------------------------------------
# softmax
# ---------------------------------------------------------------------------
from examples.softmax import softmax as softmax_kernel

def _test_softmax(m: int, n: int) -> bool:
    x = torch.randn([m, n], device=DEVICE, dtype=torch.float16)
    return _check(f"softmax({m},{n})", softmax_kernel(x), torch.nn.functional.softmax(x, dim=1), atol=1e-2, rtol=1e-2)

TESTS.append(SizeTest("softmax", [
    ("128x128",   128,  128),
    ("1024x2560", 1024, 2560),
    ("4096x2560", 4096, 2560),
    ("8192x4096", 8192, 4096),
], _test_softmax))


# ---------------------------------------------------------------------------
# softmax_decomposed  (runs on XLA, skip here — it writes a file and runs XLA)
# ---------------------------------------------------------------------------
TESTS.append(SizeTest("softmax_decomposed", [], lambda: True,
    skip_reason="writes kernel file + runs on XLA; tested via run_nki_examples.py"))


# ---------------------------------------------------------------------------
# split_k_barrier
# ---------------------------------------------------------------------------
from examples.split_k_barrier import split_k_matmul

def _test_split_k_barrier(m: int, k: int, n: int) -> bool:
    a = torch.randn(m, k, device=DEVICE)
    b = torch.randn(n, k, device=DEVICE).T
    got = split_k_matmul(a, b)
    ref = torch.matmul(a, b)
    return _check(f"split_k_barrier({m},{k},{n})", got, ref, atol=1.0, rtol=1e-2)

TESTS.append(SizeTest("split_k_barrier", [
    ("16x4096x16",  16, 4096,  16),
    ("32x4096x32",  32, 4096,  32),
    ("64x8192x64",  64, 8192,  64),
    ("128x8192x128",128, 8192, 128),
], _test_split_k_barrier))


# ---------------------------------------------------------------------------
# squeeze_and_excitation_net
# ---------------------------------------------------------------------------
from examples.squeeze_and_excitation_net import (
    squeeze_and_excitation_net,
    squeeze_and_excitation_net_pytorch,
)

def _test_squeeze_and_excitation(m: int, k: int, n: int) -> bool:
    x = torch.randn([m, n], device=DEVICE, dtype=torch.float16)
    a = torch.randn([n, k], device=DEVICE, dtype=torch.float16)
    b = torch.randn([k, n], device=DEVICE, dtype=torch.float16)
    got = squeeze_and_excitation_net(x, a, b)
    ref = squeeze_and_excitation_net_pytorch(x, a, b)
    return _check(f"squeeze_and_excitation({m},{k},{n})", got, ref, atol=1e-1, rtol=1e-1)

TESTS.append(SizeTest("squeeze_and_excitation_net", [
    ("256x256x256",   256,  256,  256),
    ("512x512x512",   512,  512,  512),
    ("1024x512x1024", 1024, 512,  1024),
    ("1024x1024x1024",1024, 1024, 1024),
], _test_squeeze_and_excitation))


# ---------------------------------------------------------------------------
# sum
# ---------------------------------------------------------------------------
from examples.sum import sum_kernel

def _test_sum(m: int, n: int) -> bool:
    x = torch.randn([m, n], device=DEVICE, dtype=torch.float32)
    return _check(f"sum({m},{n})", sum_kernel(x), x.sum(-1), atol=1e-2, rtol=1e-2)

TESTS.append(SizeTest("sum", [
    ("128x512",    128,   512),
    ("1024x2560",  1024,  2560),
    ("5120x2560",  5120,  2560),
    ("10240x10240",10240, 10240),
], _test_sum))


# ---------------------------------------------------------------------------
# swiglu
# ---------------------------------------------------------------------------
from examples.swiglu import swiglu as swiglu_kernel
import torch.nn.functional as F

def _test_swiglu(shape: tuple) -> bool:
    a = torch.randn(shape, device=DEVICE, dtype=torch.float16)
    b = torch.randn(shape, device=DEVICE, dtype=torch.float16)
    got = swiglu_kernel(a, b)
    ref = F.silu(a) * b
    return _check(f"swiglu{shape}", got, ref, atol=1e-2, rtol=1e-2)

TESTS.append(SizeTest("swiglu", [
    ("128x512",      (128, 512)),
    ("512x1024",     (512, 1024)),
    ("2x2048x1024",  (2, 2048, 1024)),
    ("4x2048x2048",  (4, 2048, 2048)),
], _test_swiglu))


# ---------------------------------------------------------------------------
# welford
# ---------------------------------------------------------------------------
from examples.welford import welford as welford_kernel, eager_layer_norm

def _test_welford(s: int, d: int) -> bool:
    weight = torch.rand((d,), device=DEVICE, dtype=torch.float32)
    bias   = torch.rand((d,), device=DEVICE, dtype=torch.float32)
    x      = torch.rand((s, d), device=DEVICE, dtype=torch.float32)
    got    = welford_kernel(weight, bias, x)
    ref    = eager_layer_norm(weight, bias, x)
    return _check(f"welford({s},{d})", got, ref, atol=1e-3, rtol=1e-3)

TESTS.append(SizeTest("welford", [
    ("512x512",   512,  512),
    ("2048x1024", 2048, 1024),
    ("4096x1536", 4096, 1536),
    ("4096x2048", 4096, 2048),
], _test_welford))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    passed = 0
    failed = 0
    errors = 0
    failures: list[str] = []

    for test in TESTS:
        if test.skip_reason:
            print(f"\n[SKIP] {test.name}: {test.skip_reason}")
            continue
        print(f"\n{'='*55}")
        print(f"  {test.name}")
        print(f"{'='*55}")
        for size_entry in test.sizes:
            label = size_entry[0]
            args  = size_entry[1:]
            tag   = f"{test.name}/{label}"
            try:
                ok = test.run_fn(*args)
                if ok:
                    print(f"  PASS  {label}")
                    passed += 1
                else:
                    print(f"  FAIL  {label}")
                    failed += 1
                    failures.append(tag)
            except Exception:
                print(f"  ERROR {label}")
                traceback.print_exc()
                errors += 1
                failures.append(f"{tag} [ERROR]")

    print(f"\n{'='*55}")
    print(f"Results: {passed} passed, {failed} failed, {errors} errors")
    if failures:
        print("Failures:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("All size tests passed.")


if __name__ == "__main__":
    main()
