#!/usr/bin/env python3
"""V1 audit for the DSpark ("DeepSeek-V4-Flash-0731") 3-stage MTP sidecar GGUF.

Cross-checks the produced sidecar (gguf-py reader) against the safetensors
index/headers of the source checkpoint and the naming/quantization rules in
dspark-design.md + dspark-reference-spec.md:

  1. Tensor inventory: every expected tensor present exactly once, nothing
     unexpected, correct GGUF dtype, correct shape (MXFP4 logical shape,
     ggml-reversed dim order).
  2. GGUF metadata: deepseek4.nextn_predict_layers=3 and the four new
     deepseek4.dspark.* keys, with correct values.
  3. mtp.2.confidence_head.* is absent from the sidecar (dropped by design).
  4. Bit-exact spot check: repacks one routed expert per stage/projection
     straight from the source safetensors bytes (independently of the cached
     conversion run) and diffs it against the on-disk MXFP4 bytes.
  5. Numeric sanity check: independently dequantizes a couple of FP8-sourced
     dense tensors (blockwise e8m0 scale) and compares against the Q8_0
     tensors actually written, expecting small (quantization-noise-level)
     relative error.

Usage:
    scripts/audit-dspark-sidecar.py \
        --gguf /home/astra/ml-local/models/DeepSeek-V4-Flash-0731-GGUF/mtp-DeepSeek-V4-Flash-0731-MXFP4_MOE.gguf \
        --src-dir /home/astra/ml-local/models/DeepSeek-V4-Flash-0731-mtp-src
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "gguf-py"))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import gguf  # noqa: E402
from gguf.gguf_reader import GGUFReader  # noqa: E402
from gguf.utility import SafetensorsLocal  # noqa: E402

from conversion.deepseek import DeepseekV4Model  # noqa: E402

MT = gguf.MODEL_TENSOR
TN = gguf.TENSOR_NAMES

FAILURES: list[str] = []
WARNINGS: list[str] = []


def fail(msg: str) -> None:
    FAILURES.append(msg)
    print(f"  FAIL: {msg}")


def warn(msg: str) -> None:
    WARNINGS.append(msg)
    print(f"  WARN: {msg}")


def ok(msg: str) -> None:
    print(f"  ok:   {msg}")


# --------------------------------------------------------------------------
# Source-of-truth loading: raw safetensors headers (no tensor data materialized
# except for the small spot-checks in sections 4/5).
# --------------------------------------------------------------------------

LOCAL_SHARDS = [
    "model-00001-of-00048.safetensors",
    "model-00045-of-00048.safetensors",
    "model-00046-of-00048.safetensors",
    "model-00047-of-00048.safetensors",
    "model-00048-of-00048.safetensors",
]


def load_source_headers(src_dir: Path) -> dict[str, tuple[str, tuple[int, ...], Path]]:
    """name -> (safetensors dtype string, shape, shard path)"""
    headers: dict[str, tuple[str, tuple[int, ...], Path]] = {}
    for shard in LOCAL_SHARDS:
        path = src_dir / shard
        st = SafetensorsLocal(path)
        for name, t in st.tensors.items():
            headers[name] = (t.dtype, t.shape, path)
    return headers


def ggml_shape(pt_shape: tuple[int, ...]) -> list[int]:
    """GGUFReader reports ne[] in ggml order = reversed PyTorch shape."""
    return list(reversed(pt_shape))


def expected_dense_dtype(src_dtype: str, ndim: int) -> str:
    """Mirrors DeepseekV4Model.tensor_force_quant()'s decision for every
    non-expert tensor in this checkpoint: 1D always F32; FP8-sourced 2D+
    dequantized then Q8_0; BF16-sourced 2D+ stays BF16; F32-sourced 2D+
    stays F32."""
    if ndim <= 1:
        return "F32"
    if src_dtype == "F8_E4M3":
        return "Q8_0"
    if src_dtype == "BF16":
        return "BF16"
    if src_dtype == "F32":
        return "F32"
    raise ValueError(f"Unhandled source dtype {src_dtype!r} for a {ndim}D tensor")


# --------------------------------------------------------------------------
# 1. Build the expected tensor inventory
# --------------------------------------------------------------------------

# (raw mtp.N. suffix, MODEL_TENSOR key, gguf suffix) for the tensors that
# exist identically in every one of the 3 DSpark stages (the "generic layer
# fallback" path in filter_tensors -> _map_dsv4_tensor_name's layer_map).
PER_STAGE_SKELETON: list[tuple[str, "gguf.MODEL_TENSOR", str]] = [
    ("attn_norm.weight", MT.ATTN_NORM, ".weight"),
    ("attn.wq_a.weight", MT.ATTN_Q_A, ".weight"),
    ("attn.wq_b.weight", MT.ATTN_Q_B, ".weight"),
    ("attn.wkv.weight", MT.ATTN_KV, ".weight"),
    ("attn.q_norm.weight", MT.ATTN_Q_A_NORM, ".weight"),
    ("attn.kv_norm.weight", MT.ATTN_KV_NORM, ".weight"),
    ("attn.wo_a.weight", MT.ATTN_OUT_A, ".weight"),
    ("attn.wo_b.weight", MT.ATTN_OUT_B, ".weight"),
    ("attn.attn_sink", MT.ATTN_SINKS, ".weight"),
    ("ffn_norm.weight", MT.FFN_NORM, ".weight"),
    ("ffn.gate.weight", MT.FFN_GATE_INP, ".weight"),
    ("ffn.gate.bias", MT.FFN_EXP_PROBS_B, ".bias"),
    ("ffn.shared_experts.w1.weight", MT.FFN_GATE_SHEXP, ".weight"),
    ("ffn.shared_experts.w2.weight", MT.FFN_DOWN_SHEXP, ".weight"),
    ("ffn.shared_experts.w3.weight", MT.FFN_UP_SHEXP, ".weight"),
    ("hc_attn_fn", MT.HC_ATTN_FN, ".weight"),
    ("hc_attn_base", MT.HC_ATTN_BASE, ".weight"),
    ("hc_attn_scale", MT.HC_ATTN_SCALE, ".weight"),
    ("hc_ffn_fn", MT.HC_FFN_FN, ".weight"),
    ("hc_ffn_base", MT.HC_FFN_BASE, ".weight"),
    ("hc_ffn_scale", MT.HC_FFN_SCALE, ".weight"),
]

EXPERT_PROJ_TO_TENSOR = {
    "w1": MT.FFN_GATE_EXP,
    "w2": MT.FFN_DOWN_EXP,
    "w3": MT.FFN_UP_EXP,
}

N_STAGES = 3
MAIN_LAYERS = 43
N_EXPERTS = 256

ROOT_TENSORS = {
    "embed.weight": (MT.TOKEN_EMBD, ".weight"),
    "norm.weight": (MT.OUTPUT_NORM, ".weight"),
    "head.weight": (MT.OUTPUT, ".weight"),
}

DSPARK_ROOT_TENSORS = {
    MT.NEXTN_HC_HEAD_FN: ("mtp.2.hc_head_fn", None),
    MT.NEXTN_HC_HEAD_BASE: ("mtp.2.hc_head_base", None),
    MT.NEXTN_HC_HEAD_SCALE: ("mtp.2.hc_head_scale", None),
    MT.NEXTN_HEAD_NORM: ("mtp.2.norm.weight", None),
    MT.NEXTN_MARKOV_W1: ("mtp.2.markov_head.markov_w1.weight", None),
    MT.NEXTN_MARKOV_W2: ("mtp.2.markov_head.markov_w2.weight", None),
}


class ExpectedTensor:
    __slots__ = ("name", "dtype", "shape", "src_name")

    def __init__(self, name: str, dtype: str, shape: list[int] | None, src_name: str | None):
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.src_name = src_name


def build_expected(headers: dict[str, tuple[str, tuple[int, ...], Path]]) -> dict[str, ExpectedTensor]:
    expected: dict[str, ExpectedTensor] = {}

    def add(name: str, dtype: str, shape: list[int] | None, src_name: str | None):
        if name in expected:
            raise AssertionError(f"expected-tensor table itself has a duplicate: {name!r}")
        expected[name] = ExpectedTensor(name, dtype, shape, src_name)

    # --- root trunk-required tensors (embed/norm/head), duplicated per the
    # existing --mtp sidecar convention (unchanged from the pre-DSpark code) ---
    for src_suffix, (tkey, tsuffix) in ROOT_TENSORS.items():
        src_dtype, src_shape, _ = headers[src_suffix]
        name = TN[tkey] + tsuffix
        add(name, expected_dense_dtype(src_dtype, len(src_shape)), ggml_shape(src_shape), src_suffix)

    # --- per-stage generic skeleton + routed experts ---
    for stage in range(N_STAGES):
        bid = MAIN_LAYERS + stage
        for suffix, tkey, tsuffix in PER_STAGE_SKELETON:
            src_name = f"mtp.{stage}.{suffix}"
            src_dtype, src_shape, _ = headers[src_name]
            name = TN[tkey].format(bid=bid) + tsuffix
            add(name, expected_dense_dtype(src_dtype, len(src_shape)), ggml_shape(src_shape), src_name)

        for proj, tkey in EXPERT_PROJ_TO_TENSOR.items():
            # every expert must have identical shape; use expert 0 as the reference
            src_name0 = f"mtp.{stage}.ffn.experts.0.{proj}.weight"
            _, w_shape, _ = headers[src_name0]  # packed I8 shape, e.g. [2048, 2048]
            out_features, packed_cols = w_shape
            logical_in = packed_cols * 2
            stacked_logical = (N_EXPERTS, out_features, logical_in)
            name = TN[tkey].format(bid=bid) + ".weight"
            add(name, "MXFP4", ggml_shape(stacked_logical), f"mtp.{stage}.ffn.experts.*.{proj}")

    # --- stage-0-only: main_norm / main_proj ---
    for suffix, tkey in (("main_norm.weight", MT.NEXTN_MAIN_NORM), ("main_proj.weight", MT.NEXTN_MAIN_PROJ)):
        src_name = f"mtp.0.{suffix}"
        src_dtype, src_shape, _ = headers[src_name]
        name = TN[tkey].format(bid=MAIN_LAYERS) + ".weight"
        add(name, expected_dense_dtype(src_dtype, len(src_shape)), ggml_shape(src_shape), src_name)

    # --- root-level DSpark-owned tensors (stage 2) ---
    for tkey, (src_name, _unused) in DSPARK_ROOT_TENSORS.items():
        src_dtype, src_shape, _ = headers[src_name]
        name = TN[tkey] + ".weight"
        add(name, expected_dense_dtype(src_dtype, len(src_shape)), ggml_shape(src_shape), src_name)

    return expected


# --------------------------------------------------------------------------
# 2 + 3. Inventory + confidence_head-absence + metadata checks
# --------------------------------------------------------------------------

def audit_inventory(reader: GGUFReader, expected: dict[str, ExpectedTensor]) -> None:
    print("\n=== 1. Tensor inventory ===")
    actual = {t.name: t for t in reader.tensors}

    dup_check: dict[str, int] = {}
    for t in reader.tensors:
        dup_check[t.name] = dup_check.get(t.name, 0) + 1
    dupes = {n: c for n, c in dup_check.items() if c > 1}
    if dupes:
        fail(f"duplicate tensor names in GGUF: {dupes}")
    else:
        ok(f"no duplicate tensor names ({len(reader.tensors)} tensors total)")

    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))

    if missing:
        fail(f"{len(missing)} expected tensor(s) missing from GGUF: {missing[:20]}{' ...' if len(missing) > 20 else ''}")
    else:
        ok(f"all {len(expected)} expected tensors present")

    if extra:
        fail(f"{len(extra)} unexpected tensor(s) in GGUF: {extra[:20]}{' ...' if len(extra) > 20 else ''}")
    else:
        ok("no unexpected tensors present")

    n_dtype_bad = 0
    n_shape_bad = 0
    for name, exp in expected.items():
        act = actual.get(name)
        if act is None:
            continue
        if act.tensor_type.name != exp.dtype:
            fail(f"{name}: dtype {act.tensor_type.name} != expected {exp.dtype} (source {exp.src_name})")
            n_dtype_bad += 1
        if exp.shape is not None and list(act.shape) != exp.shape:
            fail(f"{name}: shape {list(act.shape)} != expected {exp.shape} (source {exp.src_name})")
            n_shape_bad += 1
    if n_dtype_bad == 0:
        ok("all present tensors have the expected GGUF dtype")
    if n_shape_bad == 0:
        ok("all present tensors have the expected shape")


def audit_confidence_head_absent(reader: GGUFReader) -> None:
    print("\n=== 2. confidence_head dropped ===")
    hits = [t.name for t in reader.tensors if "confidence" in t.name.lower()]
    if hits:
        fail(f"confidence_head tensor(s) present (should be dropped): {hits}")
    else:
        ok("no confidence_head tensor in the sidecar")


def audit_metadata(reader: GGUFReader, hparams: dict) -> None:
    print("\n=== 3. Metadata ===")
    arch = "deepseek4"

    def get(key: str):
        field = reader.get_field(key)
        return field.contents() if field is not None else None

    expect_kv = {
        f"{arch}.nextn_predict_layers": 3,
        f"{arch}.dspark.target_layer_ids": list(hparams["dspark_target_layer_ids"]),
        f"{arch}.dspark.noise_token_id": hparams["dspark_noise_token_id"],
        f"{arch}.dspark.block_size": hparams["dspark_block_size"],
        f"{arch}.dspark.markov_rank": hparams["dspark_markov_rank"],
    }
    for key, expected_val in expect_kv.items():
        actual_val = get(key)
        if actual_val is None:
            fail(f"metadata key {key!r} missing")
        elif list(actual_val) if isinstance(actual_val, list) else actual_val != (list(expected_val) if isinstance(expected_val, list) else expected_val):
            if isinstance(expected_val, list) and list(actual_val) == list(expected_val):
                ok(f"{key} = {actual_val}")
            else:
                fail(f"metadata key {key!r} = {actual_val!r}, expected {expected_val!r}")
        else:
            ok(f"{key} = {actual_val}")

    # required-tensor sanity: the C++ loader unconditionally needs these three
    for name in ("token_embd.weight", "output_norm.weight", "output.weight"):
        if reader.get_tensor is not None:
            pass
    names = {t.name for t in reader.tensors}
    for name in ("token_embd.weight", "output_norm.weight", "output.weight"):
        if name in names:
            ok(f"required trunk tensor {name} present")
        else:
            fail(f"required trunk tensor {name} MISSING (C++ loader requires this unconditionally)")


# --------------------------------------------------------------------------
# 4. MXFP4 bit-exact spot check
# --------------------------------------------------------------------------

def local_tensor_to_torch(st: SafetensorsLocal, name: str, torch_dtype: torch.dtype) -> torch.Tensor:
    lt = st.tensors[name]
    raw = np.array(lt.mmap_bytes())  # copy out of the mmap so torch.frombuffer is safe
    t = torch.frombuffer(bytearray(raw), dtype=torch_dtype)
    return t.reshape(lt.shape)


def audit_mxfp4_bitexact(reader: GGUFReader, src_dir: Path) -> None:
    print("\n=== 4. MXFP4 bit-exact spot check (expert 0, per stage x projection) ===")
    actual = {t.name: t for t in reader.tensors}
    shard_cache: dict[str, SafetensorsLocal] = {}

    def shard_for(name: str) -> SafetensorsLocal:
        # cheap linear scan across the 5 local shards' index; fine for a spot check
        for shard in LOCAL_SHARDS:
            key = f"{src_dir}/{shard}"
            if key not in shard_cache:
                shard_cache[key] = SafetensorsLocal(src_dir / shard)
            if name in shard_cache[key].tensors:
                return shard_cache[key]
        raise KeyError(name)

    for stage in range(N_STAGES):
        bid = MAIN_LAYERS + stage
        for proj, tkey in EXPERT_PROJ_TO_TENSOR.items():
            w_name = f"mtp.{stage}.ffn.experts.0.{proj}.weight"
            s_name = f"mtp.{stage}.ffn.experts.0.{proj}.scale"
            st_w = shard_for(w_name)
            st_s = shard_for(s_name)
            weight = local_tensor_to_torch(st_w, w_name, torch.uint8)
            scale = local_tensor_to_torch(st_s, s_name, torch.uint8)

            expected_packed = DeepseekV4Model._pack_mxfp4_blocks(weight, scale)

            gguf_name = TN[tkey].format(bid=bid) + ".weight"
            t = actual.get(gguf_name)
            if t is None:
                fail(f"{gguf_name}: cannot spot-check, tensor missing")
                continue

            n_experts = N_EXPERTS
            out_features, packed_bytes_per_row = expected_packed.shape
            full = np.asarray(t.data)
            expected_data_shape = (n_experts, out_features, packed_bytes_per_row)
            if full.shape != expected_data_shape:
                fail(f"{gguf_name}: raw data array shape {full.shape} != expected {expected_data_shape} "
                     f"(GGUFReader axis-order assumption may be wrong - see script comment)")
                continue
            actual_expert0 = full[0]

            if actual_expert0.shape != expected_packed.shape:
                fail(f"{gguf_name}: expert-0 packed shape {actual_expert0.shape} != recomputed {expected_packed.shape}")
                continue

            if np.array_equal(actual_expert0, expected_packed):
                ok(f"{gguf_name}: expert 0 MXFP4 bytes bit-exact vs. fresh repack from source ({actual_expert0.nbytes} bytes)")
            else:
                n_diff = int(np.sum(actual_expert0 != expected_packed))
                fail(f"{gguf_name}: expert 0 MXFP4 bytes differ from fresh repack in {n_diff}/{actual_expert0.size} bytes")


# --------------------------------------------------------------------------
# 5. Q8_0 numeric sanity check (independent FP8 dequant vs. on-disk Q8_0)
# --------------------------------------------------------------------------

def independent_fp8_dequant(st: SafetensorsLocal, weight_name: str) -> np.ndarray:
    scale_name = weight_name.removesuffix(".weight") + ".scale"
    w_lt = st.tensors[weight_name]
    s_lt = st.tensors[scale_name]

    w_raw = np.array(w_lt.mmap_bytes())
    w = torch.frombuffer(bytearray(w_raw), dtype=torch.float8_e4m3fn).reshape(w_lt.shape).float().numpy()

    s_raw = np.array(s_lt.mmap_bytes())
    s_bits = torch.frombuffer(bytearray(s_raw), dtype=torch.uint8).reshape(s_lt.shape).numpy()
    s = np.exp2(s_bits.astype(np.float32) - 127.0)  # E8M0: unsigned power-of-two exponent

    out_features, in_features = w.shape
    s = np.repeat(s, 128, axis=0)[:out_features]
    s = np.repeat(s, 128, axis=1)[:, :in_features]
    return w * s


def audit_q8_0_numeric(reader: GGUFReader, src_dir: Path) -> None:
    print("\n=== 5. Q8_0 numeric sanity check (independent FP8 dequant vs. on-disk Q8_0) ===")
    actual = {t.name: t for t in reader.tensors}

    samples = [
        ("mtp.0.attn.wkv.weight", "blk.43.attn_kv.weight", "model-00046-of-00048.safetensors"),
        ("mtp.0.main_proj.weight", "blk.43.nextn.main_proj.weight", "model-00046-of-00048.safetensors"),
        ("mtp.2.ffn.shared_experts.w2.weight", "blk.45.ffn_down_shexp.weight", "model-00048-of-00048.safetensors"),
    ]

    for src_name, gguf_name, shard in samples:
        st = SafetensorsLocal(src_dir / shard)
        expected_fp32 = independent_fp8_dequant(st, src_name)

        t = actual.get(gguf_name)
        if t is None:
            fail(f"{gguf_name}: cannot spot-check, tensor missing")
            continue

        # gguf.quants.dequantize() returns the same [out_features, in_features]
        # (PyTorch-natural) orientation as the source weight - no transpose needed
        # (verified empirically: t.data is byte-shaped [out, in_bytes], and
        # dequantize_rows() expands the last axis back to logical in_features).
        actual_fp32 = gguf.quants.dequantize(np.asarray(t.data), gguf.GGMLQuantizationType.Q8_0)

        if actual_fp32.shape != expected_fp32.shape:
            fail(f"{gguf_name}: dequant shape {actual_fp32.shape} != source shape {expected_fp32.shape}")
            continue

        # Per-element relative error blows up near zero-crossings (Q8_0 rounds
        # tiny per-block values to 0, which is a huge *relative* but tiny
        # *absolute* error - not a bug). Use full-scale-normalized error
        # instead: absolute error as a fraction of the tensor's own peak
        # magnitude, which is the standard way to size quantization noise.
        abs_err = np.abs(actual_fp32.astype(np.float64) - expected_fp32.astype(np.float64))
        scale = float(np.max(np.abs(expected_fp32))) or 1.0
        norm_err = abs_err / scale
        max_norm = float(np.max(norm_err))
        mean_norm = float(np.mean(norm_err))
        frac_gt_25pct_rel = float(np.mean(abs_err > 0.25 * np.maximum(np.abs(expected_fp32), 1e-6)))

        # Q8_0 is an 8-bit per-32-value-block quantizer; a small full-scale
        # error is expected and healthy. Flag only gross mismatches, which
        # would indicate a wiring bug (wrong tensor, transpose, or
        # scale-application error) rather than ordinary quantization noise.
        if max_norm > 0.05:
            fail(f"{gguf_name}: max full-scale-normalized error {max_norm:.4f} (mean {mean_norm:.4f}) "
                 f"vs. independent FP8 dequant - too large, likely a wiring bug")
        else:
            ok(f"{gguf_name}: Q8_0 vs. independent FP8 dequant - full-scale error max {max_norm:.4f}, "
               f"mean {mean_norm:.5f} ({frac_gt_25pct_rel:.1%} of elements >25% relative error, "
               f"expected near zero-crossings)")


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument("--src-dir", type=Path, required=True)
    ap.add_argument("--skip-spot-checks", action="store_true", help="skip sections 4/5 (they re-read raw safetensors bytes)")
    args = ap.parse_args()

    with open(args.src_dir / "config.json", "r", encoding="utf-8") as f:
        hparams = json.load(f)

    print(f"Loading GGUF: {args.gguf}")
    reader = GGUFReader(str(args.gguf))
    print(f"  {len(reader.tensors)} tensors, {len(reader.fields)} metadata fields")

    print(f"\nLoading source safetensors headers from: {args.src_dir}")
    headers = load_source_headers(args.src_dir)
    print(f"  {len(headers)} tensors indexed across {len(LOCAL_SHARDS)} local shards")

    expected = build_expected(headers)
    print(f"  {len(expected)} tensors expected in the sidecar")

    audit_inventory(reader, expected)
    audit_confidence_head_absent(reader)
    audit_metadata(reader, hparams)

    if not args.skip_spot_checks:
        audit_mxfp4_bitexact(reader, args.src_dir)
        audit_q8_0_numeric(reader, args.src_dir)

    print("\n=== Summary ===")
    if FAILURES:
        print(f"FAIL: {len(FAILURES)} failure(s), {len(WARNINGS)} warning(s)")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    else:
        print(f"PASS: 0 failures, {len(WARNINGS)} warning(s)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
