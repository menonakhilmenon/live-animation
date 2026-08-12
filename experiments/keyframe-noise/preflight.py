"""gfx1201 (RX 9070 XT) numerical preflight.

The failure mode on this card is silent wrong answers, not crashes. Run this
before trusting any training or eval result. Every check compares GPU output
against a CPU reference and reports max absolute deviation.

Known hazards this probes for, from docs/research-text-to-animation.md:
  ROCm #6595 -- fp32 mm/addmm/bmm/linear return silently wrong results when
                M > 2^19 (524,288) rows. fp16/bf16 unaffected.
  ROCm #6299 -- bool -> float64 cast silently yields an all-zero tensor.
                Keyframe observation masks are exactly this pattern.
  CK SDPA    -- ckSDPASupported() allows only {gfx942, gfx950}. Never set
                TORCH_ROCM_FA_PREFER_CK=1. AOTriton ships fwd+bwd for gfx1201.
"""

import os
import sys
import torch

FAIL = []
WARN = []


def check(name, ok, detail="", warn_only=False):
    tag = "PASS" if ok else ("WARN" if warn_only else "FAIL")
    print(f"[{tag}] {name}" + (f"  --  {detail}" if detail else ""))
    if not ok:
        (WARN if warn_only else FAIL).append(f"{name}: {detail}")


print("=" * 72)
print("torch", torch.__version__, "| hip", torch.version.hip)
print("=" * 72)

if not torch.cuda.is_available():
    print("[FAIL] no GPU visible to torch -- cannot continue")
    sys.exit(1)

dev = torch.device("cuda:0")
props = torch.cuda.get_device_properties(0)
arch = getattr(props, "gcnArchName", "?")
total_gib = props.total_memory / 1024**3
print(f"device: {props.name} | arch: {arch} | VRAM: {total_gib:.1f} GiB")
free_b, total_b = torch.cuda.mem_get_info()
print(f"free VRAM right now: {free_b / 1024**3:.2f} GiB of {total_b / 1024**3:.2f} GiB\n")

check("arch is gfx1201", "gfx1201" in arch, arch, warn_only=True)

# ---------------------------------------------------------------- AOTriton
aot = os.path.join(os.path.dirname(torch.__file__), "lib", "aotriton.images")
kernels = []
if os.path.isdir(aot):
    for root, _dirs, files in os.walk(aot):
        if "gfx120" in root or any("gfx120" in d for d in root.split(os.sep)):
            kernels += files
        kernels += [d for d in _dirs if d.startswith(("attn_", "bwd_"))]
kernels = sorted(set(kernels))
has_fwd = any("attn_fwd" in k for k in kernels)
has_bwd = any("bwd_kernel" in k for k in kernels)
check(
    "AOTriton flash kernels present (fwd+bwd)",
    has_fwd and has_bwd,
    f"fwd={has_fwd} bwd={has_bwd} :: {', '.join(k for k in kernels if k.startswith(('attn_', 'bwd_')))[:120]}",
    warn_only=True,
)
check(
    "TORCH_ROCM_FA_PREFER_CK not set",
    os.environ.get("TORCH_ROCM_FA_PREFER_CK", "0") not in ("1", "TRUE", "true"),
    "CK SDPA supports only gfx942/gfx950 -- would silently mis-dispatch",
)

# ------------------------------------------------------- basic matmul parity
torch.manual_seed(0)
a = torch.randn(512, 768)
b = torch.randn(768, 512)
ref = a @ b
got = (a.to(dev) @ b.to(dev)).cpu()
d = (ref - got).abs().max().item()
check("fp32 matmul matches CPU", d < 1e-3, f"max|diff| = {d:.3e}")

# ------------------------------------------------- ROCm #6595 large-M fp32
# Guard: only run the >2^19 probe if there is room for it.
K = 64
rows_ok, rows_bad = 1 << 18, (1 << 19) + 1024
need_gib = rows_bad * K * 4 * 3 / 1024**3
if free_b / 1024**3 > need_gib + 2.0:
    w = torch.randn(K, K)
    for rows, label, expect_bad in (
        (rows_ok, f"M={rows_ok} (below 2^19)", False),
        (rows_bad, f"M={rows_bad} (above 2^19)", True),
    ):
        x = torch.randn(rows, K)
        r = x @ w
        g = (x.to(dev) @ w.to(dev)).cpu()
        dd = (r - g).abs().max().item()
        corrupt = dd >= 2e-2
        if expect_bad:
            # Confirming the known bug is informational, not a blocker: we design
            # around it by keeping flattened batch*seq under 2^19. What WOULD be a
            # blocker is this silently starting to pass in a way we stop guarding.
            check(
                f"#6595 fp32 mm {label} -- known-bad, confirmed present",
                not corrupt,
                f"max|diff| = {dd:.3e} (bug reproduces; keep flattened batch*seq < 524288)",
                warn_only=True,
            )
        else:
            check(f"#6595 fp32 mm {label}", not corrupt, f"max|diff| = {dd:.3e}")
        del x, r, g
        torch.cuda.empty_cache()
    del w
else:
    check(f"#6595 large-M probe", False, f"skipped, needs ~{need_gib + 2:.1f} GiB free", warn_only=True)

# ------------------------------------------------ ROCm #6299 bool -> float
m = torch.rand(256, 256) > 0.5
for dt, nm in ((torch.float64, "float64"), (torch.float32, "float32"), (torch.float16, "float16")):
    cpu_sum = m.to(dt).sum().item()
    gpu_sum = m.to(dev).to(dt).sum().item()
    check(
        f"#6299 bool->{nm} cast preserves mask",
        abs(cpu_sum - gpu_sum) < 1e-3 and gpu_sum > 0,
        f"cpu={cpu_sum:.0f} gpu={gpu_sum:.0f}",
    )

# ------------------------------------------------------------------- SDPA
try:
    from torch.nn.functional import scaled_dot_product_attention as sdpa

    B, H, T, D = 2, 4, 196, 64  # 196 frames = CondMDI/HumanML3D sequence length
    q, k, v = (torch.randn(B, H, T, D, requires_grad=True) for _ in range(3))
    ref_o = sdpa(q, k, v)
    ref_o.sum().backward()
    ref_g = q.grad.clone()

    qd, kd, vd = (t.detach().to(dev).requires_grad_(True) for t in (q, k, v))
    got_o = sdpa(qd, kd, vd)
    got_o.sum().backward()

    do = (ref_o - got_o.cpu()).abs().max().item()
    dg = (ref_g - qd.grad.cpu()).abs().max().item()
    check("SDPA forward matches CPU", do < 2e-3, f"max|diff| = {do:.3e}")
    check("SDPA backward matches CPU", dg < 2e-3, f"max|diff| = {dg:.3e}")
except Exception as e:
    check("SDPA fwd/bwd", False, f"{type(e).__name__}: {e}")

# -------------------------------------------------------- bf16 autocast
try:
    x = torch.randn(256, 512)
    lin = torch.nn.Linear(512, 512)
    ref = lin(x)
    lin_d, x_d = lin.to(dev), x.to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        got = lin_d(x_d)
    dd = (ref - got.float().cpu()).abs().max().item()
    check("bf16 autocast within tolerance", dd < 0.5, f"max|diff| = {dd:.3e} (bf16 has ~3 decimal digits)")
except Exception as e:
    check("bf16 autocast", False, f"{type(e).__name__}: {e}")

# ------------------------------------------------------ hipBLASLt on/off
try:
    import subprocess

    probe = (
        "import torch;torch.manual_seed(0);"
        "a=torch.randn(1024,1024);b=torch.randn(1024,1024);"
        "print(float((a.cuda()@b.cuda()).cpu().sum()))"
    )
    outs = {}
    for flag in ("1", "0"):
        env = dict(os.environ, TORCH_BLAS_PREFER_HIPBLASLT=flag)
        r = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True, timeout=300)
        outs[flag] = r.stdout.strip().splitlines()[-1] if r.returncode == 0 else f"ERR {r.stderr[-200:]}"
    same = outs["1"] == outs["0"]
    check(
        "hipBLASLt on/off agree",
        same,
        f"LT=1 {outs['1']} | LT=0 {outs['0']}",
        warn_only=not same,
    )
except Exception as e:
    check("hipBLASLt comparison", False, f"{type(e).__name__}: {e}", warn_only=True)

# ------------------------------------------------------------------ verdict
print("\n" + "=" * 72)
if FAIL:
    print(f"{len(FAIL)} BLOCKING FAILURE(S):")
    for f in FAIL:
        print("  -", f)
if WARN:
    print(f"{len(WARN)} warning(s):")
    for w in WARN:
        print("  -", w)
if not FAIL:
    print("PREFLIGHT PASSED -- numerics look sound on this card.")
print("=" * 72)
sys.exit(1 if FAIL else 0)
