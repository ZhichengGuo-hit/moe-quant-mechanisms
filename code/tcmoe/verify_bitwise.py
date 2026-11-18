#!/usr/bin/env python3
"""Bit-exact check: patched experts read back by llama-tcmoe-verify vs moe_lib reference."""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "coserving"))
sys.path.insert(0, str(ROOT / "tcq_exp"))

from moe_lib import quantize_int4, dequant_int4  # noqa: E402

PROJ_NAMES = ("gate", "up", "down")


def reference(expert_reader, layer, proj, expert, moe_inter):
    if proj == "down":
        w = expert_reader.expert(layer, "down_proj", expert)
    else:
        gu = expert_reader.expert(layer, "gate_up_proj", expert)
        w = gu[:moe_inter] if proj == "gate" else gu[moe_inter:]
    t = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32))
    packed, scale = quantize_int4(t)
    return dequant_int4(packed, scale).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=str(HERE / "out" / "test_patch.gguf"))
    ap.add_argument("--model-dir", default=str(ROOT.parent / "model" / "Qwen3.6-35B-A3B"))
    ap.add_argument("--sens", default=str(ROOT / "tcq_exp" / "results" / "sens_matrix.json"))
    ap.add_argument("--exe", default=str(ROOT / "llama.cpp" / "build-tcmoe" / "bin" / "Release" / "llama-tcmoe-verify.exe"))
    ap.add_argument("--domains", default="global,math,code")
    ap.add_argument("--per-layer", type=int, default=17, help="experts sampled per layer (17x2x3 = 102 dumps)")
    ap.add_argument("--moe-inter", type=int, default=512)
    args = ap.parse_args()

    from pack_tcmoe import ExpertReader, domain_score, select_per_layer

    sm = __import__("json").load(open(args.sens, encoding="utf-8"))
    s = np.array(sm["s"], dtype=np.float64)
    n_layers = s.shape[0]
    domains = [d.strip() for d in args.domains.split(",")]

    reader = ExpertReader(args.model_dir)

    ok_total = 0
    fail_total = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for dom in domains:
            masks = select_per_layer(domain_score(s, sm["domains"], dom), 0.25)
            samples = []
            for l in (0, 1):
                for e in masks[l][:args.per_layer]:
                    for p, name in enumerate(PROJ_NAMES):
                        samples.append((l, name, int(e)))

            sample_path = tmp / f"sample_{dom}.txt"
            with open(sample_path, "w", encoding="utf-8") as f:
                for l, name, e in samples:
                    f.write(f"{l} {name} {e}\n")

            out_path = tmp / f"out_{dom}.bin"
            cmd = [args.exe, "-m", args.gguf, "--sample", str(sample_path),
                   "--domain", dom, "-o", str(out_path)]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                print(res.stdout, res.stderr)
                return 1

            blob = np.fromfile(out_path, dtype=np.float32)
            manifest = [ln.split() for ln in open(str(out_path) + ".manifest", encoding="utf-8")
                        if not ln.startswith("#")]
            miss = sum(1 for m in manifest if m[0] == "miss")
            if miss:
                print(f"[{dom}] ERROR: {miss} patched experts reported as missing")
                return 1

            for l, name, e in samples:
                row = [m for m in manifest if m[0] == "ok" and int(m[3]) == e and int(m[2]) == PROJ_NAMES.index(name) and int(m[1]) == l]
                assert len(row) == 1, row
                n0, n1, off = int(row[0][5]), int(row[0][6]), int(row[0][7])
                got = blob[off // 4: off // 4 + n0 * n1].reshape(n1, n0)
                ref = reference(reader, l, name, e, args.moe_inter)
                assert ref.shape == got.shape, (ref.shape, got.shape)
                bad = int(np.count_nonzero(got != ref))
                if bad:
                    dmax = float(np.max(np.abs(got - ref)))
                    print(f"[{dom}] MISMATCH layer {l} {name} expert {e}: {bad}/{got.size} values, max |d| {dmax:g}")
                    fail_total += 1
                else:
                    ok_total += 1

            print(f"[{dom}] {len(samples)} dumps checked", flush=True)

    print(f"\nRESULT: {ok_total} dumps bit-exact, {fail_total} mismatched")
    return 0 if fail_total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
