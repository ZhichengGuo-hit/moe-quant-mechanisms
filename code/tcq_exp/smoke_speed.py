"""速度对比: GPU缓存prefill(默认) vs CPU批量直接INT4 prefill, 以及decode。
用于选定后续 NLL 评测的执行路径。
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
COSERVING = os.path.join(os.path.dirname(HERE), "coserving")
sys.path.insert(0, COSERVING)
os.environ["MOE_PREFILL_CPU_BATCH"] = "1"

import torch
from moe_lib import (MOE_MODEL_PATH, STORE_DIR, Int4Store, ExpertCache,
                     build_moe_model)
import int4_ext


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MOE_MODEL_PATH, trust_remote_code=True)
    exec_ext = int4_ext.load_ext(threads=16)
    store = Int4Store(STORE_DIR, mmap_mode="r")
    cache = ExpertCache(store, device="cuda", dtype=torch.float16, slots=512,
                        pin_pool=8, trace=False, prefetch=False)
    t0 = time.time()
    model = build_moe_model("gpu", MOE_MODEL_PATH, device="cuda",
                            dtype=torch.float16, store=store, cache=cache,
                            cpu_cache=None, direct_ext=exec_ext)
    print(f"[model] built {time.time()-t0:.1f}s vram={torch.cuda.memory_allocated()/2**30:.2f}GB",
          flush=True)

    text = ("Solve the following problem step by step.\n"
            "A store sells notebooks for $3 each and pens for $2 each. "
            "Mia buys 7 notebooks and 5 pens. How much does she pay in total?\n"
            "Answer: 7*3 + 5*2 = 31 dollars.\n")
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
    n = ids.shape[1]
    print(f"[data] {n} tokens", flush=True)

    ds = {"prefill_cpu": 1, "decode": 1}
    for tag, use_cpu in ds.items():
        os.environ["MOE_PREFILL_CPU_BATCH"] = "1" if use_cpu else "0"
        with torch.no_grad():
            t0 = time.time()
            if tag == "decode":
                out = model(input_ids=ids[:, :1].cuda(), use_cache=True)
                past = out.past_key_values
                for i in range(1, min(n, 33)):
                    out = model(input_ids=ids[:, i:i+1].cuda(),
                                past_key_values=past, use_cache=True)
                    past = out.past_key_values
                dt = time.time() - t0
                print(f"[{tag}] {min(n,33)-1} steps in {dt:.2f}s -> "
                      f"{1000*dt/(min(n,33)-1):.1f} ms/tok", flush=True)
            else:
                out = model(input_ids=ids.cuda(), use_cache=False)
                dt = time.time() - t0
                print(f"[{tag}] {n} tokens in {dt:.2f}s -> {n/dt:.1f} tok/s, "
                      f"vram={torch.cuda.memory_allocated()/2**30:.2f}GB", flush=True)


if __name__ == "__main__":
    main()
