# Quick check: candidate vs stock, N tokens each: same token ids? rough speed (scores come from rsi_mlx.py).
import sys, time, importlib.util
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
model, tok = load(sys.argv[1]); N = int(sys.argv[2]); cands = sys.argv[3:]
p = tok.apply_chat_template([{"role": "user", "content": "Write a Rust function that parses a CSV line with quoted fields, with unit tests. Code only."}],
                            add_generation_prompt=True, reasoning_effort="none")
sm = make_sampler(temp=0.0)
def run():
    ids = []; last = None
    for r in stream_generate(model, tok, p, max_tokens=N, sampler=sm): ids.append(r.token); last = r
    return ids, last.generation_tps
run(); ref, t = run(); print(f"base {t:.2f}", flush=True)
for c in cands:
    spec = importlib.util.spec_from_file_location("c", c); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    m.apply(model)
    try:
        run(); ids, tc = run()
        diff = next((i for i, (a, b) in enumerate(zip(ids, ref)) if a != b), None)
        print(f"{c.split('/')[-1]} {tc:.2f} x{tc/t:.3f} same={ids == ref} first_diff={diff}", flush=True)
    finally:
        m.revert(model)
    _, t = run()
