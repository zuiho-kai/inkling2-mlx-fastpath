# Record real per-token timestamps: same prompt, same process, stock then the fast path (one warm-up each) -> trace.json
import sys, time, json, importlib.util
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
model_dir, cand, N, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
model, tok = load(model_dir)
PROMPT = "Write a Python function that merges overlapping intervals, with three pytest tests. Code only."
p = tok.apply_chat_template([{"role": "user", "content": PROMPT}], add_generation_prompt=True, reasoning_effort="none")
sm = make_sampler(temp=0.0)
def run():
    t0 = time.perf_counter(); ev = []; last = None
    for r in stream_generate(model, tok, p, max_tokens=N, sampler=sm):
        ev.append([time.perf_counter() - t0, r.token, r.text]); last = r
    return dict(events=ev, decode_tps=last.generation_tps, prefill_tps=last.prompt_tps, n=last.generation_tokens)
run(); base = run()
sys.path.insert(0, cand); import inkling2_fastpath as m  # cand = this repo dir: records the published package itself
m.enable(model); run(); ours = run(); m.revert(model)
same = [e[1] for e in base["events"]] == [e[1] for e in ours["events"]]
json.dump(dict(prompt=PROMPT, base=base, ours=ours, same_tokens=same, chip="Apple M5 Max", model="Inkling 2 Mini (MLX 4-bit)"), open(out, "w"))
print(f"base {base['decode_tps']:.1f} ours {ours['decode_tps']:.1f} same={same}")
