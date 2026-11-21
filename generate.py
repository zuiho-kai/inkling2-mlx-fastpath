"""Generate with Inkling 2 Mini on MLX, stock or fast path.

    python generate.py --model mlx-community/Inkling-2-Mini-4bit --prompt "Write a quicksort in Rust."
    python generate.py --model ... --prompt ... --stock      # stock mlx-lm, for comparison
"""
import argparse
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
import inkling2_fastpath

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--prompt", required=True)
ap.add_argument("--max-tokens", type=int, default=512)
ap.add_argument("--stock", action="store_true")
a = ap.parse_args()
model, tok = load(a.model)
if not a.stock:
    inkling2_fastpath.enable(model)
p = tok.apply_chat_template([{"role": "user", "content": a.prompt}], add_generation_prompt=True, reasoning_effort="none")
last = None
for r in stream_generate(model, tok, p, max_tokens=a.max_tokens, sampler=make_sampler(temp=0.0)):
    print(r.text, end="", flush=True); last = r
print(f"\n\n[{'stock' if a.stock else 'fast path'}] prefill {last.prompt_tps:.1f} tok/s, decode {last.generation_tps:.1f} tok/s, "
      f"{last.generation_tokens} tokens, peak {last.peak_memory:.1f} GB")
