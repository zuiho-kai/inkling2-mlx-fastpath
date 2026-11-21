# 机制证据：解码一步里「Python 建图」和整步耗时各多少；编译候选应主要减少建图时间（少发小算子）
# 用法: python mechanism.py <模型目录> [候选.py]
import sys, time, importlib.util, statistics
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

model, tok = load(sys.argv[1]); cand = sys.argv[2] if len(sys.argv) > 2 else None
if cand:
    spec = importlib.util.spec_from_file_location("c", cand); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); m.apply(model)
ids = mx.array(tok.encode("def lru_cache(capacity):"))[None]
cache = make_prompt_cache(model); out = model(ids, cache=cache); mx.eval(out)
y = mx.argmax(out[:, -1:], -1)
build, total = [], []
for i in range(60):  # 60 步，前 10 步当预热丢掉（首步含编译）
    t0 = time.perf_counter(); out = model(y, cache=cache); t1 = time.perf_counter()
    y = mx.argmax(out[:, -1:], -1); mx.eval(y); t2 = time.perf_counter()
    if i >= 10: build.append(t1 - t0); total.append(t2 - t0)
print(f"{'cand' if cand else 'base'} build_ms={statistics.median(build)*1e3:.2f} step_ms={statistics.median(total)*1e3:.2f}")
