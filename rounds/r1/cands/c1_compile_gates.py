"""把循环层的消息门（MessageGate：两次 rms_norm、拼接、三次投影、两次 tanh）整段编译成一张图，减少解码时每步的小算子发射。循环段后两遍每层 4 个门，每 token 约 96 次调用。"""
import sys, os; sys.path.insert(0, os.path.dirname(__file__)); import _patch
from mlx_lm.models.inkling2 import layers as L
def apply(model): _patch.compile_class(L.MessageGate); return {}
def revert(model): _patch.revert_all()
