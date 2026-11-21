"""第 1 轮三个编译点（消息门、MoE、深度残差）全上。"""
import sys, os; sys.path.insert(0, os.path.dirname(__file__)); import _patch
from mlx_lm.models.inkling2 import layers as L
def apply(model):
    for c in (L.MessageGate, L.SparseMoeBlock, L.BlockAttentionResidual): _patch.compile_class(c)
    return {}
def revert(model): _patch.revert_all()
