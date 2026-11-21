"""把 MoE 块（路由打分、top-k、两次专家矩阵乘、软截断 SwiGLU、加权求和）整段编译。每 token 走 48 次层访问，每次一个 MoE 块。"""
import sys, os; sys.path.insert(0, os.path.dirname(__file__)); import _patch
from mlx_lm.models.inkling2 import layers as L
def apply(model): _patch.compile_class(L.SparseMoeBlock); return {}
def revert(model): _patch.revert_all()
