# 候选共用：把某个模块类的 __call__ 换成「每个实例一份 mx.compile」的版本；revert 换回
import mlx.core as mx
_saved = {}
def compile_class(cls):
    orig = cls.__call__; _saved[cls] = orig; cache = {}
    def call(self, *a, **k):
        f = cache.get(id(self))
        if f is None:
            f = cache[id(self)] = mx.compile(lambda *aa: orig(self, *aa))
        return f(*a) if not k else orig(self, *a, **k)
    cls.__call__ = call
def revert_all():
    for cls, orig in _saved.items(): cls.__call__ = orig
    _saved.clear()
