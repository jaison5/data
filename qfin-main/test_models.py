import time, numpy as np
from ad import *
from models import MODELS
rng = np.random.default_rng(1)
X = rng.uniform(0, 1, size=(3, 5, 4)); y = rng.uniform(0, 1, size=(3, 1))
for name, cls in MODELS.items():
    m = cls(rng); ps = m.params()
    f = lambda: mean(square(m(X) - T(y)))
    for p in ps: p.grad = None
    f().backward(); g = [p.grad.copy() for p in ps]
    err = 0
    for p, gp in zip(ps, g):
        it = np.nditer(p.data, flags=["multi_index"])
        for _ in it:
            i = it.multi_index; o = p.data[i]
            p.data[i] = o + 1e-6; a = f().data
            p.data[i] = o - 1e-6; b = f().data
            p.data[i] = o
            num = (a - b) / 2e-6
            err = max(err, abs(gp[i] - num) / (abs(num) + 1e-8))
    # timing for B=1 and B=64 train step
    out = []
    for B in (1, 64):
        Xb = rng.uniform(0, 1, size=(B, 5, 4)); yb = rng.uniform(size=(B, 1))
        t0 = time.time()
        for _ in range(20):
            for p in ps: p.grad = None
            mean(square(m(Xb) - T(yb))).backward()
        out.append((time.time() - t0) / 20 * 1000)
    print(f"{name}: max rel grad err {err:.1e}; ms/step B=1 {out[0]:.2f}, B=64 {out[1]:.2f}")
