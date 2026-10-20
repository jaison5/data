import numpy as np
from ad import *
from models import MODELS
rng = np.random.default_rng(1)
X = rng.uniform(0, 1, size=(3, 5, 4)); y = rng.uniform(0, 1, size=(3, 1))
for name, cls in MODELS.items():
    m = cls(rng); ps = m.params()
    f = lambda: mean(square(m(X) - T(y)))
    for p in ps: p.grad = None
    f().backward(); g = np.concatenate([p.grad.ravel() for p in ps])
    num = []
    for p in ps:
        it = np.nditer(p.data, flags=["multi_index"])
        for _ in it:
            i = it.multi_index; o = p.data[i]
            p.data[i] = o + 1e-5; a = f().data
            p.data[i] = o - 1e-5; b = f().data
            p.data[i] = o
            num.append((a - b) / 2e-5)
    num = np.array(num)
    print(name, "max|g|", np.abs(num).max(), "max abs err", np.abs(g - num).max(), "rel(norm)", np.linalg.norm(g-num)/np.linalg.norm(num))
