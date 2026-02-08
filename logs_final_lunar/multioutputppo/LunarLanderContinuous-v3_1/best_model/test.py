
import torch as th

m = th.load("./policy.pth")
for k, v in m.items():
    print(f"Key: {k}, Value shape: {v.shape if isinstance(v, th.Tensor) else type(v)}")
