import torch

def test_moe_infer_simple_index():
    class FakeExperts:
        def __init__(self):
            self.experts = [torch.nn.Linear(4,4) for _ in range(2)]
        def moe_infer_simple(self, h, s, w):
            outs = torch.zeros_like(h)
            for i in range(s.size(0)):
                for j in range(s.size(1)):
                    exp = self.experts[int(s[i,j])]
                    outs[i] += exp(h[i]) * w[i,j]
            return outs
    f = FakeExperts()
    h = torch.randn(2,4)
    s = torch.tensor([[0,1],[1,0]])
    w = torch.ones(2,2)
    out = f.moe_infer_simple(h,s,w)
    assert out.shape == h.shape
