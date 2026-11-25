import torch, sys
sys.path.insert(0,'archive/ktransformers/operators')
from experts import KQwen2MoeSparseMoeBlock

def test_real_moe_infer_simple():
    # Light mock: minimal instance with needed experts attr
    block = object.__new__(KQwen2MoeSparseMoeBlock)
    block.experts = [torch.nn.Linear(4,4) for _ in range(2)]
    h = torch.randn(2,4)
    s = torch.tensor([[0,1],[1,0]])
    w = torch.ones(2,2)
    out = block.moe_infer_simple(h, s, w)
    assert out.shape == h.shape
