import os

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("CHORA_MLX_TEST")!="1",reason="Set CHORA_MLX_TEST=1 on a Mac with GPU access")


@pytest.mark.parametrize("tied",[False,True])
def test_compact_loss_matches_full_loss_and_gradients(tied):
    """The faster head projection must optimize exactly the intended token loss."""
    import mlx.core as mx
    import mlx.nn as nn
    from types import SimpleNamespace
    from chora.learn import assistant_loss
    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(11,6)
        def __call__(self,x):
            return self.embed_tokens(x)
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()
            if not tied:
                self.lm_head = nn.Linear(6,11)
            self.args = SimpleNamespace(tie_word_embeddings=tied)
    mx.random.seed(1)
    model = Tiny()
    batch = mx.array([[1,2,3,4,5,6,0,0],[2,3,4,5,6,7,8,9]])
    lengths = mx.array([[2,6],[4,8]])
    def reference(model):
        hidden = model.model(batch[:,:-1])
        logits = model.model.embed_tokens.as_linear(hidden) if tied else model.lm_head(hidden)
        indices = mx.arange(batch.shape[1]-1)[None,:]
        mask = (indices>=lengths[:,0:1]-1)&(indices<lengths[:,1:2]-1)
        return (nn.losses.cross_entropy(logits,batch[:,1:])*mask).sum()/mask.sum()
    def compact(model):
        return assistant_loss(model,batch,lengths,completion_length=4)[0]
    loss1,grad1 = nn.value_and_grad(model,reference)(model)
    loss2,grad2 = nn.value_and_grad(model,compact)(model)
    assert abs(loss1.item()-loss2.item())<1e-6
    from mlx.utils import tree_flatten
    for (name,a),(other,b) in zip(tree_flatten(grad1),tree_flatten(grad2)):
        assert name==other
        assert mx.allclose(a,b,atol=1e-6).item(),name
