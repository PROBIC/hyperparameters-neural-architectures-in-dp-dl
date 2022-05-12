from tempered_sigmoid import Temperedsigmoid
import torch


def test_equal_to_sigmoid():
    N = 100
    x = torch.rand(N)
    tempered_sigmoid = Temperedsigmoid(offset=0, inverse_temperature=1, scale=1)
    sigmoid = torch.nn.Sigmoid()
    assert torch.all(tempered_sigmoid.forward(x) == sigmoid.forward(x))
    assert tempered_sigmoid.forward(x).size()[0] == N


def test_equal_to_tanH():
    N = 100
    x = torch.rand(N)
    tempered_sigmoid = Temperedsigmoid(offset=1, inverse_temperature=2, scale=2)
    tanH = torch.nn.Tanh()
    # good enough, up to 1e-4
    assert torch.allclose(tempered_sigmoid.forward(x), tanH.forward(x))
    assert tempered_sigmoid.forward(x).size()[0] == N

def test_scale():
    N = 100
    x = torch.rand(N)
    tempered_sigmoid = Temperedsigmoid(offset=0, inverse_temperature=1, scale=10)
    sigmoid = torch.nn.Sigmoid()
    assert torch.all(tempered_sigmoid.forward(x) == sigmoid.forward(x) * 10)
    assert tempered_sigmoid.forward(x).size()[0] == N


def test_offset():
    N = 100
    x = torch.rand(N)
    tempered_sigmoid = Temperedsigmoid(offset=10, inverse_temperature=1, scale=1)
    sigmoid = torch.nn.Sigmoid()
    assert torch.all(tempered_sigmoid.forward(x) == sigmoid.forward(x) - 10)
    assert tempered_sigmoid.forward(x).size()[0] == N

def test_temperature():
    N = 100
    x = torch.rand(N)
    tempered_sigmoid = Temperedsigmoid(offset=0, inverse_temperature=3, scale=1)
    sigmoid = torch.nn.Sigmoid()
    assert torch.all(tempered_sigmoid.forward(x) == sigmoid.forward(3*x))
    assert tempered_sigmoid.forward(x).size()[0] == N