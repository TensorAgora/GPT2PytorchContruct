import math

import torch
import torch.nn.functional as F

from gpt2_pytorch.model import GPT2GELU, GPT2MLP


def test_gelu_is_the_tanh_approximation_not_erf():
    x = torch.linspace(-6, 6, 241)
    gelu = GPT2GELU()
    torch.testing.assert_close(gelu(x), F.gelu(x, approximate="tanh"), rtol=1e-5, atol=1e-6)
    assert (gelu(x) - F.gelu(x)).abs().max() > 1e-4  # differs from exact erf GELU
    manual = 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))
    torch.testing.assert_close(gelu(x), manual)


def test_mlp_expands_activates_projects(tiny_config):
    torch.manual_seed(0)
    mlp = GPT2MLP(tiny_config).eval()
    x = torch.randn(2, 5, 32)
    output, d = mlp(x, return_debug=True)
    assert mlp.fc.weight.shape == (64, 32) and mlp.projection.weight.shape == (32, 64)
    assert d["fc.output"].shape == (2, 5, 64) and d["activation.output"].shape == (2, 5, 64)
    assert torch.equal(output, mlp.projection(F.gelu(mlp.fc(x), approximate="tanh")))
