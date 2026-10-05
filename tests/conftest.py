import pytest
import torch

from gpt2_pytorch import DistilGPT2LMHeadModel, GPT2Config
from gpt2_pytorch.checkpoint import DEFAULT_CHECKPOINT, load_pretrained, load_state_dict_file

needs_checkpoint = pytest.mark.skipif(not DEFAULT_CHECKPOINT.exists(), reason=f"{DEFAULT_CHECKPOINT} not present")


@pytest.fixture(scope="session")
def checkpoint_state():
    return load_state_dict_file(DEFAULT_CHECKPOINT)


@pytest.fixture(scope="session")
def pretrained():
    """(model, config, report) with the real distilgpt2 weights, loaded once."""
    return load_pretrained(DEFAULT_CHECKPOINT)


@pytest.fixture(scope="session")
def real_model(pretrained):
    return pretrained[0]


@pytest.fixture
def tiny_config():
    return GPT2Config(vocab_size=100, max_position_embeddings=16, hidden_size=32, num_layers=2, num_heads=4, intermediate_size=64)


@pytest.fixture
def tiny_model(tiny_config):
    torch.manual_seed(0)
    return DistilGPT2LMHeadModel(tiny_config).eval()


@pytest.fixture
def tiny_ids():
    return torch.tensor([[1, 5, 9, 2, 7, 3], [4, 4, 8, 0, 6, 99]])
