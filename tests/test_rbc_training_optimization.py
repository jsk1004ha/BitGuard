import copy

import numpy as np
import pytest
import torch

from bitguard_bnn.rbc import _batches
from bitguard_bnn.rbc_model import DetectionFirstBNN, conditional_type_loss


def legacy_type_training(model, values, target, type_targets, *, epochs, batch_size, seed, lr):
    optimizer = torch.optim.Adam(model.type_parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        model.train()
        for idx in _batches(len(values), batch_size, rng):
            if not bool(target[idx].any()):
                continue
            optimizer.zero_grad(set_to_none=True)
            loss = conditional_type_loss(model(values[idx]).type_logits,
                                         type_targets[idx], attack_mask=target[idx].bool())
            loss.backward()
            optimizer.step()


@pytest.mark.parametrize("classes", [1, 2, 4])
@pytest.mark.parametrize("cache_bytes", [0, 100, 1_000_000])
def test_type_training_matches_original_updates(classes, cache_bytes):
    from bitguard_bnn.rbc import _train_types

    torch.manual_seed(17)
    model = DetectionFirstBNN(8, [8, 4], attack_classes=classes, dropout=.3)
    model.freeze_detector()
    original = copy.deepcopy(model)
    values = torch.where(torch.randn(33, 8) > 0, 1., -1.)
    targets = (torch.arange(33) % 3 != 0).float()
    types = torch.arange(33) % classes
    initial_detector = copy.deepcopy(model.detector.state_dict())
    kwargs = dict(epochs=3, batch_size=8, seed=29, lr=.001)
    legacy_type_training(original, values, targets, types, **kwargs)
    report = _train_types(model, values, targets, types, cache_max_bytes=cache_bytes, **kwargs)
    for key, value in original.state_dict().items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    for key, value in initial_detector.items():
        torch.testing.assert_close(model.detector.state_dict()[key], value, rtol=0, atol=0)
    assert report["cached_hidden"] == (cache_bytes >= 33 * 4)
    assert report["cache_bytes"] <= cache_bytes
    assert report["optimizer_steps"] == 12


def test_cache_rejects_unfrozen_detector():
    from bitguard_bnn.rbc import _train_types

    with pytest.raises(ValueError, match="frozen"):
        _train_types(DetectionFirstBNN(2, [2]), torch.ones(2, 2), torch.ones(2),
                     torch.zeros(2, dtype=torch.long), epochs=2, batch_size=2,
                     seed=1, lr=.001, cache_max_bytes=100)


def test_zero_type_epochs_does_not_evaluate_shared_network():
    from bitguard_bnn.rbc import _train_types

    model = DetectionFirstBNN(2, [2])
    model.freeze_detector()
    report = _train_types(model, torch.ones(2, 2), torch.ones(2), torch.zeros(2, dtype=torch.long),
                          epochs=0, batch_size=2, seed=1, lr=.001, cache_max_bytes=100)
    assert report["cache_bytes"] == 0
    assert report["optimizer_steps"] == 0
