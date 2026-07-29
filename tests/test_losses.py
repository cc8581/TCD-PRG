import torch

from tcd_prg.losses.masked import multi_positive_listwise_loss, safe_smooth_l1


def test_nan_loss_is_masked() -> None:
    prediction = torch.tensor([1.0, 2.0], requires_grad=True)
    target = torch.tensor([1.5, float("nan")])
    loss = safe_smooth_l1(prediction, target, torch.tensor([True, False]))
    assert torch.isfinite(loss)
    loss.backward()
    assert prediction.grad[1] == 0


def test_multiple_correct_action_set_loss() -> None:
    logits = torch.tensor([[0.0, 2.0, 1.0]], requires_grad=True)
    positive = torch.tensor([[False, True, True]])
    valid = torch.ones_like(positive)
    loss = multi_positive_listwise_loss(logits, positive, valid)
    assert 0 <= loss < 1
    loss.backward()


def test_unknown_candidate_excluded_from_listwise_denominator() -> None:
    logits = torch.tensor([[0.0, 1.0, 100.0]])
    positive = torch.tensor([[False, True, False]])
    evaluated = torch.tensor([[True, True, False]])
    loss = multi_positive_listwise_loss(logits, positive, evaluated)
    assert loss < 1


def test_sequence_topology_mask_only_masks_order_loss() -> None:
    from tcd_prg.losses.masked import safe_bce_with_logits

    logits = torch.tensor([[100.0]], requires_grad=True)
    topology_valid = torch.tensor([False])[:, None]
    topology = safe_bce_with_logits(logits, torch.zeros_like(logits), topology_valid)
    action = safe_bce_with_logits(logits, torch.zeros_like(logits), torch.ones_like(topology_valid))
    assert topology == 0
    assert action > 0
