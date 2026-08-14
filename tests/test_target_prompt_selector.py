from __future__ import annotations

import torch

from tcd_prg.models.target_prompt import TargetPromptSelector


def _scene():
    # Two visually/category-identical instances, spatially separated.
    xyz = torch.tensor([[[
        0.00, 0.00, 0.10,
    ], [
        0.01, 0.00, 0.10,
    ], [
        0.00, 0.01, 0.10,
    ], [
        0.40, 0.00, 0.10,
    ], [
        0.41, 0.00, 0.10,
    ], [
        0.40, 0.01, 0.10,
    ]]], dtype=torch.float32)
    mask = torch.tensor([[[.98, .97, .96, .02, .02, .02],
                          [.02, .02, .02, .98, .97, .96]]])
    centers = torch.tensor([[[0.003, 0.003, .10], [.403, .003, .10]]])
    objectness = torch.tensor([[4.0, 4.0]])
    category = torch.tensor([[[5.0, -2.0], [5.0, -2.0]]])
    tokens = torch.tensor([[[1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0]]])
    return xyz, mask, centers, objectness, category, tokens


def _selector():
    module = TargetPromptSelector(
        radius_m=.04,
        sigma_m=.015,
        prompt_weight=6.0,
        category_weight=1.0,
        objectness_weight=.25,
        center_weight=.25,
        learned_weight=0.0,
        reid_weight=2.0,
        reid_center_weight=.5,
        temperature=.25,
    )
    return module.eval()


def test_positive_prompt_resolves_same_category_instances():
    xyz, mask, centers, obj, cat, tokens = _scene()
    selector = _selector()
    common = dict(
        instance_probability=mask,
        objectness_logits=obj,
        category_logits=cat,
        centers_world=centers,
        object_tokens=tokens,
        xyz=xyz,
        point_mask=torch.ones(1, 6, dtype=torch.bool),
        task_category_id=torch.tensor([0]),
        target_prompt_label=torch.tensor([[1]]),
        target_prompt_valid=torch.tensor([[True]]),
    )
    left = selector(
        **common,
        target_prompt_xyz=torch.tensor([[[.005, .002, .10]]]),
    )
    right = selector(
        **common,
        target_prompt_xyz=torch.tensor([[[.405, .002, .10]]]),
    )
    assert int(left.query_index[0]) == 0
    assert int(right.query_index[0]) == 1
    assert float(left.positive_prompt_support[0, 0]) > float(left.positive_prompt_support[0, 1])
    assert float(right.positive_prompt_support[0, 1]) > float(right.positive_prompt_support[0, 0])


def test_negative_prompt_can_disambiguate_touching_candidates():
    xyz, mask, centers, obj, cat, tokens = _scene()
    selector = _selector()
    result = selector(
        instance_probability=mask,
        objectness_logits=obj,
        category_logits=cat,
        centers_world=centers,
        object_tokens=tokens,
        xyz=xyz,
        point_mask=torch.ones(1, 6, dtype=torch.bool),
        task_category_id=torch.tensor([0]),
        target_prompt_xyz=torch.tensor([[[.405, .002, .10], [.005, .002, .10]]]),
        target_prompt_label=torch.tensor([[1, 0]]),
        target_prompt_valid=torch.tensor([[True, True]]),
    )
    assert int(result.query_index[0]) == 1


def test_reidentification_never_assumes_query_id_stability():
    xyz, mask, centers, obj, cat, tokens = _scene()
    # New frame swaps query identities/tokens and moves the target slightly.
    mask2 = mask[:, [1, 0]]
    centers2 = centers[:, [1, 0]]
    tokens2 = tokens[:, [1, 0]]
    selector = _selector()
    result = selector(
        instance_probability=mask2,
        objectness_logits=obj,
        category_logits=cat,
        centers_world=centers2,
        object_tokens=tokens2,
        xyz=xyz,
        point_mask=torch.ones(1, 6, dtype=torch.bool),
        task_category_id=torch.tensor([0]),
        target_prompt_xyz=None,
        target_prompt_valid=None,
        target_reid_token=tokens[:, 1],
        target_reid_center=centers[:, 1],
        target_reid_valid=torch.tensor([True]),
    )
    # Previous physical right-hand object was query 1; after swap it is query 0.
    assert int(result.query_index[0]) == 0


def test_selector_adds_no_trainable_parameters_for_v1_checkpoint_compatibility():
    selector = _selector()
    assert list(selector.parameters()) == []
