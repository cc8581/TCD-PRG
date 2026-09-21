"""Physical contact and direction invariants for deployment PUSH rules."""

from types import SimpleNamespace

import numpy as np
import torch

from tcd_prg.models.push.rules import RulePushGenerator, observed_contact_boundary
from tcd_prg.models.push_condition import PushCondition


def _scene(*, close_bottom: bool, target_offset_x: float = 0.0):
    target = np.array([
        [x + target_offset_x, y, 0.0]
        for x in np.linspace(-0.02, 0.02, 5)
        for y in np.linspace(-0.02, 0.02, 5)
    ])
    edges = [
        [[x, 0.05, 0.05] for x in np.linspace(-0.05, 0.05, 21)],
        [[-0.05, y, 0.05] for y in np.linspace(-0.05, 0.05, 21)],
        [[0.05, y, 0.05] for y in np.linspace(-0.05, 0.05, 21)],
    ]
    if close_bottom:
        edges.append([[x, -0.05, 0.05] for x in np.linspace(-0.05, 0.05, 21)])
    upper = np.unique(np.array([point for edge in edges for point in edge]), axis=0)
    xyz = torch.tensor(np.vstack((target, upper)), dtype=torch.float32)[None]
    count = len(target) + len(upper)
    probability = torch.zeros(1, 2, count)
    probability[0, 0, :len(target)] = 1
    probability[0, 1, len(target):] = 1
    condition = PushCondition(
        probability, torch.ones(1, 2, dtype=torch.bool),
        probability[0, 0][None], probability[0, 0][None],
        torch.ones(1, dtype=torch.bool),
        torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long),
    )
    sensor = {"xyz": xyz, "point_mask": torch.ones(1, count, dtype=torch.bool)}
    return sensor, condition, upper


def _actions(sensor, condition):
    settings = SimpleNamespace(push_contact_spacing_m=0.01, push_above_margin_m=0.002)
    return RulePushGenerator(settings)(sensor, condition)


def test_concave_object_never_gets_a_contact_on_a_phantom_hull_edge():
    sensor, condition, upper = _scene(close_bottom=False, target_offset_x=-0.03)
    actions = _actions(sensor, condition)
    assert len(actions.object) > 0
    contact = actions.contact_world.numpy()[:, :2]
    nearest = np.linalg.norm(contact[:, None] - upper[None, :, :2], axis=-1).min(axis=1)
    assert np.all(nearest < 1e-6)


def test_coincident_centers_have_no_defined_main_direction():
    sensor, condition, _ = _scene(close_bottom=True)
    actions = _actions(sensor, condition)
    assert len(actions.object) == 0


def test_offset_target_pushes_the_obstruction_away_from_target():
    sensor, condition, _ = _scene(close_bottom=True, target_offset_x=-0.03)
    actions = _actions(sensor, condition)
    assert len(actions.object) > 0
    assert np.all(actions.direction_world.numpy()[:, 0] > 0)
    assert np.all(actions.contact_world.numpy()[:, 0] <= 0)
    directions = actions.direction_world.numpy()[:, :2]
    assert np.ptp(directions[:, 1]) > 0.1
    center = np.array([0., 0.])
    main = np.array([1., 0.])
    for contact, direction in zip(actions.contact_world.numpy(), directions, strict=True):
        inward = center - contact[:2]
        inward /= np.linalg.norm(inward)
        angle = np.arctan2(main[0] * inward[1] - main[1] * inward[0], main @ inward)
        u = np.clip((abs(np.degrees(angle)) - 20.) / 50., 0., 1.)
        weight = .15 + .60 * u * u * (3. - 2. * u)
        assert np.allclose(direction, [np.cos(weight * angle), np.sin(weight * angle)], atol=.02)


def test_push_direction_enters_the_observed_contact_surface():
    sensor, condition, upper = _scene(close_bottom=True, target_offset_x=-0.03)
    actions = _actions(sensor, condition)
    boundary = observed_contact_boundary(upper)
    assert len(actions.object) > 0
    for contact, direction in zip(
        actions.contact_world.numpy(), actions.direction_world.numpy(), strict=True
    ):
        normals = next(
            normals for point, normals in boundary
            if np.linalg.norm(point - contact) < 1e-6
        )
        assert max(float(direction[:2] @ normal) for normal in normals) >= 0.5


def test_contact_generation_is_invariant_to_point_order():
    sensor, condition, _ = _scene(close_bottom=True)
    original = _actions(sensor, condition)
    permutation = torch.randperm(sensor["xyz"].shape[1], generator=torch.Generator().manual_seed(7))
    reordered_sensor = {
        "xyz": sensor["xyz"][:, permutation],
        "point_mask": sensor["point_mask"][:, permutation],
    }
    reordered_condition = PushCondition(
        condition.object_probability[:, :, permutation], condition.object_valid,
        condition.target_probability[:, permutation],
        condition.region_probability[:, permutation], condition.target_valid,
        condition.task_category_id, condition.task_region_id,
    )
    reordered = _actions(reordered_sensor, reordered_condition)
    assert np.allclose(original.contact_world.numpy(), reordered.contact_world.numpy())
    assert np.allclose(original.direction_world.numpy(), reordered.direction_world.numpy())


def test_adjacent_blocker_can_generate_without_projection_overlap_or_height_gap():
    sensor, condition, _ = _scene(close_bottom=True, target_offset_x=-0.10)
    shifted = sensor["xyz"].clone()
    mask = condition.object_probability[0, 1].bool()
    shifted[0, mask, 2] = 0.0
    adjacent_sensor = {**sensor, "xyz": shifted}
    generator = RulePushGenerator(SimpleNamespace(
        push_contact_spacing_m=.01, push_above_margin_m=.002,
    ))
    assert len(generator(adjacent_sensor, condition).object) == 0
    actions = generator(adjacent_sensor, condition, adjacent_objects=(1,))
    assert len(actions.object) > 0
    assert set(actions.object.tolist()) == {1}
    assert np.all(actions.direction_world.numpy()[:, 0] > 0)
