from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fgac.data.chunk_dataset import FlowMatchingChunkDataset
from fgac.models.flow_matching import FlowMatchingMLP, euler_sample
from fgac.models.policy_io import LoadedFlowPolicy, sample_normalized_actions
from fgac.transforms.dct import (
    channel_topk_frequency_bins,
    dct_time_torch,
    dct_time,
    decode_action_target,
    idct_time,
    idct_time_torch,
    sparse_residual_target,
)


def test_sparse_residual_target_decodes_to_original_actions() -> None:
    rng = np.random.default_rng(0)
    actions = rng.normal(size=(6, 16, 3)).astype(np.float32)

    target = sparse_residual_target(actions, k=8)
    decoded = decode_action_target(
        target,
        target_type="dct_sparse_residual",
        horizon=16,
        action_dim=3,
        dct_k=8,
    )

    np.testing.assert_allclose(decoded, actions, atol=1.0e-6)


def test_channel_sparse_target_uses_per_channel_bandwidth() -> None:
    rng = np.random.default_rng(8)
    obs = rng.normal(size=(5, 6)).astype(np.float32)
    actions = rng.normal(size=(5, 16, 3)).astype(np.float32)
    channel_k = [2, 4, 8]

    ds = FlowMatchingChunkDataset(obs, actions, "dct_channel_sparse", dct_k=channel_k)
    expected_coeffs, mask = channel_topk_frequency_bins(dct_time(actions), channel_k)
    decoded = decode_action_target(
        ds.target_sequences,
        target_type="dct_channel_sparse",
        horizon=16,
        action_dim=3,
        dct_k=channel_k,
    )

    assert ds.target_shape == (16, 3)
    assert ds.target_dim == 48
    assert mask[0, :, 0].sum() == 2
    assert mask[0, :, 1].sum() == 4
    assert mask[0, :, 2].sum() == 8
    np.testing.assert_allclose(ds.target_sequences, expected_coeffs, atol=0.0)
    np.testing.assert_allclose(decoded, idct_time(expected_coeffs), atol=1.0e-6)


def test_softmask_decode_with_full_coefficients_matches_fullfreq() -> None:
    rng = np.random.default_rng(1)
    actions = rng.normal(size=(4, 16, 5)).astype(np.float32)
    coeffs = dct_time(actions)

    softmask_decoded = decode_action_target(
        coeffs,
        target_type="dct_softmask",
        horizon=16,
        action_dim=5,
    )
    fullfreq_decoded = decode_action_target(
        coeffs,
        target_type="dct_fullfreq",
        horizon=16,
        action_dim=5,
    )

    np.testing.assert_allclose(softmask_decoded, fullfreq_decoded, atol=1.0e-6)
    np.testing.assert_allclose(softmask_decoded, actions, atol=1.0e-6)


def test_torch_idct_matches_scipy_idct() -> None:
    rng = np.random.default_rng(2)
    coeffs = rng.normal(size=(3, 16, 4)).astype(np.float64)

    expected = idct_time(coeffs)
    actual = idct_time_torch(torch.from_numpy(coeffs)).numpy()

    np.testing.assert_allclose(actual, expected, atol=1.0e-12)


def test_dataset_shapes_for_v2_targets() -> None:
    rng = np.random.default_rng(3)
    obs = rng.normal(size=(7, 9)).astype(np.float32)
    actions = rng.normal(size=(7, 16, 2)).astype(np.float32)

    residual_ds = FlowMatchingChunkDataset(obs, actions, "dct_sparse_residual", dct_k=8)
    softmask_ds = FlowMatchingChunkDataset(obs, actions, "dct_softmask", dct_k=None)

    assert residual_ds.target_shape == (16, 4)
    assert residual_ds.target_dim == 64
    assert softmask_ds.target_shape == (16, 2)
    assert softmask_ds.target_dim == 32


def test_raw_anchored_residual_dataset_keeps_raw_shape() -> None:
    rng = np.random.default_rng(4)
    obs = rng.normal(size=(5, 6)).astype(np.float32)
    actions = rng.normal(size=(5, 16, 3)).astype(np.float32)

    ds = FlowMatchingChunkDataset(obs, actions, "raw_anchored_residual", dct_k=None)
    decoded = decode_action_target(
        ds.target_sequences,
        target_type="raw_anchored_residual",
        horizon=16,
        action_dim=3,
    )

    assert ds.target_shape == (16, 3)
    assert ds.target_dim == 48
    np.testing.assert_allclose(decoded, actions, atol=0.0)


def test_sparse_dct_anchored_residual_dataset_keeps_residual_shape() -> None:
    rng = np.random.default_rng(5)
    obs = rng.normal(size=(5, 6)).astype(np.float32)
    residual = rng.normal(size=(5, 16, 3)).astype(np.float32)
    coeffs = dct_time(residual)

    ds = FlowMatchingChunkDataset(obs, residual, "sparse_dct_anchored_residual", dct_k=None)
    decoded = decode_action_target(
        coeffs,
        target_type="sparse_dct_anchored_residual",
        horizon=16,
        action_dim=3,
    )

    assert ds.target_shape == (16, 3)
    assert ds.target_dim == 48
    np.testing.assert_allclose(decoded, residual, atol=1.0e-6)


def test_torch_dct_matches_scipy_dct() -> None:
    rng = np.random.default_rng(6)
    actions = rng.normal(size=(3, 16, 4)).astype(np.float64)

    expected = dct_time(actions)
    actual = dct_time_torch(torch.from_numpy(actions)).numpy()

    np.testing.assert_allclose(actual, expected, atol=1.0e-12)


def test_sample_normalized_actions_decodes_sparse_dct_base() -> None:
    class ZeroPolicy(torch.nn.Module):
        def forward(self, x_t: torch.Tensor, t: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(x_t)

    policy = LoadedFlowPolicy(
        model=ZeroPolicy(),
        cfg={"target": {"type": "dct_sparse", "dct_k": 4}, "sampling": {"num_flow_steps": 2}},
        metadata={"dataset": {"action_dim": 3}},
        target_type="dct_sparse",
        model_type="temporal_unet",
        action_dim=3,
        horizon=16,
        target_shape=(16, 3),
        target_dim=48,
    )
    obs = torch.randn(2, 6)

    decoded = sample_normalized_actions(policy, obs, num_steps=2, noise_mode="zero")

    torch.testing.assert_close(decoded, torch.zeros(2, 16, 3))


def test_sample_normalized_actions_decodes_channel_sparse_dct_base() -> None:
    class ZeroPolicy(torch.nn.Module):
        def forward(self, x_t: torch.Tensor, t: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(x_t)

    policy = LoadedFlowPolicy(
        model=ZeroPolicy(),
        cfg={"target": {"type": "dct_channel_sparse", "dct_k": [2, 4, 8]}, "sampling": {"num_flow_steps": 2}},
        metadata={"dataset": {"action_dim": 3}},
        target_type="dct_channel_sparse",
        model_type="temporal_unet",
        action_dim=3,
        horizon=16,
        target_shape=(16, 3),
        target_dim=48,
    )
    obs = torch.randn(2, 6)

    decoded = sample_normalized_actions(policy, obs, num_steps=2, noise_mode="zero")

    torch.testing.assert_close(decoded, torch.zeros(2, 16, 3))


def test_raw_anchored_dct_residual_is_retired() -> None:
    rng = np.random.default_rng(7)
    obs = rng.normal(size=(5, 6)).astype(np.float32)
    actions = rng.normal(size=(5, 16, 3)).astype(np.float32)

    try:
        FlowMatchingChunkDataset(obs, actions, "raw_anchored_dct_residual", dct_k=4)
    except ValueError:
        pass
    else:
        raise AssertionError("raw_anchored_dct_residual should not be an active dataset target")

    try:
        decode_action_target(actions, target_type="raw_anchored_dct_residual", horizon=16, action_dim=3, dct_k=4)
    except ValueError:
        pass
    else:
        raise AssertionError("raw_anchored_dct_residual should not decode as an active target")


def test_euler_sample_zero_noise_repeats() -> None:
    torch.manual_seed(0)
    model = FlowMatchingMLP(obs_dim=2, target_dim=4, hidden_dim=8, num_layers=1)
    obs = torch.randn(3, 2)

    first = euler_sample(model, obs, target_dim=4, num_steps=2, noise_mode="zero")
    second = euler_sample(model, obs, target_dim=4, num_steps=2, noise_mode="zero")

    torch.testing.assert_close(first, second)
