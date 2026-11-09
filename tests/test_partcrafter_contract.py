from pathlib import Path

import numpy as np
import pytest
import torch

from qamg.part_model import ROOT, clean_neural_components, load_native_scheduler


def test_native_scheduler_matches_checkpoint_and_positive_velocity_reference(tmp_path):
    import json
    import shutil

    scheduler = load_native_scheduler(ROOT/"configs/partcrafter")
    assert type(scheduler).__name__ == "RectifiedFlowScheduler"
    scheduler.set_timesteps(2, device="cpu")
    np.testing.assert_allclose(scheduler.timesteps.numpy(), [1000., 500.])
    sample = torch.zeros(1, 1, 1)
    # Independent constant-velocity ODE: two half steps advance zero to +1.
    for timestep in scheduler.timesteps:
        sample = scheduler.step(torch.ones_like(sample), timestep, sample).prev_sample
    np.testing.assert_allclose(sample.numpy(), np.ones((1, 1, 1)))
    shutil.copytree(ROOT/"configs/partcrafter/scheduler", tmp_path/"scheduler")
    wrong = {"scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"]}
    (tmp_path/"model_index.json").write_text(json.dumps(wrong))
    with pytest.raises(ValueError, match="不匹配"):
        load_native_scheduler(tmp_path)


def test_cfg_batches_keep_the_same_part_identity_in_both_branches(monkeypatch):
    # A batched CFG call must equal two independent calls with identical part IDs.
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"vendor/partcrafter"))
    from partcrafter.models.transformers import PartCrafterDiTModel

    torch.manual_seed(12)
    model = PartCrafterDiTModel(num_attention_heads=4, width=32, in_channels=8, num_layers=3,
                               cross_attention_dim=24, max_num_parts=8,
                               global_attn_block_ids=[0, 2]).eval()
    latent = torch.randn(2, 4, 8)
    image = torch.randn(2, 5, 24)
    with torch.no_grad():
        together = model(torch.cat([latent, latent]), torch.ones(4),
                         encoder_hidden_states=torch.cat([torch.zeros_like(image), image]),
                         attention_kwargs={"num_parts": 2}).sample
        separate = torch.cat([
            model(latent, torch.ones(2), encoder_hidden_states=condition,
                  attention_kwargs={"num_parts": 2}).sample
            for condition in [torch.zeros_like(image), image]
        ])
    np.testing.assert_allclose(together.numpy(), separate.numpy(), atol=1e-6, rtol=1e-5)


def test_paired_limbs_are_kept_but_distributed_fragment_cloud_is_rejected():
    import trimesh

    arms = [trimesh.creation.box([10, 2, 2]), trimesh.creation.box([10, 2, 2])]
    arms[1].apply_translation([25, 0, 0])
    dust = []
    for i in range(12):
        piece = trimesh.creation.box([.2, .2, .2])
        piece.apply_translation([i, 10, 0])
        dust.append(piece)
    cleaned, record = clean_neural_components(trimesh.util.concatenate([*arms, *dust]))
    assert len(cleaned.split()) == 2
    assert 0 < record["removed_volume_fraction"] < .02
    cloud = []
    for i in range(120):
        piece = trimesh.creation.box([1, 1, 1])
        piece.apply_translation([i*3, 0, 0])
        cloud.append(piece)
    with pytest.raises(ValueError, match="分散碎片"):
        clean_neural_components(trimesh.util.concatenate(cloud))
