import functools
import pathlib
from typing import Any, Iterable
import torch
from torch.utils.data import ConcatDataset
from utils.data_loading import load_dataset

import numpy as np

ACTION_DIM_IS_PAD = {
    'dual_arm_7dof': torch.zeros(16, dtype=torch.bool),
    'dual_arm_6dof': torch.tensor(
        [False, False, False, False, False, False, True, False] * 2,
        dtype=torch.bool
    ),
    'single_arm_7dof': torch.tensor([False] * 8 + [True] * 8, dtype=torch.bool),
}

MergeSpec = Iterable[tuple[str, slice] | tuple[float, int]]

def merge_features(
    inst: dict[str, Any],
    merge_prefix: str,
    merge_spec: MergeSpec,
    merged_feature_key: str | None = None,
):
    """
    Merge features from the instance dictionary into a single feature.
    Feature keys starting with `merge_prefix` are removed from the instance dictionary.
    If `merged_feature_key` is not provided, the merged feature key is the same as the `merge_prefix`.

    Merge spec is a list of tuples of:
    -  feature key (without `merge_prefix`) and a slice object of dimensions to take, OR
    -  a float value (which would be used for padding) and the number of dimensions to pad.

    The merge output is a single tensor with the concatenated features dimensions.
    """
    first_feature = next(feat for feat, _ in merge_spec if feat is not None)
    first_feature_tensor = inst[f"{merge_prefix}.{first_feature}"]
    leading_dims = first_feature_tensor.shape[:-1]
    to_cat = [
        inst[f"{merge_prefix}.{feat}"][..., dims] if isinstance(feat, str) \
            else torch.full((*leading_dims, dims), fill_value = feat, device = first_feature_tensor.device)
        for feat, dims in merge_spec
    ]

    new_inst = {k: v for k, v in inst.items() if not k.startswith(merge_prefix + '.')}
    new_inst[merged_feature_key or merge_prefix] = torch.cat(to_cat, dim = -1)
    return new_inst

def generic_to_nora_instance(
    batch: dict[str, Any],
    merge_spec: MergeSpec,
    action_prefix: str,
    state_prefix: str,
    action_dim_is_pad: torch.Tensor,
    embodiment_prompt: str,
):
    batch = merge_features(batch, action_prefix, merge_spec, 'action')
    batch = merge_features(batch, state_prefix, merge_spec, 'observation.state')
    batch['action_dim_is_pad'] = action_dim_is_pad
    batch['info'] = {"embodiment_prompt": embodiment_prompt}
    return batch

def agibot_world_to_nora_instance(batch: dict[str, Any]):
    """
    Convert from raw AgiBot World dataset format to format that is ready to be converted to `EnvTransition`:
    - Merge relevant actions into `action`, discarding other actions.
    - Merge relevant states into `observation.state`, discarding other states.
    - Invert the gripper action by 1-x.
    """
    batch['actions.effector.position'] = 1 - batch['actions.effector.position']
    return generic_to_nora_instance(
        batch,
        merge_spec = [
            ('joint.position', slice(0, 7)),
            ('effector.position', slice(0, 1)),
            ('joint.position', slice(7, 14)),
            ('effector.position', slice(1, 2)),
        ],
        action_prefix = 'actions',
        state_prefix = 'observation.states',
        action_dim_is_pad = ACTION_DIM_IS_PAD['dual_arm_7dof'],
        embodiment_prompt = "AgiBot G1 with 2 grippers",
    )

def galaxea_to_nora_instance(batch: dict[str, Any]):
    """
    Convert from raw Galaxea Open World Dataset format to format that is ready to be converted to `EnvTransition`:
    - Merge relevant actions into `action`, discarding other actions.
    - Merge relevant states into `observation.state`, discarding other states.
    """
    batch['observation.state.left_gripper'] = batch['observation.state.left_gripper'].view(-1)
    batch['observation.state.right_gripper'] = batch['observation.state.right_gripper'].view(-1)
    batch['action.left_gripper'] = batch['action.left_gripper'].view(-1, 1)
    batch['action.right_gripper'] = batch['action.right_gripper'].view(-1, 1)
    batch = generic_to_nora_instance(
        batch,
        merge_spec = [
            ('left_arm', slice(None)),
            (0.0, 1),
            ('left_gripper', slice(None)),
            ('right_arm', slice(None)),
            (0.0, 1),
            ('right_gripper', slice(None)),
        ],
        action_prefix = 'action',
        state_prefix = 'observation.state',
        action_dim_is_pad = ACTION_DIM_IS_PAD['dual_arm_6dof'],
        embodiment_prompt = "Galaxea R1 Lite",
    )
    # rename image keys, drop right head image
    batch['observation.images.head'] = batch['observation.images.head_rgb']
    batch['observation.images.hand_left'] = batch['observation.images.left_wrist_rgb']
    batch['observation.images.hand_right'] = batch['observation.images.right_wrist_rgb']
    del batch['observation.images.head_rgb']
    del batch['observation.images.head_right_rgb']
    del batch['observation.images.left_wrist_rgb']
    del batch['observation.images.right_wrist_rgb']
    return batch

def interndata_a1_to_nora_instance(
    batch: dict[str, Any],
    merge_spec: MergeSpec,
    action_dim_is_pad: torch.Tensor,
    embodiment_prompt: str,
):
    for key in batch:
        if key.startswith('states.') and key.endswith('gripper.position'):
            batch[key] = batch[key].view(-1)
        elif key.startswith('actions.') and key.endswith('gripper.position'):
            batch[key] = batch[key].view(-1, 1)

    batch = generic_to_nora_instance(
        batch,
        merge_spec = merge_spec,
        action_prefix = 'actions',
        state_prefix = 'states',
        action_dim_is_pad = action_dim_is_pad,
        embodiment_prompt = embodiment_prompt,
    )
    batch = {k.replace('images.rgb.', 'observation.images.'): v for k, v in batch.items()}
    return batch

interndata_a1_genie1_to_nora_instance = functools.partial(
    interndata_a1_to_nora_instance,
    merge_spec = [
        ('left_joint.position', slice(None)),
        ('left_gripper.position', slice(None)),
        ('right_joint.position', slice(None)),
        ('right_gripper.position', slice(None)),
    ],
    action_dim_is_pad = ACTION_DIM_IS_PAD['dual_arm_7dof'],
    embodiment_prompt = "InternData-A1 simulated Genie1 with 2 grippers",
)
interndata_a1_lift2_to_nora_instance = functools.partial(
    interndata_a1_to_nora_instance,
    merge_spec = [
        ('left_joint.position', slice(None)),
        (0.0, 1),
        ('left_gripper.position', slice(None)),
        ('right_joint.position', slice(None)),
        (0.0, 1),
        ('right_gripper.position', slice(None)),
    ],
    action_dim_is_pad = ACTION_DIM_IS_PAD['dual_arm_6dof'],
    embodiment_prompt = "InternData-A1 simulated Lift-2 with R5a arms",
)
interndata_a1_split_aloha_to_nora_instance = functools.partial(
    interndata_a1_to_nora_instance,
    merge_spec = [
        ('left_joint.position', slice(None)),
        (0.0, 1),
        ('left_gripper.position', slice(None)),
        ('right_joint.position', slice(None)),
        (0.0, 1),
        ('right_gripper.position', slice(None)),
    ],
    action_dim_is_pad = ACTION_DIM_IS_PAD['dual_arm_6dof'],
    embodiment_prompt = "InternData-A1 simulated Split Aloha with 2 Piper-100 arms",
)

def interndata_a1_franka_to_nora_instance(batch: dict[str, Any]):
    batch = interndata_a1_to_nora_instance(
        batch,
        merge_spec = [
            ('joint.position', slice(None)),
            ('gripper.position', slice(None)),
            (0.0, 8),
        ],
        action_dim_is_pad = ACTION_DIM_IS_PAD['single_arm_7dof'],
        embodiment_prompt = "InternData-A1 simulated Franka Emika Panda",
    )
    batch['observation.images.hand_left'] = batch['observation.images.hand']
    batch['observation.images.hand_right'] = None
    del batch['observation.images.hand']
    return batch

def load_agibot_world_dataset(
    root: str,
    canonical_action_chunk_size: int,
):
    return load_dataset(
        root,
        ("actions.joint.position", "actions.effector.position"),
        canonical_action_chunk_size,
        canonical_action_chunk_size,
        raw_fps = 30,
        instance_transform = agibot_world_to_nora_instance,
        norm_stats_transform = lambda norm_stats: {
            "action": {
                "q01": np.append(np.insert(norm_stats['actions.joint.position']['q01'], 7, 0), 0),
                "q99": np.append(np.insert(norm_stats['actions.joint.position']['q99'], 7, 1), 1),
            }
        },
    )

def load_galaxea_dataset(
    root: str,
    canonical_action_chunk_size: int,
):
    assert canonical_action_chunk_size % 2 == 0
    return load_dataset(
        root,
        ("action.left_arm", "action.left_gripper", "action.right_arm", "action.right_gripper"),
        canonical_action_chunk_size // 2,
        canonical_action_chunk_size,
        raw_fps = 15,
        instance_transform = galaxea_to_nora_instance,
        norm_stats_transform = lambda norm_stats:
            {
                "action": {
                    "q01": np.concatenate([
                        norm_stats['action.left_arm']['q01'],
                        np.array([0.0, 0.0]),
                        norm_stats['action.right_arm']['q01'],
                        np.array([0.0, 0.0]),
                    ]),
                    "q99": np.concatenate([
                        norm_stats['action.left_arm']['q99'],
                        np.array([0.0, 1.0]),
                        norm_stats['action.right_arm']['q99'],
                        np.array([0.0, 1.0]),
                    ]),
                }
            },
    )

def load_interndata_a1_dataset(
    root: str | pathlib.Path,
    canonical_action_chunk_size: int,
):
    root = pathlib.Path(root)

    franka_dataset = load_dataset(
        root / 'franka',
        ("actions.joint.position", "actions.gripper.position"),
        canonical_action_chunk_size,
        canonical_action_chunk_size,
        raw_fps = 30,
        instance_transform = interndata_a1_franka_to_nora_instance,
        norm_stats_transform = lambda norm_stats: {
            "action": {
                "q01": np.concatenate([
                    norm_stats['actions.joint.position']['q01'],
                    np.array([0.0] * 9),
                ]),
                "q99": np.concatenate([
                    norm_stats['actions.joint.position']['q99'],
                    np.array([1.0] * 9),
                ]),
            }
        },
    )
    dual_arm_action_keys = (
        "actions.left_joint.position",
        "actions.left_gripper.position",
        "actions.right_joint.position",
        "actions.right_gripper.position",
    )
    genie1_dataset = load_dataset(
        root / 'genie1',
        dual_arm_action_keys,
        canonical_action_chunk_size,
        canonical_action_chunk_size,
        raw_fps = 30,
        instance_transform = interndata_a1_genie1_to_nora_instance,
        norm_stats_transform = lambda norm_stats: {
            "action": {
                "q01": np.concatenate([
                    norm_stats['actions.left_joint.position']['q01'],
                    np.array([0.0]),
                    norm_stats['actions.right_joint.position']['q01'],
                    np.array([0.0]),
                ]),
                "q99": np.concatenate([
                    norm_stats['actions.left_joint.position']['q99'],
                    np.array([1.0]),
                    norm_stats['actions.right_joint.position']['q99'],
                    np.array([1.0]),
                ]),
            }
        },
    )
    dual_arm_6dof_norm_stats_transform = lambda norm_stats: {
        "action": {
            "q01": np.concatenate([
                norm_stats['actions.left_joint.position']['q01'],
                np.array([0.0, 0.0]),
                norm_stats['actions.right_joint.position']['q01'],
                np.array([0.0, 0.0]),
            ]),
            "q99": np.concatenate([
                norm_stats['actions.left_joint.position']['q99'],
                np.array([0.0, 1.0]),
                norm_stats['actions.right_joint.position']['q99'],
                np.array([0.0, 1.0]),
            ]),
        }
    }
    lift2_dataset = load_dataset(
        root / 'lift2',
        dual_arm_action_keys,
        canonical_action_chunk_size,
        canonical_action_chunk_size,
        raw_fps = 30,
        instance_transform = interndata_a1_lift2_to_nora_instance,
        norm_stats_transform = dual_arm_6dof_norm_stats_transform,
    )
    split_aloha_dataset = load_dataset(
        root / 'split_aloha',
        dual_arm_action_keys,
        canonical_action_chunk_size,
        canonical_action_chunk_size,
        raw_fps = 30,
        instance_transform = interndata_a1_split_aloha_to_nora_instance,
        norm_stats_transform = dual_arm_6dof_norm_stats_transform,
    )
    return ConcatDataset([franka_dataset, genie1_dataset, lift2_dataset, split_aloha_dataset])
