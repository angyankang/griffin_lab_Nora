import sys
import pathlib

_root = pathlib.Path(__file__).resolve().parent.parent  # repo root
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import math
import os
import logging
from typing import List, Any, Optional
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, default_collate

from accelerate import Accelerator
from accelerate.logging import get_logger

# --- Updated Qwen3 Imports ---
from transformers import AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from transformers import get_scheduler

import lerobot.processor
from lerobot.configs.types import  PipelineFeatureType
from qwen_vl_utils import process_vision_info
import numpy as np
from tqdm import tqdm

import torchvision

from utils.data_loading import load_dataset

logger = get_logger(__name__)

# --- 1. Configuration ---
@dataclass
class TrainingConfig:
    per_device_batch_size: int = 256
    learning_rate: float = 5e-5
    gradient_accumulation_steps: int = 1
    num_warmup_steps: int = 1000
    max_epochs: int = 5
    output_dir: str = './nora_finetune_object'
    resume_from_checkpoint: str = ''
    load_model_weights: Optional[str] = None
    agibot_world_root: str = "data/agibot/tasks"
    galaxea_open_world_ds_root: str = "data/galaxea/subsets"
    wandb_project_name: str = "Nora VLA with LeRobotDataset"
    checkpoint_save_frequency: int = 20000
    logging_frequency: int = 100
    gradient_clipping: Optional[float] = None
    dataloader_num_workers: int = 4
    image_augmentation: bool = True
    action_chunk_size: int = 50
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct" 
    action_vocab_size: int = 2048

# --- 2. Data Loading and Preprocessing ---
def load_agibot_world_dataset(
    root: str,
    canonical_action_chunk_size: int,
    use_image_augmentation: bool,
):
    return load_dataset(
        root,
        ("actions.joint.position", "actions.effector.position"),
        canonical_action_chunk_size,
        canonical_action_chunk_size,
        use_image_augmentation,
        raw_fps = 30,
        aspect_ratio = 4/3,
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
    use_image_augmentation: bool,
):
    assert canonical_action_chunk_size % 2 == 0
    return load_dataset(
        root,
        ("action.left_arm", "action.left_gripper", "action.right_arm", "action.right_gripper"),
        canonical_action_chunk_size // 2,
        canonical_action_chunk_size,
        use_image_augmentation,
        raw_fps = 15,
        aspect_ratio = 16/9,
        instance_transform = galaxea_to_nora_instance,
        norm_stats_transform = lambda norm_stats:
            {
                "action": {
                    "q01": np.concatenate([
                        norm_stats['action.left_arm']['q01'],
                        np.array([-1.0, 0.0]),
                        norm_stats['action.right_arm']['q01'],
                        np.array([-1.0, 0.0]),
                    ]),
                    "q99": np.concatenate([
                        norm_stats['action.left_arm']['q99'],
                        np.array([1.0, 1.0]),
                        norm_stats['action.right_arm']['q99'],
                        np.array([1.0, 1.0]),
                    ]),
                }
            },
    )

def agibot_world_to_nora_instance(batch: dict[str, Any]):
    """
    Convert from raw AgiBot World dataset format to format that is ready to be converted to `EnvTransition`:
    - Merge relevant actions into `action`, discarding other actions.
    - Merge relevant states into `observation.state`, discarding other states.
    - Invert the gripper action by 1-x.
    """
    prev_dim_sizes = batch['actions.joint.position'].shape[:-1]
    action = torch.cat(
        [
            batch['actions.joint.position'].view(*prev_dim_sizes, 2, 7),
            1 - batch['actions.effector.position'].view(*prev_dim_sizes, 2, 1)
        ],
        dim = -1
    ).view(*prev_dim_sizes, 16)
    state = torch.cat(
        [
            batch['observation.states.joint.position'].view(2, 7),
            # Effector position below is only used for its shape, the values should be unused through the
            # delta transform mask. The values are in meters rather than [0, 1] so not actually comparable.
            batch['observation.states.effector.position'].view(2, 1),
        ],
        dim = -1
    ).view(16)
    batch = {k: v for k, v in batch.items() if not k.startswith('actions.') and not k.startswith('observation.states.')}
    batch['action'] = action
    batch['observation.state'] = state
    batch['action_dim_is_pad'] = torch.zeros(action.shape[-1], dtype=torch.bool)
    batch['info'] = {"embodiment_prompt": "AgiBot G1 with 2 grippers"}
    return batch

def galaxea_to_nora_instance(batch: dict[str, Any]):
    """
    Convert from raw Galaxea Open World Dataset format to format that is ready to be converted to `EnvTransition`:
    - Merge relevant actions into `action`, discarding other actions.
    - Merge relevant states into `observation.state`, discarding other states.
    """
    zero_padding = torch.zeros((*batch['action.left_arm'].shape[:-1], 1), dtype=batch['action.left_arm'].dtype)
    action = torch.cat(
        [
            batch['action.left_arm'],
            zero_padding,
            batch['action.left_gripper'].unsqueeze(-1),
            batch['action.right_arm'],
            zero_padding,
            batch['action.right_gripper'].unsqueeze(-1),
        ],
        dim = -1
    )
    zero_padding = torch.zeros((1,), dtype=batch['action.left_arm'].dtype)
    state = torch.cat(
        [
            batch['observation.state.left_arm'],
            zero_padding,
            batch['observation.state.left_gripper'].unsqueeze(-1),
            batch['observation.state.right_arm'],
            zero_padding,
            batch['observation.state.right_gripper'].unsqueeze(-1),
        ],
        dim = -1
    )
    batch = {k: v for k, v in batch.items() if not k.startswith('action.') and not k.startswith('observation.state.')}
    batch['action'] = action
    batch['observation.state'] = state
    batch['action_dim_is_pad'] = torch.tensor(
        [False, False, False, False, False, False, True, False] * 2,
    )
    batch['info'] = {"embodiment_prompt": "Galaxea R1 Lite"}
    # rename image keys, drop right head image
    batch['observation.images.head'] = batch['observation.images.head_rgb']
    batch['observation.images.hand_left'] = batch['observation.images.left_wrist_rgb']
    batch['observation.images.hand_right'] = batch['observation.images.right_wrist_rgb']
    del batch['observation.images.head_rgb']
    del batch['observation.images.head_right_rgb']
    del batch['observation.images.left_wrist_rgb']
    del batch['observation.images.right_wrist_rgb']
    return batch

def map_fast_token_to_vlm_action(tokens: List[str]) -> str:
    """Maps fast action tokens to the VLM action format."""
    return ''.join([f"<robot_action_{token}>" for token in tokens])

@dataclass
@lerobot.processor.ProcessorStepRegistry.register("nora_processor")
class NoraPolicyProcessorStep(lerobot.processor.ProcessorStep):
    # Added to accept Qwen3 config and processor
    config: TrainingConfig 
    transformer_processor: Any 
    action_token_min: int = -1
    action_token_max: int = -1

    IMAGE_KEYS = (
        'observation.images.head',
        'observation.images.hand_left',
        'observation.images.hand_right',
    )

    def __post_init__(self):
        self.fast_tokenizer = AutoProcessor.from_pretrained(
            "physical-intelligence/fast", trust_remote_code=True
        )
        # Dynamically calculate Action Token range to replace the original hardcoded values (151665 ~ 153712)
        action_tokens = [f"<robot_action_{i}>" for i in range(self.config.action_vocab_size)]
        action_ids = self.transformer_processor.tokenizer.convert_tokens_to_ids(action_tokens)
        action_ids = [id for id in action_ids if id is not None and id != self.transformer_processor.tokenizer.unk_token_id]
        
        if action_ids:
            self.action_token_min = min(action_ids)
            self.action_token_max = max(action_ids)
        else:
            raise ValueError("Action Tokens not found in the Qwen3 vocabulary! Please ensure they were injected during the model loading phase.")

    def __call__(self, transition: lerobot.processor.EnvTransition) -> lerobot.processor.EnvTransition:
        # list of lists, shape (batch_size, n_keys)
        per_sample_images = [
            [
                torchvision.transforms.functional.to_pil_image(img)
                for img in image_tuple
            ]
            for image_tuple in zip(*(transition['observation'][k] for k in self.IMAGE_KEYS))
        ]

        fast_tokens = []
        for i in range(transition['action'].shape[0]):
            action = transition['action'][i]
            action = action[:, transition['complementary_data']['action_dim_is_pad'][i].logical_not()]
            fast_tokens.extend(self.fast_tokenizer(action.cpu()))

        vlm_action = [map_fast_token_to_vlm_action(ft) for ft in fast_tokens]
        tasks = transition['complementary_data']['task']
        embodiment_prompts = transition['info']['embodiment_prompt']

        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        *(
                            {"type": "image", "image": img, "resized_height": 224, "resized_width": 224}
                            for img in imgs
                        ),
                        {"type": "text", "text": f"[embodiment: {embodiment}] {task}"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": act},
                    ],
                }
            ]
            for imgs, task, embodiment, act in zip(per_sample_images, tasks, embodiment_prompts, vlm_action)
        ]

        text = self.transformer_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        image_inputs, video_inputs = process_vision_info(messages)
        batch_input = self.transformer_processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        labels = batch_input['input_ids'].clone()

        # Use dynamically calculated action_token_min/max
        for i in range(labels.size(0)):
            seq = labels[i]
            mask_seq = (seq >= self.action_token_min) & (seq <= self.action_token_max)
            nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
            if nonzero_indices.numel() > 0:
                first_action_index = nonzero_indices[0].item()
                seq[:first_action_index] = -100
            else:
                seq[:] = -100

        labels[labels == self.transformer_processor.tokenizer.pad_token_id] = -100
        batch_input['labels'] = labels

        return lerobot.processor.create_transition(complementary_data = batch_input)

    def transform_features(self, features):
        return {
            PipelineFeatureType.ACTION: {},
            PipelineFeatureType.OBSERVATION: {},
        }

def make_policy_processor(config: TrainingConfig, transformer_processor: Any) -> lerobot.processor.PolicyProcessorPipeline:
    # Pass config and processor to the step
    return lerobot.processor.PolicyProcessorPipeline(
        steps = [NoraPolicyProcessorStep(config=config, transformer_processor=transformer_processor)],
        to_output=lambda tr: tr['complementary_data'],
    )


# --- 3. Model Initialization ---
def load_model_and_processor(config: TrainingConfig, accelerator: Accelerator):
    """Loads Qwen3, its processor, and injects action tokens."""
    transformer_processor = AutoProcessor.from_pretrained(config.model_id, trust_remote_code=True)
    transformer_processor.tokenizer.padding_side = 'left'

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        config.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    )

    # Inject action tokens
    action_tokens = [f"<robot_action_{i}>" for i in range(config.action_vocab_size)]
    transformer_processor.tokenizer.add_tokens(action_tokens, special_tokens=True)
    model.resize_token_embeddings(len(transformer_processor.tokenizer))
    accelerator.print(f"Added {len(action_tokens)} action tokens to Qwen3 vocabulary.")

    if config.load_model_weights:
        tensors = {}
        from safetensors import safe_open
        with safe_open(config.load_model_weights, framework="pt") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        model.load_state_dict(tensors, strict=False)
        accelerator.print("Pretrained weights loaded.")

    return model, transformer_processor

# --- 4. Training Loop ---
def train(config: TrainingConfig):
    """Main training loop."""
    accelerator = Accelerator(gradient_accumulation_steps=config.gradient_accumulation_steps,log_with="wandb")
    accelerator.dataloader_config.dispatch_batches = False
    logger.info(accelerator.state, main_process_only=False)

    accelerator.init_trackers(config.wandb_project_name, config=config)

    # Unpack both model and processor
    model, transformer_processor = load_model_and_processor(config, accelerator)

    with accelerator.main_process_first():
        agibot_world = load_agibot_world_dataset(
            root = config.agibot_world_root,
            canonical_action_chunk_size = config.action_chunk_size,
            use_image_augmentation = config.image_augmentation,
        )
        galaxea_open_world_ds = load_galaxea_dataset(
            root = config.galaxea_open_world_ds_root,
            canonical_action_chunk_size = config.action_chunk_size,
            use_image_augmentation = config.image_augmentation,
        )
        dataset = agibot_world + galaxea_open_world_ds
        policy_preprocessor = make_policy_processor(config, transformer_processor)

    train_dataloader = DataLoader(
        dataset,
        batch_size=config.per_device_batch_size,
        collate_fn=lambda examples: policy_preprocessor(default_collate(examples)),
        shuffle=True,
        num_workers=config.dataloader_num_workers,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=1e-8,
        eps=1e-8,
    )

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    max_train_steps = len(train_dataloader) * config.max_epochs
    max_optim_steps = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps) * config.max_epochs
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=math.ceil(config.num_warmup_steps / config.gradient_accumulation_steps),
        num_training_steps=max_optim_steps
    )

    lr_scheduler = accelerator.prepare(lr_scheduler)

    if config.resume_from_checkpoint:
        accelerator.load_state(config.resume_from_checkpoint)
        accelerator.print(f"Resumed from local checkpoint: {config.resume_from_checkpoint}")

    total_batch_size = config.per_device_batch_size * accelerator.num_processes * config.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num steps = {max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {config.per_device_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_optim_steps}")

    completed_steps = 0
    progress_bar = tqdm(range(completed_steps, max_train_steps), disable=not accelerator.is_local_main_process)

    while completed_steps < max_train_steps:
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                
                # Filter dictionary keys to prevent unexpected kwargs to Qwen3
                model_inputs = {
                    'input_ids': batch['input_ids'],
                    'attention_mask': batch['attention_mask'],
                    'pixel_values': batch['pixel_values'],
                    'image_grid_thw': batch['image_grid_thw'],
                    'labels': batch['labels']
                }
                
                outputs = model(**model_inputs)
                loss = outputs.loss

                accelerator.backward(loss)

                progress_bar.update(1)
                completed_steps += 1

                if accelerator.sync_gradients:
                    if config.gradient_clipping is not None:
                        accelerator.clip_grad_norm_(model.parameters(), config.gradient_clipping)

                    optimizer.step()
                    lr_scheduler.step()

                    if completed_steps % config.logging_frequency == 0:
                        if accelerator.is_main_process:
                            total_norm = 0.0
                            for p in model.parameters():
                                if p.grad is not None:
                                    total_norm += p.grad.data.norm(2).item() ** 2

                            total_norm = total_norm**0.5
                            lr = lr_scheduler.get_last_lr()[0]

                            logger.info(f"Step {completed_steps}, Loss: {loss.item()}, Grad Norm: {total_norm}", main_process_only=True)
                            accelerator.log({"train_loss": loss.item(), "learning_rate": lr,"grad_norm":total_norm}, step=completed_steps)

            if completed_steps % config.checkpoint_save_frequency == 0 and completed_steps > 0:
                accelerator.save_state(os.path.join(config.output_dir, f"steps_{completed_steps}"))

            if completed_steps >= max_train_steps:
                break

    accelerator.save_state(os.path.join(config.output_dir, f"steps_{completed_steps}"))
    if accelerator.is_main_process:
        checkpoint_path = os.path.join(config.output_dir, f"steps_{completed_steps}")
        logger.info(f"Training finished. Final checkpoint saved at {checkpoint_path}")


def main():
    config = TrainingConfig()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    train(config)

if __name__ == "__main__":
    main()