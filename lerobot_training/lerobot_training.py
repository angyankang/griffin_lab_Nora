from functools import lru_cache
import sys
import pathlib

from torch.utils.data.dataset import ConcatDataset

_root = pathlib.Path(__file__).resolve().parent.parent  # repo root
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import math
import os
import logging
from typing import List, Any, Optional
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T_v2

from accelerate import Accelerator
from accelerate.logging import get_logger

from transformers import AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from transformers import get_scheduler

import lerobot.processor
from lerobot.configs.types import PipelineFeatureType
from qwen_vl_utils import process_vision_info
import numpy as np
from tqdm import tqdm

import load_datasets
from utils.data_loading import collate_with_observation_image_lists, load_dataset
import lerobot.processor.converters

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
    agibot_world_root: str = "data/agibot-world/tasks"
    galaxea_open_world_ds_root: str = "data/galaxea-open-world-dataset/subsets"
    interndata_a1_root: str = "data/interndata-a1/"
    wandb_project_name: str = "Nora VLA with LeRobotDataset"
    checkpoint_save_frequency: int = 20000
    logging_frequency: int = 100
    gradient_clipping: Optional[float] = None
    dataloader_num_workers: int = 4
    action_chunk_size: int = 50
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct" 
    action_vocab_size: int = 2048
    image_target_pixels: int = 65536   # mimimum size for Qwen3 VL, corresponds to 256 patches
    """Approximate target number of pixels in the resized image that the model receives."""
    
    # Number of frames to input (5 past + 1 current = 6)
    num_frames: int = 6


# --- 2. Data Preprocessing & Transforms ---
def map_fast_token_to_vlm_action(tokens: List[str]) -> str:
    """Maps fast action tokens to the VLM action format."""
    return ''.join([f"<robot_action_{token}>" for token in tokens])

@dataclass
class NoraImageTransform:
    target_pixels: int
    patch_size: int
    merge_size: int
    min_pixels: int
    max_pixels: int

    def __post_init__(self):
        self.color_jitter = T_v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)

    @staticmethod
    @lru_cache
    def _get_random_resized_crop_transform(
        target_pixels: int,
        patch_size: int,
        merge_size: int,
        min_pixels: int,
        max_pixels: int,
        aspect_ratio: float,
    ) -> T_v2.RandomResizedCrop:
        resize_target = smart_resize(
            (target_pixels / aspect_ratio)**0.5,
            (target_pixels * aspect_ratio)**0.5,
            factor = patch_size * merge_size,
            min_pixels = min_pixels,
            max_pixels = max_pixels,
        )
        return T_v2.RandomResizedCrop(
            size=resize_target,
            scale=(0.9, 0.9),
            ratio=(aspect_ratio, aspect_ratio),
        )

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        aspect_ratio = image.shape[-1] / image.shape[-2]
        random_resized_crop_transform = self._get_random_resized_crop_transform(
            self.target_pixels,
            self.patch_size,
            self.merge_size,
            self.min_pixels,
            self.max_pixels,
            aspect_ratio,
        )
        image = random_resized_crop_transform(image)
        image = self.color_jitter(image)
        return image


@dataclass
@lerobot.processor.ProcessorStepRegistry.register("nora_processor")
class NoraPolicyProcessorStep(lerobot.processor.ProcessorStep):
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
        # Dynamically calculate Action Token range
        action_tokens = [f"<robot_action_{i}>" for i in range(self.config.action_vocab_size)]
        action_ids = self.transformer_processor.tokenizer.convert_tokens_to_ids(action_tokens)
        action_ids = [id for id in action_ids if id is not None and id != self.transformer_processor.tokenizer.unk_token_id]
        
        if action_ids:
            self.action_token_min = min(action_ids)
            self.action_token_max = max(action_ids)
        else:
            raise ValueError("Action Tokens not found in the Qwen3 vocabulary!")

        self.nora_image_transform = NoraImageTransform(
            target_pixels = self.config.image_target_pixels,
            patch_size = self.transformer_processor.image_processor.patch_size,
            merge_size = self.transformer_processor.image_processor.merge_size,
            min_pixels = self.transformer_processor.image_processor.size["shortest_edge"],
            max_pixels = self.transformer_processor.image_processor.size["longest_edge"],
        )

    def __call__(self, transition) -> lerobot.processor.EnvTransition:
        batch_size = transition['action'].shape[0]
        per_sample_images = []
        
        obs_dict = transition['observation'] if 'observation' in transition else transition
        
        for i in range(batch_size):
            imgs_for_this_sample = []
            
            first_cam = obs_dict[self.IMAGE_KEYS[0]][i]
            T = first_cam.shape[0] if first_cam.dim() == 4 else 1
            
            # 时序循环：提取历史帧并经过最新的 Transform 处理
            for t in range(T):
                for k in self.IMAGE_KEYS:
                    img_tensor = obs_dict[k][i]
                    frame = img_tensor[t] if img_tensor.dim() == 4 else img_tensor
                    
                    transformed_frame = self.nora_image_transform(frame) if frame is not None else None
                    imgs_for_this_sample.append(transformed_frame)
            
            per_sample_images.append(imgs_for_this_sample)

        fast_tokens = []
        for i in range(batch_size):
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
                        {"type": "text", "text": f"[embodiment: {embodiment}]"},
                        *(
                            {"type": "image", "image": img}
                            for img in imgs if img is not None
                        ),
                        {"type": "text", "text": task},
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

        batch_input = self.transformer_processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
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
        attn_implementation="sdpa",
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

    model, transformer_processor = load_model_and_processor(config, accelerator)

    with accelerator.main_process_first():
        agibot_world = load_datasets.load_agibot_world_dataset(
            root = config.agibot_world_root,
            canonical_action_chunk_size = config.action_chunk_size,
            use_image_augmentation = config.image_augmentation,
            num_frames = config.num_frames, 
        )
        galaxea_open_world_ds = load_datasets.load_galaxea_dataset(
            root = config.galaxea_open_world_ds_root,
            canonical_action_chunk_size = config.action_chunk_size,
            use_image_augmentation = config.image_augmentation,
            num_frames = config.num_frames, 
        )
        interndata_a1 = load_datasets.load_interndata_a1_dataset(
            root = config.interndata_a1_root,
            canonical_action_chunk_size = config.action_chunk_size,
        )
        dataset = ConcatDataset([agibot_world, galaxea_open_world_ds, interndata_a1])
        policy_preprocessor = make_policy_processor(config, transformer_processor)

    train_dataloader = DataLoader(
        dataset,
        batch_size=config.per_device_batch_size,
        collate_fn=lambda examples: policy_preprocessor(collate_with_observation_image_lists(examples)),
        shuffle=True,
        num_workers=config.dataloader_num_workers,
        pin_memory=True,
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