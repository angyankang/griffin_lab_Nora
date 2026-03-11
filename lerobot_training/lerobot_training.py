import sys
import pathlib

_root = pathlib.Path(__file__).resolve().parent.parent  # repo root
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import math
import os
import logging
from typing import Collection, List, Any, Optional
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, default_collate, Dataset, ConcatDataset

from accelerate import Accelerator
from accelerate.logging import get_logger
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers import get_scheduler

import lerobot.processor
import lerobot.datasets.utils
from lerobot.configs.types import  NormalizationMode, PipelineFeatureType
from qwen_vl_utils import process_vision_info
import numpy as np
from tqdm import tqdm

import torchvision
from utils.skip_episodes_lerobot_dataset import SkipEpisodesLeRobotDataset

logger = get_logger(__name__)

# --- 1. Configuration ---
class TrainingConfig:
    def __init__(
        self,
        per_device_batch_size: int = 1,  ##训练时候修改
        learning_rate: float = 5e-5,
        gradient_accumulation_steps: int = 16,   ##修改
        num_warmup_steps: int = 1000,
        max_epochs: int = 5,
        output_dir: str = './nora_finetune_object',
        resume_from_checkpoint: str = '',
        load_model_weights: Optional[str] = None,
        lerobot_dataset_repo_id: str | None = None,
        lerobot_dataset_root: str = "C:/Users/yankang.ang/Downloads/sample_dataset_lerobot_head_only/agibotworld",
        wandb_project_name: str = "Nora VLA with LeRobotDataset",
        checkpoint_save_frequency: int = 20000,
        logging_frequency: int = 100,
        gradient_clipping: Optional[float] = None,
        invert_grippler_action: bool = True,
        dataloader_num_workers: int = 0,
        # 【显式指定 Qwen 3.5】
        model_id: str = "Qwen/Qwen3.5-4B", 
        action_vocab_size: int = 1024,
    ):
        self.per_device_batch_size = per_device_batch_size
        self.learning_rate = learning_rate
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_warmup_steps = num_warmup_steps
        self.max_epochs = max_epochs
        self.output_dir = output_dir
        self.resume_from_checkpoint = resume_from_checkpoint
        self.load_model_weights = load_model_weights
        self.lerobot_dataset_repo_id = lerobot_dataset_repo_id
        self.lerobot_dataset_root = lerobot_dataset_root
        self.wandb_project_name = wandb_project_name
        self.checkpoint_save_frequency = checkpoint_save_frequency
        self.logging_frequency = logging_frequency
        self.gradient_clipping = gradient_clipping
        self.invert_grippler_action = invert_grippler_action
        self.dataloader_num_workers = dataloader_num_workers
        self.model_id = model_id
        self.action_vocab_size = action_vocab_size
        self.image_keys = (
            'observation.images.head',
        )
        self.action_key = 'action'
        self.task_key = 'task'
        self.fps = 30
        self.action_chunk_size = 50

# --- 原生 Action Tokenizer (替代报错的 FAST 库) ---
def native_action_tokenizer(action_tensor: torch.Tensor, vocab_size: int = 1024) -> List[List[str]]:
    """
    直接将连续动作 [0, 1] 映射为离散的 string token 列表，完全绕过 HF 依赖。
    action_tensor shape: [batch_size, chunk_size, action_dim]
    """
    clamped_actions = torch.clamp(action_tensor, 0.0, 0.9999)
    discrete_actions = (clamped_actions * vocab_size).int()
    
    result = []
    for batch_idx in range(discrete_actions.shape[0]):
        # 将每个 batch 下的 chunk 和 dim 展平为一维 token 序列
        tokens = discrete_actions[batch_idx].flatten().tolist()
        result.append([str(t) for t in tokens])
    return result

def map_fast_token_to_vlm_action(tokens: List[str]) -> str:
    return ''.join([f"<robot_action_{token}>" for token in tokens])

# --- 2. Data Loading and Preprocessing ---
def load_and_prepare_dataset(config: TrainingConfig) -> tuple[Dataset, dict[str, dict[str, np.ndarray]]]:
    delta_timestamps = [i / config.fps for i in range(-1, config.action_chunk_size)]
    delta_timestamps_dict = {
        "actions.joint.position": delta_timestamps,
        "actions.effector.position": delta_timestamps,
    }
    task_roots = list(pathlib.Path(config.lerobot_dataset_root).glob("task_*"))
    dataset = ConcatDataset([
        SkipEpisodesLeRobotDataset(task_root.name, root=task_root, delta_timestamps=delta_timestamps_dict)
        for task_root in tqdm(task_roots, desc="Loading AgiBot World Beta datasets")
    ])
    
    raw_norm_stats = lerobot.datasets.utils.cast_stats_to_numpy(
        lerobot.datasets.utils.load_json(pathlib.Path(config.lerobot_dataset_root) / 'norm_stats.json')
    )['norm_stats']
    
    norm_stats = {
        "action": {
            "min": np.append(np.insert(raw_norm_stats['actions.joint.position']['q01'], 7, 0), 0),
            "max": np.append(np.insert(raw_norm_stats['actions.joint.position']['q99'], 7, 1), 1),
        }
    }
    return dataset, norm_stats

def agibot_world_to_nora_instance(batch: dict[str, Any], img_keys: Collection[str]):
    images = {k: batch[k] for k in img_keys}
    prev_dim_sizes = batch['actions.joint.position'].shape[:2]
    action = torch.cat(
        [
            batch['actions.joint.position'].view(*prev_dim_sizes, 2, 7),
            1 - batch['actions.effector.position'].view(*prev_dim_sizes, 2, 1)
        ],
        dim = -1
    ).view(*prev_dim_sizes, 16)
    batch = {k: v for k, v in batch.items() if not k.startswith('actions.') and not k.startswith('observation.')}
    batch.update(images)
    batch['action'] = action
    return batch

@dataclass
@lerobot.processor.ProcessorStepRegistry.register("abs2delta_action_processor")
class Abs2DeltaActionProcessorStep(lerobot.processor.PolicyActionProcessorStep):
    mask: torch.Tensor

    def action(self, action):
        assert self.mask.shape[-1] == action.shape[-1]
        future_actions = action[...,1:,:]
        deltas = future_actions - action[...,:-1,:]
        return torch.where(self.mask.expand(future_actions.shape), deltas, future_actions)

    def transform_features(self, features):
        old_shape = features['ACTION']['action'].shape
        features['ACTION'].shape = old_shape[:-2] + (old_shape[-2] - 1, old_shape[-1]) 
        return features

@dataclass
@lerobot.processor.ProcessorStepRegistry.register("nora_processor")
class NoraPolicyProcessorStep(lerobot.processor.ProcessorStep):
    config: TrainingConfig
    transformer_processor: Any 

    def __post_init__(self):
        # 【彻底剥离 FAST 分词器依赖】
        self.transformer_processor.tokenizer.padding_side = 'left'

        # 动态获取 Qwen3.5 中 Action Token 的 ID 范围
        action_tokens = [f"<robot_action_{i}>" for i in range(self.config.action_vocab_size)]
        action_ids = self.transformer_processor.tokenizer.convert_tokens_to_ids(action_tokens)
        action_ids = [id for id in action_ids if id is not None and id != self.transformer_processor.tokenizer.unk_token_id]
        
        if action_ids:
            self.action_token_min = min(action_ids)
            self.action_token_max = max(action_ids)
            logger.info(f"成功定位 Qwen3.5 Action Token 范围: {self.action_token_min} 到 {self.action_token_max}")
        else:
            raise ValueError("在 Qwen3.5 词表中未找到 Action Tokens！")

    def __call__(self, transition: lerobot.processor.EnvTransition) -> lerobot.processor.EnvTransition:
        per_sample_images = [
            [
                torchvision.transforms.functional.to_pil_image(img)
                for img in image_tuple
            ]
            for image_tuple in zip(*(transition['observation'][k] for k in self.config.image_keys))
        ]

        action = transition['action']
        lang = transition['complementary_data']['task']
        
        # 【使用纯原生 Python 函数处理动作离散化】
        native_tokens_list = native_action_tokenizer(action.cpu(), vocab_size=self.config.action_vocab_size)
        vlm_action = [map_fast_token_to_vlm_action(ft) for ft in native_tokens_list]

        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        *(
                            {"type": "image", "image": img, "resized_height": 224, "resized_width": 224}
                            for img in imgs
                        ),
                        {"type": "text", "text": l},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": act},
                    ],
                }
            ]
            for imgs, l, act in zip(per_sample_images, lang, vlm_action)
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

def make_policy_processor(
        config: TrainingConfig,
        norm_stats: dict[str, dict[str, np.ndarray]],
        transformer_processor: Any
) -> lerobot.processor.PolicyProcessorPipeline:
    norm_map = {
        'ACTION': NormalizationMode.MIN_MAX,
    }

    return lerobot.processor.PolicyProcessorPipeline(
        steps = [
            Abs2DeltaActionProcessorStep(
                mask = torch.tensor(
                    [
                        True, True, True, True, True, True, True, False,
                        True, True, True, True, True, True, True, False,
                    ],
                    dtype=torch.bool,
                ),
            ),
            lerobot.processor.NormalizerProcessorStep({}, norm_map , norm_stats),
            NoraPolicyProcessorStep(config, transformer_processor),
        ],
        to_transition=lambda batch:
            lerobot.processor.converters.batch_to_transition(agibot_world_to_nora_instance(batch, config.image_keys)),
        to_output=lerobot.processor.converters.transition_to_batch,
    )

# --- 3. Model Initialization ---
def load_model_and_processor(config: TrainingConfig, accelerator: Accelerator, processor: Any):
    # 【显式调用基础大模型类，通吃所有 Qwen 3.5 变体】
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", # 避免 flash_attn 在 Windows 上的编译报错
        trust_remote_code=True
    )

    model.resize_token_embeddings(len(processor.tokenizer))
    accelerator.print(f"成功将 Qwen3.5 模型词表扩充至: {len(processor.tokenizer)}")

    if config.load_model_weights:
        tensors = {}
        from safetensors import safe_open
        with safe_open(config.load_model_weights, framework="pt") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        model.load_state_dict(tensors, strict=False)
        accelerator.print("Pretrained weights loaded.")

    return model

# --- 4. Training Loop ---
def train(config: TrainingConfig):
    accelerator = Accelerator(gradient_accumulation_steps=config.gradient_accumulation_steps, log_with="wandb")
    accelerator.dataloader_config.dispatch_batches = False
    logger.info(accelerator.state, main_process_only=False)

    accelerator.init_trackers(config.wandb_project_name, config=config)

    # 【直接初始化 Qwen 3.5 的 Processor】
    transformer_processor = AutoProcessor.from_pretrained(config.model_id, trust_remote_code=True)
    action_tokens = [f"<robot_action_{i}>" for i in range(config.action_vocab_size)]
    transformer_processor.tokenizer.add_tokens(action_tokens, special_tokens=True)
    accelerator.print(f"成功向 Qwen 3.5 的词表中注入 {len(action_tokens)} 个动作 Token。")

    model = load_model_and_processor(config, accelerator, transformer_processor)

    with accelerator.main_process_first():
        dataset, norm_stats = load_and_prepare_dataset(config)
        policy_preprocessor = make_policy_processor(config, norm_stats, transformer_processor)

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
    logger.info("***** Running Qwen 3.5 Training *****")
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
                outputs = model(**batch)
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