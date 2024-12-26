from __future__ import annotations

import os
import sys
import warnings
import math
import logging

import torch
from peft import get_peft_model
from transformers import PreTrainedModel

from ..base import BaseModel
from .prompt import Qwen2VLPromptMixin
from ...smp import get_rank_and_world_size, get_gpu_memory, auto_split_flag


def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image;']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')


def ensure_video_url(video: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:video;']
    if any(video.startswith(prefix) for prefix in prefixes):
        return video
    if os.path.exists(video):
        return 'file://' + video
    raise ValueError(f'Invalid video: {video}')

def load_model_with_adapters(
    base_model: PreTrainedModel,
    adapter_path: str
) -> PreTrainedModel:
    """
    Load LoRA and MoE adapters into a base model.
    
    Args:
        base_model: Pre-trained model to add adapters to
        adapter_path: Path to the adapter weights file
        
    Returns:
        Model with adapters loaded
    """
    # Configure adapter settings
    target_modules_for_lora = ["q_proj", "k_proj", "v_proj"]
    target_modules_for_moe = ["o_proj", "gate_proj", "up_proj", "down_proj"]
    num_experts = 4
    g_enable = True
    from peft import MoeConfig
    # LoRA config
    lora_config = MoeConfig(
        r=256,
        lora_alpha=512,
        target_modules=target_modules_for_lora,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        modules_to_save=None,
    )
    
    # MoE config
    moe_config = MoeConfig(
        r=256,
        lora_alpha=512,
        target_modules=target_modules_for_moe,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        modules_to_save=None,
        multiple_loras=True,
        g_enable=g_enable,
        noise_std=0.1,
        gates_tmp=1.0,
        topk=1,
        num_experts=num_experts,
        loss_coef=0,
        token=False,
        freeze_gate=True,
    )
    import time
    # Add adapters
    print("Adding LoRA adapter...")
    start_time = time.time()
    model = get_peft_model(base_model, lora_config, adapter_name='default')
    print(f"LoRA adapter added in {time.time() - start_time:.2f}s")

    print("Adding MoE adapters...")
    start_time = time.time()
    for i in range(num_experts):
        model.add_adapter(str(i), moe_config)
    print(f"MoE experts added in {time.time() - start_time:.2f}s")

    if g_enable:
        print("Adding gating adapter...")
        start_time = time.time()
        model.add_adapter("g", moe_config)
        print(f"Gating adapter added in {time.time() - start_time:.2f}s")
    
    # Load weights
    print("Loading adapter weights...")
    start_time = time.time()
    ckpt = torch.load(adapter_path)
    model.load_state_dict(ckpt, strict=True)
    print(f"Weights loaded in {time.time() - start_time:.2f}s")
    
    return model

def find_n_position(target_list, target_value, n):
    count = 0
    for i, element in enumerate(target_list):
        if element == target_value:
            count += 1
            if count == n:
                return i
        
    return -1

def process_prompt_pos(inputs, images):
    vision_start_id = 151652
    vision_end_id = 151653
    im_start_id = 151644
    im_end_id = 151645
    prompt_pos = [[0,0]]
    input_ids = inputs["input_ids"][0].tolist()
    if images:
        all_vision_starts = [i for i, x in enumerate(input_ids) if x == vision_start_id]
        start_pos = all_vision_starts[-1] if all_vision_starts else -1
    else:
        start_pos = find_n_position(input_ids, im_start_id, 2) + 2
    end_pos = find_n_position(input_ids, im_end_id, 2)
    assert end_pos != -1, "end_pos error!"
    assert start_pos != -1,  "start_pos error!"
    prompt_pos[0][0] = start_pos
    prompt_pos[0][1] = end_pos
    inputs["prompt_pos"] = torch.tensor(prompt_pos)
    return inputs

class Awaker(Qwen2VLPromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path: str,
        adapter_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        max_new_tokens=2048,
        top_p=0.001,
        top_k=1,
        temperature=0.01,
        repetition_penalty=1.0,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        verbose: bool = False,
    ):
        super().__init__(use_custom_prompt=use_custom_prompt)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.fps = 2.0
        self.nframe = 64
        self.FRAME_FACTOR = 2

        from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
        rank, world_size = get_rank_and_world_size()
        device = f'cuda:{rank}'

        assert model_path is not None
        self.model_path = model_path
        assert adapter_path is not None
        self.adapter_path = adapter_path
        self.processor = Qwen2VLProcessor.from_pretrained(model_path)

        gpu_mems = get_gpu_memory()
        max_gpu_mem = max(gpu_mems) if gpu_mems != [] else -1
        assert max_gpu_mem > 0

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype='auto',
            device_map=device,
            # attn_implementation='flash_attention_2'
        )
        self.model = load_model_with_adapters(self.model, adapter_path)
        self.model.to(device)
        self.model.eval()

    

        torch.cuda.empty_cache()

    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if dataset == 'OCRBench':
                    item['min_pixels'] = 10 * 10 * 28 * 28
                    warnings.warn(f"OCRBench dataset uses custom min_pixels={item['min_pixels']}")
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
            elif s['type'] == 'video':
                item = {'type': 'video', 'video': ensure_video_url(s['value'])}
                if self.fps is not None:
                    item['fps'] = self.fps
                elif self.nframe is not None:
                    import cv2
                    video = cv2.VideoCapture(s['value'])
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.release()
                    if frame_count < self.nframe:
                        new_frame_count = frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR
                        print(f"use {new_frame_count} for {s['value']}")
                        item['nframes'] = new_frame_count
                    else:
                        item['nframes'] = self.nframe
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content

    def generate_inner(self, message, dataset=None):
        try:
            from qwen_vl_utils import process_vision_info
        except Exception as err:
            logging.critical("qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'")
            raise err

        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        text = self.processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info([messages])
        inputs = self.processor(text=text, images=images, videos=videos, padding=True, return_tensors='pt')
        inputs = process_prompt_pos(inputs, images)

        inputs = inputs.to('cuda')

        generated_ids = self.model.generate(
            **inputs,
            **self.generate_kwargs,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        out = self.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = out[0]
        if self.verbose:
            print(f'\033[32m{response}\033[0m')
        return response
