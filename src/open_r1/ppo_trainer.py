# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import math
import os
import textwrap
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from typing import Optional, Union
import torch.distributed as dist
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import broadcast, gather_object
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    BaseImageProcessor,
    DataCollatorWithPadding,
    FeatureExtractionMixin,
    GenerationConfig,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
    TrainerControl,
    is_wandb_available,
)
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer_callback import CallbackHandler, ExportableState, PrinterCallback
from transformers.utils import is_peft_available, PaddingStrategy, is_accelerate_available

from trl.core import masked_mean, masked_whiten
from trl.models import create_reference_model
from trl.models.utils import unwrap_model_for_generation
# from trl.trainer.ppo_config import PPOConfig
from configs import PPOConfig
from trl.trainer.utils import (
    OnlineTrainerState,
    batch_generation,
    disable_dropout_in_model,
    exact_div,
    first_true_indices,
    forward,
    generate_model_card,
    get_comet_experiment_url,
    get_reward,
    log_table_to_comet_experiment,
    peft_module_casting_to_bf16,
    prepare_deepspeed,
    print_rich_table,
    selective_log_softmax,
    truncate_response,
)
from typing import List, Dict, Any
import warnings
from unittest.mock import patch
from trl.import_utils import is_vllm_available
from accelerate.utils import is_peft_model
import time

if is_peft_available():
    from peft import PeftConfig, PeftModel, get_peft_model

if is_wandb_available():
    import wandb

if is_vllm_available():
    from vllm import LLM, SamplingParams

if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.state import AcceleratorState
    from accelerate.utils import (
        AutocastKwargs,
        DistributedDataParallelKwargs,
        DistributedType,
        load_fsdp_model,
        load_fsdp_optimizer,
        save_fsdp_model,
        save_fsdp_optimizer,
    )

from trainer import Trainer, DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK



INVALID_LOGPROB = 1.0
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from dataclasses import dataclass, field
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed

def padding_sequence(seqs, pad_val, dim=-1):
    device = seqs[0].device
    max_len = max([seq.size()[dim] for seq in seqs])

    seqs = [torch.cat([seq, torch.ones((seq.size()[0], max_len - seq.size()[dim]), device=device, dtype=seq.dtype)*pad_val], dim=dim) for seq in seqs]
    return torch.cat(seqs, dim=0)

def get_reward(
        model,
        query_responses: torch.Tensor,
        pad_token_id: int,
        context_length: int,
        reward_weights: torch.Tensor = None,
        processing_class: PreTrainedTokenizerBase = None,
        inputs: Dict[str, str] = None,
        stop_token_id: int = None,
        device: Any = None
):
    """
    Computes the reward logits and the rewards for a given model and query responses.

    Args:
        model (`torch.nn.Module`):
            The model used to compute the reward logits.
        query_responses (`torch.Tensor`):
            The tensor containing the query responses.
        pad_token_id (`int`):
            The token ID representing the pad token.
        context_length (`int`):
            The length of the context in the query responses.

    Returns:
        tuple:
            - `reward_logits` (`torch.Tensor`):
                The logits for the reward model.
            - `final_rewards` (`torch.Tensor`):
                The final rewards for each query response.
            - `sequence_lengths` (`torch.Tensor`):
                The lengths of the sequences in the query responses.
    """

    if isinstance(model, List):
        rewards_per_func = torch.zeros(len(query_responses), len(model), device=device)
        responses = query_responses[:, context_length:]

        postprocessed_response = truncate_response(stop_token_id, processing_class.pad_token_id, responses) if stop_token_id is not None else responses
        completions = processing_class.batch_decode(postprocessed_response, skip_special_tokens=True)

        # print('inputs: ', inputs)
        # print(completions)
        completions = [[{"content": completion}] for completion in completions]
        for i in range(len(model)):
            keys = [key for key in inputs if key not in ["prompt", "completion"]]
            reward_kwargs = {key: inputs[key] for key in keys}
            prompts = inputs['prompt']
            # print('reward_kwargs: ', reward_kwargs)
            output_reward_func = model[i](prompts=prompts, completions=completions, **reward_kwargs)
            # print('output_reward_func: ', output_reward_func)
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        rewards = (rewards_per_func * reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        return (None, rewards, rewards_per_func)

    attention_mask = query_responses != pad_token_id
    position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
    # lm_backbone = getattr(model, model.base_model_prefix)
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    print(f"query_responses: {query_responses.shape}, attention_mask: {attention_mask.shape}, position_ids: {position_ids.shape}")
    # output = lm_backbone(
    #     input_ids=input_ids,
    #     attention_mask=attention_mask,
    #     position_ids=position_ids,
    #     return_dict=True,
    #     output_hidden_states=True,
    #     use_cache=False,  # otherwise mistral-based RM would error out
    # )
    # reward_logits = model.score(output.hidden_states[-1])
    reward_logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
        use_cache=False,  # otherwise mistral-based RM would error out
    ).logits
    sequence_lengths = first_true_indices(query_responses[:, context_length:] == pad_token_id) - 1 + context_length
    # https://github.com/huggingface/transformers/blob/dc68a39c8111217683bf49a4912d0c9018bab33d/src/transformers/models/gpt2/modeling_gpt2.py#L1454
    return (
        reward_logits,
        reward_logits[
            torch.arange(reward_logits.size(0), device=device),
            sequence_lengths,
        ].squeeze(-1),
        None,
    )


def get_value(
        model,
        query_responses: torch.Tensor,
        pad_token_id: int,
        context_length: int,
        reward_weights: torch.Tensor = None,
        processing_class: PreTrainedTokenizerBase = None,
        inputs: Dict[str, str] = None,
        stop_token_id: int = None,
        device: Any = None
):
    attention_mask = query_responses != pad_token_id
    position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
    # lm_backbone = getattr(model, model.base_model_prefix)
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)

    if model.value_model is None:
        # value and policy model shape llm backbone
        lm_backbone = getattr(model.policy, model.policy.base_model_prefix)
        score_layer = getattr(model, "score")
    else:
        lm_backbone = getattr(model.value_model, model.value_model.base_model_prefix)
        score_layer = getattr(model.value_model, "score")

    output = lm_backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
        use_cache=False,  # otherwise mistral-based RM would error out
    )
    reward_logits = score_layer(output.hidden_states[-1])

    sequence_lengths = first_true_indices(query_responses[:, context_length:] == pad_token_id) - 1 + context_length
    # https://github.com/huggingface/transformers/blob/dc68a39c8111217683bf49a4912d0c9018bab33d/src/transformers/models/gpt2/modeling_gpt2.py#L1454
    return (
        reward_logits,
        reward_logits[
            torch.arange(reward_logits.size(0), device=device),
            sequence_lengths,
        ].squeeze(-1),
        None,
    )


@dataclass
class DataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.0 (Volta).
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            [{"input_ids": feat["input_ids"]} for feat in features],
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        for key in features[0].keys():
            if key not in ['input_ids']:
                batch[key] = [feat[key] for feat in features]

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch


# taken from https://github.com/OpenLMLab/MOSS-RLHF/blob/40b91eb2f2b71b16919addede0341d2bef70825d/ppo/ppo_trainer.py#L29
# we did this we can do a single `model = accelerator.prepare(model)`
class PolicyAndValueWrapper(nn.Module):
    def __init__(self, policy, value_model) -> None:
        super().__init__()
        self.policy = policy
        self.value_model = value_model

        if value_model is None:
            self.score = nn.Linear(policy.config.hidden_size, 1, bias=False)
        # self.critic_backbone = getattr(value_model, value_model.base_model_prefix)

    def forward(self, **kwargs):
        # print(f"self.value_model.base_model_prefix: {self.value_model.base_model_prefix}")
        # print(getattr(self.value_model, self.value_model.base_model_prefix))
        # print(self.value_model)
        # output = getattr(self.value_model, self.value_model.base_model_prefix)(**kwargs)

        policy_output = self.policy(**kwargs)

        if self.value_model is not None:
            output = self.value_model(**kwargs)
            logits = self.value_model.score(output.hidden_states[-1])
        else:
            logits = self.score(policy_output.hidden_states[-1])

        # value_output = self.value_model(**kwargs).logits
        return policy_output, logits


class PPOTrainer(Trainer):
    _tag_names = ["trl", "ppo"]

    def __init__(
        self,
        args: PPOConfig,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ],
        model: nn.Module,
        ref_model: Optional[nn.Module],
        reward_model: Optional[Union[nn.Module, List]],
        train_dataset: Dataset,
        value_model: Optional[nn.Module] = None,
        data_collator: Optional[DataCollatorWithPadding] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        # less commonly used
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        callbacks: Optional[list[TrainerCallback]] = None,
        peft_config: Optional["PeftConfig"] = None,
    ) -> None:
        if ref_model is model:
            raise ValueError(
                "`model` and `ref_model` cannot be the same object. If you want `ref_model` to be the "
                "same as `model`, you must make a copy of it, or `None` if you use peft."
            )

        self.args = args
        self.processing_class = processing_class
        self.policy_model = model
        self.value_model = value_model
        # self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        self.reward_weights = torch.tensor(args.reward_weights) if isinstance(reward_model, List) else None

        # Define the collator if not provided
        if data_collator is None:
            data_collator = DataCollatorWithPadding(self.processing_class)

        print("self.policy_model.generation_config.eos_token_id: ", self.policy_model.generation_config.eos_token_id)
        # Handle stop token settings: update policy model's generation_config to use provided stop token
        if args.stop_token and args.stop_token_id:
            raise ValueError("You cannot set both `stop_token` and `stop_token_id`.")
        elif args.stop_token:
            if args.stop_token == "eos":
                self.policy_model.generation_config.eos_token_id = self.stop_token_id = processing_class.eos_token_id
            else:
                raise ValueError(
                    f"Unknown `stop_token` {args.stop_token}. Allowed values are: `'eos'` and `None` (no stop token)."
                )
        else:
            # self.policy_model.generation_config.eos_token_id = self.stop_token_id = args.stop_token_id  # None or int
            self.stop_token_id = self.policy_model.generation_config.eos_token_id# = self.stop_token_id = args.stop_token_id  # None or int
        print(f'self.stop_token_id: {self.stop_token_id}, processing_class.eos_token_id: {processing_class.eos_token_id}, processing_class.pad_token_id: {processing_class.pad_token_id}')
        # peft support
        if is_peft_available() and peft_config is not None:
            # if model is a peft model and we have a peft_confg, we merge and unload it first
            print('isinstance(self.policy_model, PeftModel): ', isinstance(self.policy_model, PeftModel))
            if isinstance(self.policy_model, PeftModel):
                self.policy_model = self.policy_model.merge_and_unload()

            # get peft model with the given config
            self.policy_model = get_peft_model(self.policy_model, peft_config)
            self.value_model = get_peft_model(self.value_model, peft_config) if self.value_model is not None else None
            if args.bf16 and getattr(self.policy_model, "is_loaded_in_4bit", False):
                peft_module_casting_to_bf16(self.policy_model)

        self.is_peft_model = is_peft_available() and isinstance(self.policy_model, PeftModel)
        self.model_adapter_name = args.model_adapter_name
        self.ref_adapter_name = args.ref_adapter_name
        print(f'self.model_adapter_name: {self.model_adapter_name}, self.ref_adapter_name: {self.ref_adapter_name}')

        if ref_model:
            self.ref_model = ref_model
        elif self.is_peft_model:
            self.ref_model = None
        else:
            self.ref_model = create_reference_model(self.policy_model)

        self.reward_model = reward_model
        self.train_dataset = train_dataset
        self.train_dataset_len = len(train_dataset)
        self.data_collator = data_collator
        self.eval_dataset = eval_dataset
        self.optimizer, self.lr_scheduler = optimizers
        self.optimizer_cls_and_kwargs = None  # needed for transformers >= 4.47

        #########
        # calculate various batch sizes
        #########
        if args.total_episodes is None:  # allow the users to define episodes in terms of epochs.
            args.total_episodes = int(args.num_train_epochs * self.train_dataset_len)

        self.accelerator = Accelerator(deepspeed_plugin=args.deepspeed_plugin)

        args.world_size = self.accelerator.num_processes
        args.local_batch_size = (
            args.per_device_train_batch_size * args.gradient_accumulation_steps * args.num_mini_batches
        )
        args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
        args.batch_size = int(args.local_batch_size * args.world_size)
        args.mini_batch_size = exact_div(
            args.batch_size, args.num_mini_batches, "`batch_size` must be a multiple of `num_mini_batches`"
        )
        args.local_mini_batch_size = exact_div(
            args.local_batch_size, args.num_mini_batches, "`local_batch_size` must be a multiple of `num_mini_batches`"
        )
        print(f'args.local_batch_size: {args.local_batch_size}, args.micro_batch_size: {args.micro_batch_size}, args.mini_batch_size: {args.mini_batch_size}, args.local_mini_batch_size: {args.local_mini_batch_size}')

        # `per_rank_rollout_batch_size` is our `args.local_batch_size`
        # `per_rank_minibatch_size` is our `args.local_mini_batch_size`
        args.num_total_batches = math.ceil(
            args.total_episodes / args.batch_size
        )  # we may train for more than `total_episodes`
        time_tensor = torch.tensor(int(time.time()), device=self.accelerator.device)
        time_int = broadcast(time_tensor, 0).item()  # avoid different timestamps across processes
        args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
        self.local_seed = args.seed + self.accelerator.process_index * 100003  # Prime
        if args.num_sample_generations > 0:
            self.sample_generations_freq = max(1, args.num_total_batches // args.num_sample_generations)
        self.local_dataloader_batch_size = args.local_batch_size

        #########
        # setup model, optimizer, and others
        #########
        for module in [self.policy_model, self.ref_model, self.value_model, self.reward_model]:
            if module is not None and isinstance(module, nn.Module):
                disable_dropout_in_model(module)
        self.model = PolicyAndValueWrapper(self.policy_model, self.value_model)
        self.create_optimizer_and_scheduler(
            num_training_steps=args.num_total_batches
        )  # note that we are calling `self.lr_scheduler.step()` manually only at the batch level
        # self.device = "cuda"
        # self.model.to(self.device)
        print(self.model)

        train_params = 0
        total_params = 0
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                train_params += p.numel()
            total_params += p.numel()
        print(f'Train params = {train_params}, total params = {total_params}, train ratio = {train_params / total_params}')

        #########
        ### trainer specifics
        #########
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        self.callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            self.callbacks, self.model, self.processing_class, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)
        self.control = TrainerControl()
        self.state = OnlineTrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ],
        )
        self.current_flos = 0
        self.hp_search_backend = None
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        # Create distant repo and output directory if needed
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        #########
        ### setup dataloader
        #########
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.local_dataloader_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            drop_last=True,  # needed; otherwise the last batch will be of ragged shape
        )
        # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
        # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
        torch.manual_seed(args.seed)
        self.model, self.optimizer, self.dataloader = self.accelerator.prepare(self.model, self.optimizer, self.dataloader)

        torch.manual_seed(self.local_seed)  # reset the local seed again

        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=True,
        )  # no need to shuffle eval dataset
        self.eval_dataloader = self.accelerator.prepare(self.eval_dataloader)

        print("self.is_deepspeed_enabled: ", self.is_deepspeed_enabled)
        if self.is_deepspeed_enabled:
            if isinstance(self.reward_model, nn.Module):
                self.reward_model = prepare_deepspeed(
                    self.reward_model, args.per_device_train_batch_size, args.fp16, args.bf16
                )

            if self.ref_model is None:
                if not self.is_peft_model:
                    raise ValueError("No reference model and model is not a Peft model.")
            else:
                self.ref_model = prepare_deepspeed(
                    self.ref_model, args.per_device_train_batch_size, args.fp16, args.bf16
                )
        else:
            if self.ref_model is None:
                if not self.is_peft_model:
                    raise ValueError("No reference model and model is not a Peft model.")
            else:
                self.ref_model = self.ref_model.to(self.accelerator.device)
            if isinstance(self.reward_model, nn.Module):
                self.reward_model = self.reward_model.to(self.accelerator.device)

        self.use_vllm = args.use_vllm

        self.init_generation_config()

    def init_generation_config(self):
        if self.use_vllm:
            if self.accelerator.is_main_process:
                vllm_device = self.args.vllm_device
                if vllm_device == "auto":
                    if torch.cuda.device_count() == 1:
                        vllm_device = "cuda:0"  # particular case when training with onyl 1 GPU: share it
                    else:
                        vllm_device = f"cuda:{self.accelerator.num_processes}"  # take the next GPU idx
                # Check that the requested device is available
                if vllm_device.split(":")[0] == "cuda" and int(vllm_device.split(":")[1]) >= torch.cuda.device_count():
                    raise ValueError(
                        f"The requested device for vllm ({vllm_device}) is not available. You are likely using vLLM "
                        "without restricting the number of GPUs for training. Set the `--num_processes` argument to a "
                        "value lower than the number of GPUs available on your machine—typically, reducing it by one "
                        f"is sufficient. In your case: `--num_processes {torch.cuda.device_count() - 1}`."
                    )
                # Check that the requested device is not also used for training
                if vllm_device in {f"cuda:{idx}" for idx in range(self.accelerator.num_processes)}:
                    warnings.warn(
                        f"The requested device {vllm_device} is also being used for training. For higher throughput "
                        "and to avoid out-of-memory errors, it is recommended to use a dedicated device for vLLM. "
                        "If this is intentional, you may ignore this warning but should adjust "
                        "`vllm_gpu_memory_utilization` accordingly."
                    )
                # vLLM is not compatible with accelerate. So we need to patch it to make sure we can (1) place the vLLM
                # model on the desired device (world_size_patch) and (2) avoid a test that is not designed for our
                # setting (profiling_patch).
                world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
                profiling_patch = patch(
                    "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None
                )
                with world_size_patch, profiling_patch:
                    self.llm = LLM(
                        model=self.policy_model.name_or_path,
                        device=vllm_device,
                        gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
                        dtype=self.args.vllm_dtype,
                        # Automatic Prefix Caching caches the KV cache of existing queries, so that a new query can
                        # directly reuse the KV cache if it shares the same prefix with one of the existing queries.
                        # This is particularly useful here because we generate completions from the same prompts.
                        enable_prefix_caching=False,
                        max_model_len=self.args.vllm_max_model_len,
                    )
                self.sampling_params = SamplingParams(
                    temperature=self.args.temperature,
                    max_tokens=self.args.response_length,
                )

            self._last_loaded_step = 0  # tag to avoid useless loading during grad accumulation

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()
        else:
            self.generation_config = GenerationConfig(
                max_new_tokens=self.args.response_length,
                do_sample=True,
                temperature=self.args.temperature + 1e-7,
                pad_token_id=self.processing_class.pad_token_id,
            )

    def get_train_dataloader(self) -> DataLoader:
        return self.dataloader

    def get_eval_dataloader(self) -> DataLoader:
        return self.eval_dataloader

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        backup_model = self.model
        self.model = self.model.policy  # save only the policy

        if self.is_deepspeed_enabled:
            backup_deepspeed = self.deepspeed
            self.deepspeed = self.model

        super().save_model(output_dir, _internal_call)

        self.model = backup_model

        if self.is_deepspeed_enabled:
            self.deepspeed = backup_deepspeed

    def _move_model_to_vllm(self):
        with unwrap_model_for_generation(
            self.model.policy, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:
            # print('is_peft_model(unwrapped_model): ', is_peft_model(unwrapped_model))
            if is_peft_model(unwrapped_model):
                unwrapped_model.merge_adapter()
                state_dict = unwrapped_model.state_dict()
                # Remove base_model and base_layer prefixes
                state_dict = {
                    k.removeprefix("base_model.model.").replace(".base_layer", ""): v for k, v in state_dict.items()
                }
                # Remove values with adapter prefix (example: "_lora")
                state_dict = {k: v for k, v in state_dict.items() if unwrapped_model.prefix not in k}
                # When module to save, remove its prefix and discard the original module
                state_dict = {
                    k.replace("modules_to_save.default.", ""): v
                    for k, v in state_dict.items()
                    if "original_module" not in k
                }
            else:
                state_dict = unwrapped_model.state_dict()
            if self.accelerator.is_main_process:
                llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                # print(llm_model)
                # print(state_dict)
                llm_model.load_weights(state_dict.items())
            # Unmerge the adapter to restore the model to its original state.
            # This must be done after loading weights to ensure they correspond to the merged state.
            if is_peft_model(unwrapped_model):
                unwrapped_model.unmerge_adapter()

    def generate_samples(self, queries, prompts_text, processing_class):
        if self.args.use_vllm:
            # First, have main process load weights if needed
            # if self.state.global_step != self._last_loaded_step:
            self._move_model_to_vllm()
            # self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            all_prompts_text = gather_object(prompts_text)
            # print("all_prompts_text: ", len(all_prompts_text))
            if self.accelerator.is_main_process:
                outputs = self.llm.generate(all_prompts_text, sampling_params=self.sampling_params, use_tqdm=False)
                completion_ids = [torch.LongTensor(out.token_ids) for completions in outputs for out in completions.outputs]
                # print('outputs[0]: ', outputs[0])
            else:
                completion_ids = [None] * len(all_prompts_text)

            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts_text),
                (self.accelerator.process_index + 1) * len(prompts_text),
            )
            completion_ids = completion_ids[process_slice]

            completion_ids = pad_sequence(completion_ids, padding_value=processing_class.pad_token_id).permute(1, 0).to(self.accelerator.device)
            # Broadcast the completions from the main process to all processes, ensuring each process receives its
            # print("completion_ids[0].shape, queries.shape: ", len(completion_ids), len(completion_ids[0]))
            query_responses = torch.cat([queries, completion_ids], dim=1)
            attention_mask = torch.where(query_responses == processing_class.pad_token_id, 0, 1)
            # 获取logits，pi_old模型的logits
            logits_to_keep = completion_ids.size(1)
            with torch.inference_mode():
                logits = self.model.policy(input_ids=query_responses, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
                # print(f'logits: {logits.shape}')
                logits = logits[:, :-1 ]

            return query_responses, logits

        else:
            # Regular generation path
            with unwrap_model_for_generation(
                    self.model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
            ) as unwrapped_model:
            # print(self.model.device, queries.device)
                query_responses, logitss = batch_generation(
                    unwrapped_model.policy,
                    # self.model,
                    queries,
                    self.args.local_rollout_forward_batch_size,
                    processing_class.pad_token_id,
                    self.generation_config,
                )

            return query_responses, logitss

    def prepare_inputs(self, model, ref_policy, reward_model, processing_class, args, queries, query_responses, logitss, data, device):
        context_length = queries.shape[1]
        responses = []
        postprocessed_responses = []
        logprobs = []
        ref_logprobs = []
        scores = []
        scores_per_func = []
        sequence_lengths = []
        values = []
        for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
            query = queries[i: i + args.local_rollout_forward_batch_size]
            query_response = query_responses[i: i + args.local_rollout_forward_batch_size]
            response = query_response[:, context_length:]
            logits = logitss[i: i + args.local_rollout_forward_batch_size]
            logprob = selective_log_softmax(logits, response)
            # print(f'query: {query.shape}, query_response: {query_response.shape}, response: {response.shape}, logits: {logits.shape}, logprob: {logprob.shape}')
            del logits
            torch.cuda.empty_cache()

            if ref_policy is None:
                unwrap_model = self.accelerator.unwrap_model(model.policy)
                with unwrap_model.disable_adapter():
                    ref_output = forward(unwrap_model, query_response, processing_class.pad_token_id)
            else:
                ref_output = forward(ref_policy, query_response, processing_class.pad_token_id)
            ref_logits = ref_output.logits[:, context_length - 1: -1]
            ref_logits /= args.temperature + 1e-7
            ref_logprob = selective_log_softmax(ref_logits, response)
            del ref_output, ref_logits
            torch.cuda.empty_cache()

            # Response Processing 1. truncate response after the first occurrence of `stop_token_id`
            postprocessed_response = response
            if self.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                postprocessed_response = truncate_response(
                    self.stop_token_id, processing_class.pad_token_id, response
                )

            # Response Processing 2. run reward model on the truncated responses
            postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
            sequence_length = first_true_indices(postprocessed_response == processing_class.pad_token_id) - 1

            unwrapped_model = self.accelerator.unwrap_model(model)
            full_value, _, _ = get_value(
                unwrapped_model, query_response, processing_class.pad_token_id, context_length, device=device
            )
            value = full_value[:, context_length - 1: -1].squeeze(-1)

            _, score, score_per_func = get_reward(
                reward_model, postprocessed_query_response, processing_class.pad_token_id, context_length,
                self.reward_weights, processing_class, data, self.stop_token_id, device=device
            )

            responses.append(response)
            postprocessed_responses.append(postprocessed_response)
            logprobs.append(logprob)
            ref_logprobs.append(ref_logprob)
            sequence_lengths.append(sequence_length)
            scores.append(score)
            scores_per_func.append(score_per_func)
            values.append(value)
        responses = torch.cat(responses, 0)
        postprocessed_responses = torch.cat(postprocessed_responses, 0)
        logprobs = torch.cat(logprobs, 0)
        ref_logprobs = torch.cat(ref_logprobs, 0)
        sequence_lengths = torch.cat(sequence_lengths, 0)
        scores = torch.cat(scores, 0)
        scores_per_func = torch.cat(scores_per_func, 0)
        values = torch.cat(values, 0)

        del (logprob, ref_logprob, full_value, value, score)
        torch.cuda.empty_cache()
        gc.collect()

        # print(f'GPU: {self.accelerator.process_index} values: {values.shape}, scores: {scores.shape}, logprobs: {logprobs.shape}, ref_logprobs: {ref_logprobs.shape}, scores_per_func: {scores_per_func.shape}')

        # Response Processing 3. Filter completion. Ensure that the sample contains stop_token_id
        # Completions not passing that filter will receive a lower score.
        contain_eos_token = torch.any(postprocessed_responses == self.processing_class.eos_token_id, dim=-1)
        if self.args.missing_eos_penalty is not None:
            scores[~contain_eos_token] -= self.args.missing_eos_penalty
        # accelerator.print(f"{scores=}, {(contain_eos_token.sum() / len(contain_eos_token))=}")

        # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
        response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
        padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
        logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
        ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)
        sequence_lengths_p1 = sequence_lengths + 1
        padding_mask_p1 = response_idxs > (sequence_lengths_p1.unsqueeze(1))
        values = torch.masked_fill(values, padding_mask_p1, 0)

        # 4. compute rewards
        kl = logprobs - ref_logprobs
        non_score_reward = -args.kl_coef * kl
        rewards = non_score_reward.clone()
        actual_start = torch.arange(rewards.size(0), device=rewards.device)
        actual_end = torch.where(sequence_lengths_p1 < rewards.size(1), sequence_lengths_p1, sequence_lengths)
        rewards[[actual_start, actual_end]] += scores

        # 5. whiten rewards
        if args.whiten_rewards:
            rewards = masked_whiten(rewards, mask=~padding_mask_p1, shift_mean=False)
            rewards = torch.masked_fill(rewards, padding_mask_p1, 0)

        # 6. compute advantages and returns
        lastgaelam = 0
        advantages_reversed = []
        gen_length = responses.shape[1]
        for t in reversed(range(gen_length)):
            nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
            delta = rewards[:, t] + args.gamma * nextvalues - values[:, t]
            lastgaelam = delta + args.gamma * args.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        advantages = masked_whiten(advantages, ~padding_mask)
        advantages = torch.masked_fill(advantages, padding_mask, 0)
        torch.cuda.empty_cache()

        return advantages, returns, responses, logprobs, values, padding_mask, padding_mask_p1, kl, non_score_reward, scores, sequence_lengths_p1, scores_per_func

    def compute_train_loss(self, model, args, processing_class, mb_query_responses, context_length, mb_responses, padding_mask, padding_mask_p1, mb_values, mb_return, mb_logprobs, mb_advantage, micro_batch_inds):
        output, vpred_temp = forward(model, mb_query_responses, processing_class.pad_token_id)
        # output = forward(model, mb_query_responses, processing_class.pad_token_id)
        # vpred_temp = output.scores

        logits = output.logits[:, context_length - 1: -1]
        logits /= args.temperature + 1e-7
        new_logprobs = selective_log_softmax(logits, mb_responses)
        new_logprobs = torch.masked_fill(
            new_logprobs, padding_mask[micro_batch_inds], INVALID_LOGPROB
        )
        vpred = vpred_temp[:, context_length - 1: -1].squeeze(-1)
        vpred = torch.masked_fill(vpred, padding_mask_p1[micro_batch_inds], 0)
        vpredclipped = torch.clamp(
            vpred,
            mb_values - args.cliprange_value,
            mb_values + args.cliprange_value,
        )
        vf_losses1 = torch.square(vpred - mb_return)
        vf_losses2 = torch.square(vpredclipped - mb_return)
        vf_loss_max = torch.max(vf_losses1, vf_losses2)
        vf_loss = 0.5 * masked_mean(vf_loss_max, ~padding_mask_p1[micro_batch_inds])
        vf_clipfrac = masked_mean(
            (vf_losses2 > vf_losses1).float(), ~padding_mask_p1[micro_batch_inds]
        )

        logprobs_diff = new_logprobs - mb_logprobs
        ratio = torch.exp(logprobs_diff)
        pg_losses = -mb_advantage * ratio
        pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
        pg_loss_max = torch.max(pg_losses, pg_losses2)
        pg_loss = masked_mean(pg_loss_max, ~padding_mask[micro_batch_inds])
        loss = pg_loss + args.vf_coef * vf_loss

        return loss, pg_loss, pg_losses2, pg_losses, vf_loss, logits, logprobs_diff, vf_clipfrac, new_logprobs, ratio

    def train(self):
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model
        ref_policy = self.ref_model
        reward_model = self.reward_model
        processing_class = self.processing_class
        dataloader = self.dataloader
        device = self.accelerator.device

        def repeat_generator():
            while True:
                yield from dataloader

        iter_dataloader = iter(repeat_generator())
        generation_config = GenerationConfig(
            max_new_tokens=args.response_length,
            temperature=(args.temperature + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )

        accelerator.print("===training policy===")
        start_time = time.time()
        stats_shape = (args.num_ppo_epochs, args.num_mini_batches, args.gradient_accumulation_steps)
        approxkl_stats = torch.zeros(stats_shape, device=device)
        pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
        pg_loss_stats = torch.zeros(stats_shape, device=device)
        vf_loss_stats = torch.zeros(stats_shape, device=device)
        vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
        entropy_stats = torch.zeros(stats_shape, device=device)
        ratio_stats = torch.zeros(stats_shape, device=device)
        model.train()

        # trainer state initialization
        self.state.global_step = 0
        self.state.episode = 0
        self.state.max_steps = args.num_total_batches * args.num_mini_batches
        self.state.num_train_epochs = args.total_episodes / self.train_dataset_len
        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(self.state.max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(self.state.max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(self.state.max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model
            self.model_wrapped = self.model

        for update in range(1, args.num_total_batches + 1):
            self.state.episode += 1 * args.batch_size
            data = next(iter_dataloader)
            # Generate samples
            with torch.no_grad():
                queries = data["input_ids"].to(device)
                context_length = queries.shape[1]
                local_batch_size = len(queries) // self.args.gradient_accumulation_steps
                # advantages_list, returns_list, responses_list, logprobs_list, values_list, padding_mask_list, \
                #     padding_mask_p1_list, kl_list, non_score_reward_list, scores_list, sequence_lengths_p1_list, scores_per_func_list, query_responses_list = [[] for _ in range(13)]
                # for idx in range(0, self.args.gradient_accumulation_steps):
                #     start_idx, end_idx = idx*local_batch_size, (idx+1)*local_batch_size
                #     local_data = {k:v[start_idx: end_idx] for k, v in data.items()}
                #     local_queries, local_prompt = queries[start_idx: end_idx], data['prompt'][start_idx: end_idx]
                #
                #     query_responses, logitss = self.generate_samples(local_queries, local_prompt, processing_class)
                #     # print(f"query_responses: {query_responses.shape}, logitss: {logitss.shape}, queries: {queries.shape}")
                #
                #     advantages, returns, responses, logprobs, values, padding_mask, padding_mask_p1, kl, non_score_reward, scores, sequence_lengths_p1, scores_per_func = self.prepare_inputs(
                #         model, ref_policy, reward_model, processing_class,
                #         args, local_queries, query_responses, logitss, local_data, device)
                #
                #     advantages_list.append(advantages)
                #     returns_list.append(returns)
                #     responses_list.append(responses)
                #     logprobs_list.append(logprobs)
                #     values_list.append(values)
                #     padding_mask_list.append(padding_mask)
                #     padding_mask_p1_list.append(padding_mask_p1)
                #     kl_list.append(kl)
                #     non_score_reward_list.append(non_score_reward)
                #     scores_list.append(scores)
                #     sequence_lengths_p1_list.append(sequence_lengths_p1)
                #     scores_per_func_list.append(scores_per_func)
                #     query_responses_list.append(query_responses)
                #
                # advantages = padding_sequence(advantages_list, pad_val=0)
                # returns = padding_sequence(returns_list, pad_val=0)
                # responses = padding_sequence(responses_list, pad_val=processing_class.pad_token_id)
                # query_responses = padding_sequence(query_responses_list, pad_val=processing_class.pad_token_id)
                # logprobs = padding_sequence(logprobs_list, pad_val=INVALID_LOGPROB)
                # values = padding_sequence(values_list, pad_val=0)
                # padding_mask = padding_sequence(padding_mask_list, pad_val=1).to(torch.bool)
                # padding_mask_p1 = padding_sequence(padding_mask_p1_list, pad_val=1).to(torch.bool)
                # kl = padding_sequence(kl_list, pad_val=0)
                # non_score_reward = padding_sequence(non_score_reward_list, pad_val=0)
                # scores = torch.cat(scores_list, dim=0)
                # sequence_lengths_p1 = torch.cat(sequence_lengths_p1_list, dim=0)
                # scores_per_func = torch.cat(scores_per_func_list, dim=0)
                #
                # del (advantages_list, returns_list, responses_list, logprobs_list, values_list, padding_mask_list,
                #     padding_mask_p1_list, kl_list, non_score_reward_list, scores_list, sequence_lengths_p1_list, scores_per_func_list)
                # torch.cuda.empty_cache()
                query_responses, logitss = self.generate_samples(queries, data['prompt'], processing_class)
                print(f'GPU: {self.accelerator.process_index} queries: {queries.shape}, query_responses: {query_responses.shape}, logitss: {logitss.shape}')

                advantages, returns, responses, logprobs, values, padding_mask, padding_mask_p1, kl, non_score_reward, scores, sequence_lengths_p1, scores_per_func = self.prepare_inputs(model, ref_policy, reward_model, processing_class,
                                                                                                                      args, queries, query_responses, logitss, data, device)

                print(f"advantages: {advantages.shape}, returns: {returns.shape}, responses: {responses.shape}, logprobs: {logprobs.shape}, "
                      f"values: {values.shape}, padding_mask: {padding_mask.shape}, padding_mask_p1: {padding_mask_p1.shape}, kl: {kl.shape}, "
                      f"non_score_reward: {non_score_reward.shape}, scores: {scores.shape}, sequence_lengths_p1: {sequence_lengths_p1.shape}, "
                      f"scores_per_func: {scores_per_func.shape}, query_responses: {query_responses.shape}, queries: {queries.shape}")

                print(f'torch.sum(query_responses >= len(processing_class)): {torch.sum(query_responses >= len(processing_class))}')

            # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
            for ppo_epoch_idx in range(args.num_ppo_epochs):
                b_inds = np.random.permutation(args.local_batch_size)
                minibatch_idx = 0
                # loop num_mini_batches
                # args.local_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps * args.num_mini_batches
                for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size):
                    mini_batch_end = mini_batch_start + args.local_mini_batch_size
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                    gradient_accumulation_idx = 0

                    self.accelerator.gradient_state._set_sync_gradients(True)
                    # gradient accumulation
                    # args.local_mini_batch_size = args.per_device_train_batch_size * gradient_accumulation_steps
                    for micro_batch_start in range(0, args.local_mini_batch_size, args.per_device_train_batch_size):
                        # with accelerator.accumulate(model):
                        micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                        micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]
                        mb_advantage = advantages[micro_batch_inds]
                        mb_responses = responses[micro_batch_inds]
                        mb_query_responses = query_responses[micro_batch_inds]
                        mb_logprobs = logprobs[micro_batch_inds]
                        mb_return = returns[micro_batch_inds]
                        mb_values = values[micro_batch_inds]

                        loss, pg_loss, pg_losses2, pg_losses, vf_loss, logits, logprobs_diff, vf_clipfrac, new_logprobs, ratio = self.compute_train_loss(model, args, processing_class, mb_query_responses, context_length, mb_responses, padding_mask, padding_mask_p1, mb_values, mb_return, mb_logprobs, mb_advantage, micro_batch_inds)

                        kwargs = {}
                        if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                            kwargs["scale_wrt_gas"] = False
                        accelerator.backward(loss)

                        with torch.no_grad():
                            pg_clipfrac = masked_mean(
                                (pg_losses2 > pg_losses).float(), ~padding_mask[micro_batch_inds]
                            )
                            prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                            entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                            approxkl = 0.5 * (logprobs_diff**2).mean()
                            approxkl_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
                            pg_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                pg_clipfrac
                            )
                            pg_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_loss
                            vf_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_loss
                            vf_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                vf_clipfrac
                            )
                            entropy_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy.mean()
                            ratio_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = ratio.mean()
                        gradient_accumulation_idx += 1

                        # del everything and empty cache
                        # fmt: off
                        del (
                            logits, new_logprobs, vf_loss, vf_clipfrac, logprobs_diff, ratio, pg_losses, pg_losses2,
                            pg_loss, loss,  mb_return,
                            mb_advantage, mb_values, mb_responses, mb_query_responses, mb_logprobs,
                        )
                        # fmt: on
                        torch.cuda.empty_cache()

                    # Since we perform prefetching, we need to manually set sync_gradients to True
                    self.accelerator.gradient_state._set_sync_gradients(True)

                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        print('Grad clip !')
                        _grad_norm = self.accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    optimizer.step()
                    model.zero_grad()
                    self.state.global_step += 1
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                    minibatch_idx += 1

            print(f'GPU {self.accelerator.process_index}, step {update}, computing loss. self.control.should_save = {self.control.should_save}')

            with torch.no_grad():
                mean_kl = kl.sum(1).mean()
                mean_entropy = (-logprobs).sum(1).mean()
                mean_non_score_reward = non_score_reward.sum(1).mean()
                rlhf_reward = mean_non_score_reward + scores.mean()
                eps = int(self.state.episode / (time.time() - start_time))
                metrics = {}

                postprocessed_response = responses
                if self.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                    print('++++++++truncate_response++++++++')
                    postprocessed_response = truncate_response(
                        self.stop_token_id, processing_class.pad_token_id, responses
                    )

                scores_per_func = self.accelerator.gather_for_metrics(scores_per_func).mean(0)
                for func_name, func_reward in zip(self.args.reward_funcs, scores_per_func):
                    metrics[f"rewards/{func_name}"] = func_reward.item()
                    # print(f"rewards/{func_name}: {func_reward.item()}")
                # print('self.accelerator.gather_for_metrics(sequence_lengths_p1): ', self.accelerator.gather_for_metrics(sequence_lengths_p1))
                metrics["objective/scores"] = self.accelerator.gather_for_metrics(scores.mean()).mean().item()
                metrics["completion/length"] = self.accelerator.gather_for_metrics(sequence_lengths_p1).to(torch.float).mean().item()

                metrics["objective/kl"] = self.accelerator.gather_for_metrics(mean_kl).mean().item()
                metrics["objective/entropy"] = self.accelerator.gather_for_metrics(mean_entropy).mean().item()
                metrics["objective/non_score_reward"] = (
                    self.accelerator.gather_for_metrics(mean_non_score_reward).mean().item()
                )
                metrics["objective/rlhf_reward"] = self.accelerator.gather_for_metrics(rlhf_reward).mean().item()
                metrics["policy/approxkl_avg"] = self.accelerator.gather_for_metrics(approxkl_stats).mean().item()
                metrics["policy/clipfrac_avg"] = self.accelerator.gather_for_metrics(pg_clipfrac_stats).mean().item()
                metrics["loss/policy_avg"] = self.accelerator.gather_for_metrics(pg_loss_stats).mean().item()
                metrics["loss/value_avg"] = self.accelerator.gather_for_metrics(vf_loss_stats).mean().item()
                metrics["val/clipfrac_avg"] = self.accelerator.gather_for_metrics(vf_clipfrac_stats).mean().item()
                # metrics["policy/entropy_avg"] = self.accelerator.gather_for_metrics(entropy_stats).mean().item()
                metrics["val/ratio"] = self.accelerator.gather_for_metrics(ratio_stats).mean().item()
                # metrics["val/ratio_var"] = self.accelerator.gather_for_metrics(ratio_stats).var().item()
                metrics["val/num_eos_tokens"] = (postprocessed_response == processing_class.eos_token_id).sum().item()
                metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
                metrics["episode"] = self.state.episode
                self.state.epoch = self.state.episode / self.train_dataset_len  # used by self.log
                self.state.global_step += 1
                self.log(metrics)

            # wandb
            if update % 5 == 0 and self.accelerator.is_main_process and ("wandb" in args.report_to):
                table = defaultdict(list)
                postprocessed_response = responses
                if self.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                    postprocessed_response = truncate_response(
                        self.stop_token_id, processing_class.pad_token_id, responses
                    )
                table["query"].extend(
                    gather_object(processing_class.batch_decode(queries, skip_special_tokens=True))
                )
                table["model response"].extend(
                    gather_object(processing_class.batch_decode(postprocessed_response, skip_special_tokens=True))
                )
                table["step"].extend(
                    [update] * len(table["query"])
                )
                table["scores"].extend(
                    gather_object(scores.detach().clone().cpu().numpy().tolist())#.item()
                )
                df = pd.DataFrame(table)

                if wandb.run is not None:
                    # print_rich_table(df.iloc[0: 0 + 5])
                    wandb.log({"completions": wandb.Table(dataframe=df)})

            print(f'GPU: {self.accelerator.process_index}, is_fsdp_enabled: {self.is_fsdp_enabled}, args.save_only_model: {self.args.save_only_model}, self.args.should_save: {self.args.should_save}')

            self.lr_scheduler.step()
            self.control = self.callback_handler.on_step_end(args, self.state, self.control)
            print(f'GPU: {self.accelerator.process_index}, step: {update}, self.control.should_save = {self.control.should_save}')
            # dist.barrier()
            if self.control.should_save:
                print(f'GPU: {self.accelerator.process_index}, step: {update}, saving checkpoint')
                self._save_checkpoint(model, trial=None)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            # dist.barrier()
            del kl, mean_kl, mean_non_score_reward, scores, metrics, non_score_reward
            torch.cuda.empty_cache()
            gc.collect()

            if args.num_sample_generations > 0 and (update - 1) % self.sample_generations_freq == 0:
                # self.generate_completions(sampling=True)
                torch.cuda.empty_cache()

        # HF trainer specifics
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(model, trial=None, metrics=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
