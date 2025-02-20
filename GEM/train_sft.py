import os 
import sys
import json 
import argparse
import logging
from typing import Any, Optional
from dataclasses import dataclass, field

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    DataCollatorForSeq2Seq,
    set_seed,
)

import datasets 
from datasets import load_dataset
import torch
import torch.distributed as dist
import deepspeed
from functools import partial

from sft_trainer import SFTTrainer
from utils import save_code

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    adam_beta2: float = 0.95
    loss: str = field(
        default="ce", metadata={"help": "Loss name", "choices": ["gem", "ce"]}
    )
    gem_beta: float = field(default=0.7, metadata={"help": "Hyper-parameter in GEM."})
    gem_h: str = field(
        default="logsigmoid", metadata={"help": "Hyper-parameter in GEM.", "choices": ["logsigmoid", "linear"]}
    )
    max_seq_length: Optional[int] = field(default=4096)


@dataclass
class ModelArguments:
    model_name_or_path: str
    tokenizer_name_or_path: str
    attn_implementation: str = field(default="flash_attention_2")


@dataclass
class DataArguments:
    dataset_name: str
    dataset_train_split: str = "train"
    dataset_test_split: str = "test"
    max_train_samples: Optional[int] = field(default=None)


def encode_sft_example(example, tokenizer, max_seq_length, verbose=False):
    """
    This function encodes a single example into a format that can be used for sft training.
    Here, we assume each example has a 'messages' field. Each message in it is a dict with 'role' and 'content' fields.
    We use the `apply_chat_template` function from the tokenizer to tokenize the messages and prepare the input and label tensors.
    """
    messages = example["messages"]
    if len(messages) == 0:
        raise ValueError("messages field is empty.")
    if verbose:
        chat_messages = tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=max_seq_length,
            add_generation_prompt=False,
        )
        print(f"chat_messages:\n[{chat_messages}]")
    input_ids = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=True,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_seq_length,
        add_generation_prompt=False,
    )
    labels = input_ids.clone()
    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            # we calculate the start index of this non-assistant message
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer.apply_chat_template(
                    conversation=messages[
                        :message_idx
                    ],  # here marks the end of the previous messages
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length,
                    add_generation_prompt=False,
                ).shape[1]
            # next, we calculate the end index of this non-assistant message
            if (
                message_idx < len(messages) - 1
                and messages[message_idx + 1]["role"] == "assistant"
            ):
                # for intermediate messages that follow with an assistant message, we need to
                # set `add_generation_prompt=True` to avoid the assistant generation prefix being included in the loss
                # (e.g., `<|assistant|>`)
                message_end_idx = tokenizer.apply_chat_template(
                    conversation=messages[: message_idx + 1],
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length,
                    add_generation_prompt=True,
                ).shape[1]
            else:
                # for the last message or the message that doesn't follow with an assistant message,
                # we don't need to add the assistant generation prefix
                message_end_idx = tokenizer.apply_chat_template(
                    conversation=messages[: message_idx + 1],
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length,
                    add_generation_prompt=False,
                ).shape[1]
            # set the label to -100 for the non-assistant part
            labels[:, message_start_idx:message_end_idx] = -100
            if max_seq_length and message_end_idx >= max_seq_length:
                break
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.flatten().tolist(),
        "labels": labels.flatten().tolist(),
        "attention_mask": attention_mask.flatten().tolist(),
    }


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    global_rank = dist.get_rank()
    logger.warning(
        f"Process rank: {global_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    )
    logger.info(f"Training parameters {training_args}")
    if global_rank == 0:

        def remove_non_serializable(data):
            serializable_data = {}
            for key, value in data.items():
                try:
                    json.dumps({key: value})
                    serializable_data[key] = value
                except (TypeError, ValueError):
                    pass
            return serializable_data

        args_dict = remove_non_serializable(training_args.__dict__)
        args_dict.update(remove_non_serializable(model_args.__dict__))
        args_dict.update(remove_non_serializable(data_args.__dict__))
    
        json.dump(
            dict(sorted(args_dict.items(), key=lambda x: x[0])),
            open(os.path.join(training_args.output_dir, "args.json"), "w"),
            indent=2,
        )
        save_code(training_args.output_dir)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    ################
    # Model init kwargs & Tokenizer
    ################
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=model_args.attn_implementation,
    )

    tokenizer_name_or_path = model_args.tokenizer_name_or_path if model_args.tokenizer_name_or_path else model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    # gather deepspeed to get "real" embedding size
    embeddings = model.get_input_embeddings()
    with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
        embedding_size = embeddings.weight.shape[0]
    # resize does its own gather
    if len(tokenizer) > embedding_size:
        # pad to multiple for tensor cores.
        logging.warning(f"len(tokenizer) > embedding_size!!! we are resizing...")
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

    if tokenizer.pad_token is None:
        if "llama-3" in tokenizer.name_or_path.lower():
            tokenizer.pad_token_id = len(tokenizer) - 1
            tokenizer.pad_token = tokenizer.decode(tokenizer.pad_token_id)
        else:
            raise ValueError(f"pad_token is None and not supported for this model: {tokenizer.name_or_path}")
    
    ################
    # Dataset
    ################
    dataset = load_dataset(data_args.dataset_name)
    train_dataset = dataset[data_args.dataset_train_split]
    eval_dataset = dataset[data_args.dataset_test_split] if training_args.eval_strategy != "no" else None
    max_seq_length = training_args.max_seq_length

    train_dataset = train_dataset.map(
        partial(encode_sft_example, tokenizer=tokenizer, max_seq_length=max_seq_length),
        num_proc=8, 
        desc="Tokenizing dataset"
    )
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(
            partial(encode_sft_example, tokenizer=tokenizer, max_seq_length=max_seq_length),
            num_proc=8, 
            desc="Tokenizing dataset"
        )
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    # initalize a trainer
    # here we use a custom trainer that moves the model to CPU when saving the checkpoint in FSDP mode
    # we can switch to the default trainer after moving to deepspeed (let's don't change too much for now)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, padding="longest"
        ),
        preprocess_logits_for_metrics=None,
        compute_metrics=None,
    )

    # Training
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    if "llama-3" in model.config.name_or_path.lower() and isinstance(model.generation_config.eos_token_id, int):
        model.generation_config.eos_token_id = [128001, 128009]
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)


if __name__ == "__main__":
    main()
