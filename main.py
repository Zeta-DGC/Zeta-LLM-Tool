#!/usr/bin/env python3

# -- Zeta-LLM --------------------------------------------------- #
# main.py on Zeta-LLM                                             #
# Made by DiamondGotCat, Licensed under MIT License               #
# Copyright (c) 2025 DiamondGotCat                                #
# ---------------------------------------------- DiamondGotCat -- #

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import pandas as pd
from torch import nn
from torch.utils.data import Dataset

from datasets import load_dataset, Dataset as HFDataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    PretrainedConfig,
    PreTrainedModel,
    set_seed,
)

try:
    import bitsandbytes as bnb  # noqa: F401
    BNB_AVAILABLE = True
except Exception:
    BNB_AVAILABLE = False

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # type: ignore
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

try:
    from rich.logging import RichHandler  # noqa: F401
    RICH_AVAILABLE = True
except Exception:
    RICH_AVAILABLE = False

# ------------------------- Logging ------------------------- #

def setup_logging(verbosity: int = 1) -> None:
    level = logging.INFO if verbosity <= 1 else logging.DEBUG
    handlers: List[logging.Handler] = []
    if RICH_AVAILABLE:
        handlers.append(RichHandler(markup=True, rich_tracebacks=True))
    else:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(
        level=level,
        format="%(message)s" if RICH_AVAILABLE else "%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)


# ------------------------- Minimal Empty Model ------------------------- #

class EmptyModelConfig(PretrainedConfig):
    model_type = "zeta"
    def __init__(self, vocab_size=1000, hidden_size=128, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size


class EmptyForCausalLM(PreTrainedModel):
    config_class = EmptyModelConfig

    def __init__(self, config: EmptyModelConfig):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.embed(input_ids)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return {"loss": loss, "logits": logits}


# ------------------------- CLI Config ------------------------- #

@dataclass
class LoRAArgs:
    enabled: bool = False
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: Optional[str] = None  # comma-separated, e.g. "q_proj,k_proj,v_proj,o_proj"

@dataclass
class QuantArgs:
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    bnb_compute_dtype: str = "bfloat16"  # or "float16"

@dataclass
class TrainConfig:
    # Model/tokenizer
    base_model: Optional[str] = None        # HF repo path or local
    tokenizer_path: Optional[str] = None    # fallback path when base_model is None
    use_empty_model: bool = False

    # Data
    dataset_type: str = "azukif"  # "azukif" or "huggingface"
    dataset_path: Optional[str] = None      # for azukif json
    hf_dataset_id: Optional[str] = None     # e.g. "user/repo"
    hf_subset: Optional[str] = None
    hf_input_col: str = "input"
    hf_output_col: str = "output"
    max_length: int = 2048
    block_size: int = 1024                  # group_texts chunk len

    # Training
    output_dir: str = "./results"
    epochs: float = 3.0
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3
    evaluation_strategy: str = "no"  # "no" | "steps" | "epoch"
    eval_steps: int = 500
    do_eval_split: bool = False
    eval_ratio: float = 0.02

    # Precision / Perf
    fp16: bool = False
    bf16: bool = True
    tf32: bool = True
    gradient_checkpointing: bool = True
    flash_attn: bool = True
    torch_compile: bool = False

    # Reproducibility
    seed: int = 42
    deterministic: bool = False

    # Extras
    wandb: bool = False
    wandb_project: str = "zeta-llm"
    push_to_hub: bool = False

    # LoRA / Quant
    lora: LoRAArgs = LoRAArgs()
    quant: QuantArgs = QuantArgs()

    # ETA benchmark
    benchmark_steps: int = 10  # micro benchmark steps for ETA
    benchmark_warmup: int = 3

    # Misc
    resume_from_checkpoint: Optional[str] = None
    logging_dir: str = "./logs"


# ------------------------- Dataset utilities ------------------------- #

def format_conversation_azukif(json_path: str) -> List[str]:
    """
    AzukiF 1.0 JSON Format:
    [
      [ {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."} ],
      ...
    ]
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts: List[str] = []
    for conv in data:
        buff = []
        for msg in conv:
            role = str(msg.get("role", "")).strip().upper()
            content = str(msg.get("content", "")).strip()
            buff.append(f"<{role}>{content}</{role}>")
        texts.append("".join(buff))
    return texts


def format_conversation_hf(
    ds: HFDataset,
    input_col: str,
    output_col: str,
) -> List[str]:
    texts: List[str] = []
    for ex in ds:
        inp = str(ex[input_col])
        out = str(ex[output_col])
        texts.append(f"<USER>{inp}</USER><ASSISTANT>{out}</ASSISTANT>")
    return texts


def build_hf_dataset_from_texts(texts: List[str]) -> DatasetDict:
    ds = HFDataset.from_pandas(pd.DataFrame({"text": texts}), preserve_index=False)
    return DatasetDict({"train": ds})


def tokenize_and_chunk(
    ds: DatasetDict,
    tokenizer: AutoTokenizer,
    block_size: int,
    num_proc: Optional[int] = None,
) -> DatasetDict:
    def tokenize(examples):
        return tokenizer(examples["text"])

    tokenized = ds.map(tokenize, batched=True, remove_columns=ds["train"].column_names, num_proc=num_proc)
    def group_texts(examples):
        concatenated = {}
        for k in examples.keys():
            concatenated[k] = sum(examples[k], [])
        total_len = len(concatenated["input_ids"])
        if total_len >= block_size:
            total_len = (total_len // block_size) * block_size
        result = {}
        for k in concatenated.keys():
            result[k] = [concatenated[k][i:i+block_size] for i in range(0, total_len, block_size)]
        result["labels"] = result["input_ids"].copy()
        return result

    chunks = tokenized.map(group_texts, batched=True, num_proc=num_proc)
    return chunks


# ------------------------- GPU / ETA utilities ------------------------- #

@dataclass
class GPUInfo:
    count: int
    names: List[str]
    total_mem_gb: List[float]
    capability: List[Tuple[int, int]]

def get_gpu_info() -> GPUInfo:
    if not torch.cuda.is_available():
        return GPUInfo(0, [], [], [])
    n = torch.cuda.device_count()
    names, mems, caps = [], [], []
    for i in range(n):
        prop = torch.cuda.get_device_properties(i)
        names.append(prop.name)
        mems.append(prop.total_memory / (1024**3))
        caps.append(prop.major, )
        caps[-1] = (prop.major, prop.minor)
    return GPUInfo(n, names, mems, caps)


def measure_step_time(
    trainer: Trainer,
    steps: int,
    warmup: int = 3,
) -> float:
    model = trainer.model
    model.train()
    dl = trainer.get_train_dataloader()
    it = iter(dl)

    # warmup
    for _ in range(warmup):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)
        batch = {k: v.to(model.device) for k, v in batch.items()}
        loss = trainer.training_step(model, batch)
        if hasattr(trainer, "accelerator"):
            trainer.accelerator.backward(loss)
        else:
            loss.backward()
        trainer.optimizer.step()
        trainer.lr_scheduler.step()
        model.zero_grad(set_to_none=True)

    # measure
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    measured = 0
    for _ in range(steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)
        batch = {k: v.to(model.device) for k, v in batch.items()}
        loss = trainer.training_step(model, batch)
        if hasattr(trainer, "accelerator"):
            trainer.accelerator.backward(loss)
        else:
            loss.backward()
        trainer.optimizer.step()
        trainer.lr_scheduler.step()
        model.zero_grad(set_to_none=True)
        measured += 1
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t1 = time.time()

    return (t1 - t0) / max(1, measured)


def human_time(seconds: float) -> str:
    d = int(seconds // 86400)
    h = int((seconds % 86400) // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{d}d {h}h {m}m {s}s"


# ------------------------- Model / Tokenizer build ------------------------- #

def build_tokenizer(cfg: TrainConfig) -> AutoTokenizer:
    if cfg.base_model:
        tok = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)
    elif cfg.tokenizer_path:
        tok = AutoTokenizer.from_pretrained(cfg.tokenizer_path, use_fast=True)
    else:
        raise ValueError("Either --base_model or --tokenizer_path must be provided.")

    if tok.pad_token is None:
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "<|pad|>"})
    return tok


def build_model(cfg: TrainConfig, tokenizer: AutoTokenizer) -> PreTrainedModel:
    if cfg.use_empty_model:
        config = EmptyModelConfig(vocab_size=tokenizer.vocab_size)
        model = EmptyForCausalLM(config)
        model.resize_token_embeddings(tokenizer.vocab_size)
        return model

    kwargs: Dict[str, Any] = {}
    if BNB_AVAILABLE and (cfg.quant.load_in_4bit or cfg.quant.load_in_8bit):
        kwargs["device_map"] = "auto"
        if cfg.quant.load_in_4bit:
            kwargs["load_in_4bit"] = True
            kwargs["bnb_4bit_compute_dtype"] = torch.bfloat16 if cfg.quant.bnb_compute_dtype == "bfloat16" else torch.float16
            kwargs["bnb_4bit_use_double_quant"] = True
            kwargs["bnb_4bit_quant_type"] = "nf4"
        elif cfg.quant.load_in_8bit:
            kwargs["load_in_8bit"] = True

    attn_impl = None
    if cfg.flash_attn:
        attn_impl = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        attn_implementation=attn_impl if attn_impl else None,
        torch_dtype=torch.bfloat16 if cfg.bf16 else (torch.float16 if cfg.fp16 else None),
        trust_remote_code=True,
        **kwargs,
    )

    # LoRA/QLoRA
    if cfg.lora.enabled and PEFT_AVAILABLE:
        target_modules = None
        if cfg.lora.target_modules:
            target_modules = [s.strip() for s in cfg.lora.target_modules.split(",") if s.strip()]
        if cfg.quant.load_in_4bit or cfg.quant.load_in_8bit:
            model = prepare_model_for_kbit_training(model)
        lcfg = LoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        model = get_peft_model(model, lcfg)
        logging.info("[LoRA] 有効化しました。")

    model.resize_token_embeddings(len(tokenizer))
    return model


# ------------------------- Trainer build ------------------------- #

def build_trainer(
    cfg: TrainConfig,
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    train_dataset: HFDataset,
    eval_dataset: Optional[HFDataset] = None,
) -> Trainer:

    if cfg.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if cfg.torch_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)  # type: ignore
            logging.info("[compile] torch.compile を適用しました。")
        except Exception as e:
            logging.warning(f"[compile] 失敗しました: {e}")

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    optim = "adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch"

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        weight_decay=cfg.weight_decay,
        logging_dir=cfg.logging_dir,
        logging_steps=cfg.logging_steps,
        save_strategy=cfg.save_strategy,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        evaluation_strategy=cfg.evaluation_strategy,
        eval_steps=cfg.eval_steps,
        load_best_model_at_end=cfg.evaluation_strategy != "no",
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        optim=optim,
        dataloader_pin_memory=True,
        report_to=("wandb" if cfg.wandb else "tensorboard"),
        seed=cfg.seed,
        push_to_hub=cfg.push_to_hub,
        gradient_checkpointing=cfg.gradient_checkpointing,
        torch_compile=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
    )
    return trainer


# ------------------------- Data loading pipeline ------------------------- #

def prepare_datasets(cfg: TrainConfig, tokenizer: AutoTokenizer) -> Tuple[HFDataset, Optional[HFDataset], int]:
    if cfg.dataset_type == "huggingface":
        if not cfg.hf_dataset_id:
            raise ValueError("--hf_dataset_id が必要です。")
        raw = load_dataset(cfg.hf_dataset_id, cfg.hf_subset if cfg.hf_subset else None)
        if "train" not in raw:
            raise ValueError(f"{cfg.hf_dataset_id} に 'train' split がありません。")
        texts = format_conversation_hf(raw["train"], cfg.hf_input_col, cfg.hf_output_col)
        ds = build_hf_dataset_from_texts(texts)
    elif cfg.dataset_type == "azukif":
        if not cfg.dataset_path:
            raise ValueError("--dataset_path (AzukiF JSON) が必要です。")
        texts = format_conversation_azukif(cfg.dataset_path)
        ds = build_hf_dataset_from_texts(texts)
    else:
        raise ValueError("dataset_type は 'huggingface' か 'azukif' を指定してください。")

    cpu_count = max(1, os.cpu_count() or 1)
    chunks = tokenize_and_chunk(ds, tokenizer, cfg.block_size, num_proc=min(8, cpu_count))

    if cfg.do_eval_split and cfg.evaluation_strategy != "no":
        split = chunks["train"].train_test_split(test_size=cfg.eval_ratio, seed=cfg.seed)
        train_ds = split["train"]
        eval_ds = split["test"]
    else:
        train_ds = chunks["train"]
        eval_ds = None

    return train_ds, eval_ds, cfg.block_size


# ------------------------- ETA & Training orchestration ------------------------- #

def estimate_and_log_eta(trainer: Trainer, cfg: TrainConfig, train_ds: HFDataset) -> None:
    eff_batch = cfg.per_device_train_batch_size * max(1, trainer.args.gradient_accumulation_steps)
    steps_per_epoch = math.ceil(len(train_ds) / eff_batch)
    total_steps = int(steps_per_epoch * cfg.epochs)

    logging.info("\n[ETA] ベンチマークを開始します（実データ、最適化・混合精度含む）...")
    step_time = measure_step_time(trainer, steps=cfg.benchmark_steps, warmup=cfg.benchmark_warmup)
    remaining_steps = max(0, total_steps)
    eta_seconds = remaining_steps * step_time

    gpu = get_gpu_info()
    if gpu.count > 0:
        gpu_str = " / ".join([f"{n}({m:.1f}GB)" for n, m in zip(gpu.names, gpu.total_mem_gb)])
    else:
        gpu_str = "CPU"

    logging.info(
        (
            "\n[ETA] 推定結果\n"
            f"  - GPU: {gpu_str}\n"
            f"  - block_size: {cfg.block_size}, batch(per_device): {cfg.per_device_train_batch_size}, "
            f"grad_accum: {cfg.gradient_accumulation_steps}\n"
            f"  - steps/epoch: {steps_per_epoch}, total_steps: {total_steps}\n"
            f"  - measured step time: {step_time:.3f} s/step\n"
            f"  - Estimated Total Time: ~{human_time(eta_seconds)}"
        )
    )


def save_config(cfg: TrainConfig) -> None:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "zeta_llm_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)


def maybe_init_wandb(cfg: TrainConfig) -> None:
    if not cfg.wandb:
        return
    try:
        import wandb  # type: ignore
        wandb.init(project=cfg.wandb_project, config=asdict(cfg))
        logging.info("[W&B] ログ送信を開始しました。")
    except Exception as e:
        logging.warning(f"[W&B] 初期化に失敗: {e}")


def train_main(cfg: TrainConfig) -> None:
    setup_logging()

    set_seed(cfg.seed)
    if cfg.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)

    tokenizer = build_tokenizer(cfg)
    model = build_model(cfg, tokenizer)
    train_ds, eval_ds, _context_len = prepare_datasets(cfg, tokenizer)
    trainer = build_trainer(cfg, model, tokenizer, train_ds, eval_ds)

    save_config(cfg)
    maybe_init_wandb(cfg)
    estimate_and_log_eta(trainer, cfg, train_ds)

    logging.info("\n[Train] 学習を開始します。Ctrl+C で安全に中断可能です。")
    try:
        trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
    except KeyboardInterrupt:
        logging.warning("\n[Train] ユーザーにより中断されました。現在の状態を保存します...")
    finally:
        logging.info("[Train] モデルを保存中...")
        trainer.save_model(cfg.output_dir)
        tokenizer.save_pretrained(cfg.output_dir)
        logging.info(f"[Done] 保存先: {cfg.output_dir}")

    if cfg.wandb:
        try:
            import wandb  # type: ignore
            wandb.finish()
        except Exception:
            pass


# ------------------------- Argparse ------------------------- #

def load_config_file(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Zeta-LLM training utility (refactored)")
    # Config file
    p.add_argument("--config", type=str, default=None, help="JSON 設定ファイル")

    # Model/tokenizer
    p.add_argument("--base_model", type=str, default=None, help="HF repo or local path")
    p.add_argument("--tokenizer_path", type=str, default=None)
    p.add_argument("--use_empty_model", action="store_true")

    # Data
    p.add_argument("--dataset_type", type=str, choices=["azukif", "huggingface"], default="azukif")
    p.add_argument("--dataset_path", type=str, default=None, help="AzukiF JSON path")
    p.add_argument("--hf_dataset_id", type=str, default=None)
    p.add_argument("--hf_subset", type=str, default=None)
    p.add_argument("--hf_input_col", type=str, default="input")
    p.add_argument("--hf_output_col", type=str, default="output")
    p.add_argument("--block_size", type=int, default=1024)

    # Train
    p.add_argument("--output_dir", type=str, default="./results")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=3e-5)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_strategy", type=str, choices=["no", "steps", "epoch"], default="steps")
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--evaluation_strategy", type=str, choices=["no", "steps", "epoch"], default="no")
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--do_eval_split", action="store_true")
    p.add_argument("--eval_ratio", type=float, default=0.02)

    # Precision / Perf
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--no_bf16", action="store_true", help="bf16 を無効化")
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--no_tf32", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--no_gradient_checkpointing", action="store_true")
    p.add_argument("--flash_attn", action="store_true")
    p.add_argument("--no_flash_attn", action="store_true")
    p.add_argument("--torch_compile", action="store_true")

    # Repro
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true")

    # Extras
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="zeta-llm")
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    p.add_argument("--logging_dir", type=str, default="./logs")

    # LoRA
    p.add_argument("--lora", action="store_true", help="LoRA を有効化")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", type=str, default=None)

    # Quant
    p.add_argument("--load_in_8bit", action="store_true")
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--bnb_compute_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])

    # ETA
    p.add_argument("--benchmark_steps", type=int, default=10)
    p.add_argument("--benchmark_warmup", type=int, default=3)

    return p


def merge_args_to_config(args: argparse.Namespace, file_cfg: Dict[str, Any]) -> TrainConfig:
    cfg_dict = asdict(TrainConfig())
    cfg_dict.update(file_cfg or {})
    cli = vars(args).copy()

    if cli.get("no_bf16"):
        cli["bf16"] = False
    if cli.get("no_tf32"):
        cli["tf32"] = False
    if cli.get("no_gradient_checkpointing"):
        cli["gradient_checkpointing"] = False
    if cli.get("no_flash_attn"):
        cli["flash_attn"] = False

    lora = cfg_dict.get("lora", {})
    lora.update({
        "enabled": cli.pop("lora", lora.get("enabled", False)),
        "r": cli.pop("lora_r", lora.get("r", 16)),
        "alpha": cli.pop("lora_alpha", lora.get("alpha", 32)),
        "dropout": cli.pop("lora_dropout", lora.get("dropout", 0.05)),
        "target_modules": cli.pop("lora_target_modules", lora.get("target_modules")),
    })
    quant = cfg_dict.get("quant", {})
    quant.update({
        "load_in_8bit": cli.pop("load_in_8bit", quant.get("load_in_8bit", False)),
        "load_in_4bit": cli.pop("load_in_4bit", quant.get("load_in_4bit", False)),
        "bnb_compute_dtype": cli.pop("bnb_compute_dtype", quant.get("bnb_compute_dtype", "bfloat16")),
    })

    cfg_dict.update(cli)
    cfg_dict["lora"] = lora
    cfg_dict["quant"] = quant

    cfg = TrainConfig(**cfg_dict)
    if not cfg.fp16 and not cfg.bf16:
        cfg.bf16 = True
    return cfg


# ------------------------- Entry ------------------------- #

def main():
    parser = build_argparser()
    args = parser.parse_args()

    file_cfg = load_config_file(args.config)
    cfg = merge_args_to_config(args, file_cfg)

    if not cfg.use_empty_model and not cfg.base_model:
        if sys.stdin.isatty():
            cfg.base_model = input("Base model を指定してください (HF も可): ").strip()
        if not cfg.base_model:
            raise ValueError("base_model が未指定です。--use_empty_model または --base_model を指定してください。")

    if cfg.dataset_type == "azukif" and not cfg.dataset_path:
        if sys.stdin.isatty():
            cfg.dataset_path = input("AzukiF JSON のパス: ").strip()
    if cfg.dataset_type == "huggingface" and not cfg.hf_dataset_id:
        if sys.stdin.isatty():
            cfg.hf_dataset_id = input("Hugging Face データセット ID (user/repo): ").strip()

    train_main(cfg)

if __name__ == "__main__":
    main()
