"""DPO training entry point (called by scripts/train_dpo.py)."""
from __future__ import annotations

import inspect
from pathlib import Path

from trl import DPOConfig, DPOTrainer

from src.data import load_ethics_dpo, load_pku_dpo
from src.model import attach_lora, load_base_model, load_tokenizer
from src.utils import get_logger, set_seed

log = get_logger(__name__)


def _filter_kwargs(cls, kwargs: dict) -> dict:
    """Keep only kwargs supported by `cls.__init__` (silently drop the rest).

    TRL renames/removes DPOConfig kwargs between minor versions (e.g.
    `max_prompt_length` / `max_length` moved out of DPOConfig in some
    releases). Filtering avoids hard-failing on a missing kwarg.
    """
    sig = inspect.signature(cls.__init__)
    valid = {name for name in sig.parameters}
    kept = {k: v for k, v in kwargs.items() if k in valid}
    dropped = sorted(set(kwargs) - set(kept))
    if dropped:
        log.warning(f"{cls.__name__} dropped unsupported kwargs: {dropped}")
    return kept


def train_dpo(cfg: dict) -> str:
    set_seed(cfg["training"]["seed"])

    tokenizer = load_tokenizer(cfg["model"])
    model = load_base_model(cfg["model"], for_training=True)
    model = attach_lora(model, cfg["lora"])

    source = cfg["data"].get("source", "pku")
    if source == "ethics_train":
        train_ds = load_ethics_dpo(
            tokenizer=tokenizer,
            categories=cfg["data"].get("ethics_categories",
                ("commonsense", "deontology", "justice", "virtue")),
            n_per_category=cfg["data"]["n_per_category"],
            seed=cfg["training"]["seed"],
        )
        eval_ds = None  # ETHICS train: no held-out preference eval
    else:
        train_ds = load_pku_dpo(
            tokenizer=tokenizer,
            split=cfg["data"]["train_split"],
            max_samples=cfg["data"]["max_train_samples"],
            seed=cfg["training"]["seed"],
        )
        eval_ds = None
        if cfg["data"].get("eval_split"):
            eval_ds = load_pku_dpo(
                tokenizer=tokenizer,
                split=cfg["data"]["eval_split"],
                max_samples=cfg["data"]["max_eval_samples"],
                seed=cfg["training"]["seed"],
            )

    output_dir = cfg["training"]["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    requested_kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=cfg["training"]["num_train_epochs"],
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["training"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        warmup_ratio=cfg["training"]["warmup_ratio"],
        weight_decay=cfg["training"]["weight_decay"],
        optim=cfg["training"]["optim"],
        max_grad_norm=cfg["training"]["max_grad_norm"],
        bf16=cfg["training"]["bf16"],
        fp16=cfg["training"]["fp16"],
        gradient_checkpointing=cfg["training"]["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=cfg["training"]["logging_steps"],
        save_steps=cfg["training"]["save_steps"],
        eval_steps=cfg["training"]["eval_steps"],
        eval_strategy="steps" if eval_ds is not None else "no",
        save_total_limit=cfg["training"]["save_total_limit"],
        report_to=cfg["training"]["report_to"],
        seed=cfg["training"]["seed"],
        beta=cfg["dpo"]["beta"],
        loss_type=cfg["dpo"]["loss_type"],
        label_smoothing=cfg["dpo"]["label_smoothing"],
        max_prompt_length=cfg["data"]["max_prompt_length"],
        max_length=cfg["data"]["max_length"],
        remove_unused_columns=False,
    )
    dpo_args = DPOConfig(**_filter_kwargs(DPOConfig, requested_kwargs))

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # PEFT: ref = base model with adapters disabled
        args=dpo_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    log.info("Starting DPO training")
    trainer.train()

    final_dir = str(Path(output_dir) / "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    log.info(f"Saved final adapter to {final_dir}")
    return final_dir
