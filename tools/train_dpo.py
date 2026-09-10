#!/usr/bin/env python3
"""
CyberAgent DPO Fine-Tuning Pipeline
===================================
Fine-tunes a student LLM (e.g., Qwen3.5-0.8B) on paired preference data
using Unsloth, LoRA, and Hugging Face TRL (DPOTrainer).

Outputs:
1. LoRA adapter weights (safetensors)
2. Merged model or direct Q8_0 / FP16 GGUF export for Ollama
3. Auto-generated Ollama Modelfile for seamless local deployment.
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, Any, List

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Python 3.14 compatibility patch for HuggingFace datasets and dill pickler
try:
    import datasets.utils._dill
    import dill
    def _py314_compat_batch_setitems(self, items, obj=None):
        if getattr(self, '_legacy_no_dict_keys_sorting', False):
            try:
                return super(datasets.utils._dill.Pickler, self)._batch_setitems(items, obj)
            except TypeError:
                return super(datasets.utils._dill.Pickler, self)._batch_setitems(items)
        try:
            items = sorted(items)
        except Exception:
            from datasets.fingerprint import Hasher
            items = sorted(items, key=lambda x: Hasher.hash(x[0]))
        try:
            dill.Pickler._batch_setitems(self, items, obj)
        except TypeError:
            dill.Pickler._batch_setitems(self, items)
    datasets.utils._dill.Pickler._batch_setitems = _py314_compat_batch_setitems
except Exception:
    pass

import torch
from datasets import Dataset

# Unsloth acceleration patch
from unsloth import FastLanguageModel, PatchDPOTrainer
PatchDPOTrainer()

from trl import DPOTrainer, DPOConfig
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("train_dpo")


def format_dataset_for_dpo(
    dataset_path: str,
    tokenizer: Any,
    task_filter: str = "all",
    exclude_scenarios: Optional[List[str]] = None,
    eval_split: float = 0.1,
    eval_output_path: Optional[str] = None
) -> Tuple[Dataset, Optional[Dataset], int, int]:
    """
    Loads raw CyberAgent DPO dataset and formats it for DPOTrainer:
    - prompt: ChatML formatted prompt ending with <|im_start|>assistant\\n
    - chosen: assistant response ending with <|im_end|>
    - rejected: student response ending with <|im_end|>
    
    Supports:
    - Excluding specific scenarios (Leave-One-Scenario-Out)
    - Random fractional validation split (eval_split)
    - Exporting full evaluation dataset with ground truth metadata to eval_output_path
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: '{dataset_path}'")

    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    logger.info("Loaded %d raw entries from '%s'", len(raw_data), dataset_path)

    train_candidates: List[Dict[str, Any]] = []
    excluded_rows: List[Dict[str, Any]] = []
    skipped = 0

    excluded_set = set(exclude_scenarios) if exclude_scenarios else set()

    for entry in raw_data:
        task = entry.get("task", "")
        scenario = entry.get("scenario", "")
        if task_filter != "all" and task != task_filter:
            continue

        chosen = entry.get("chosen", "").strip()
        rejected = entry.get("rejected", "").strip()
        messages = entry.get("messages", [])

        if not chosen or not rejected:
            skipped += 1
            continue

        if messages:
            raw_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            # Ensure clean prompt ending right before assistant output
            if "<think>\n\n</think>" in raw_prompt:
                clean_prompt = raw_prompt.replace("<think>\n\n</think>\n\n", "").strip() + "\n"
            else:
                clean_prompt = raw_prompt
        else:
            clean_prompt = entry.get("prompt", "").strip()

        # Append ChatML end of message token if not present
        eos = tokenizer.eos_token or "<|im_end|>"
        if not chosen.endswith(eos):
            chosen_formatted = f"{chosen}{eos}"
        else:
            chosen_formatted = chosen

        if not rejected.endswith(eos):
            rejected_formatted = f"{rejected}{eos}"
        else:
            rejected_formatted = rejected

        formatted_item = {
            "prompt": clean_prompt,
            "chosen": chosen_formatted,
            "rejected": rejected_formatted,
            # Retain original metadata for evaluation exports
            "_raw_entry": entry
        }

        if scenario in excluded_set:
            excluded_rows.append(formatted_item)
        else:
            train_candidates.append(formatted_item)

    if excluded_rows:
        logger.info("🚫 Excluded %d sample(s) from scenarios: %s", len(excluded_rows), list(excluded_set))

    # Split train vs validation
    eval_rows: List[Dict[str, Any]] = list(excluded_rows)

    if eval_split > 0 and len(train_candidates) > 10:
        import random
        random.seed(3407)
        random.shuffle(train_candidates)
        split_idx = int(len(train_candidates) * (1.0 - eval_split))
        split_train = train_candidates[:split_idx]
        split_val = train_candidates[split_idx:]
        eval_rows.extend(split_val)
        train_rows = split_train
    else:
        train_rows = train_candidates

    # Convert to HF Dataset (stripping internal _raw_entry)
    train_hf = Dataset.from_list([
        {"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]}
        for r in train_rows
    ])

    eval_hf = Dataset.from_list([
        {"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]}
        for r in eval_rows
    ]) if eval_rows else None

    # Export full raw evaluation dataset to disk
    if eval_output_path and eval_rows:
        os.makedirs(os.path.dirname(os.path.abspath(eval_output_path)) or ".", exist_ok=True)
        raw_eval_export = [r["_raw_entry"] for r in eval_rows]
        with open(eval_output_path, "w", encoding="utf-8") as f:
            json.dump(raw_eval_export, f, indent=2, ensure_ascii=False)
        logger.info("💾 Saved %d evaluation samples with ground truth to '%s'", len(raw_eval_export), eval_output_path)

    logger.info("Dataset Ready: %d training samples | %d validation samples (skipped %d incomplete)",
                len(train_hf), len(eval_hf) if eval_hf else 0, skipped)

    return train_hf, eval_hf, len(train_hf), len(eval_hf) if eval_hf else 0


def generate_ollama_modelfile(output_dir: str, gguf_path: str, base_model_name: str):
    """Creates a ready-to-use Modelfile for Ollama integration."""
    modelfile_path = os.path.join(output_dir, "Modelfile")
    rel_gguf = os.path.abspath(gguf_path)

    content = f"""# CyberAgent Fine-Tuned DPO Model
FROM {rel_gguf}

TEMPLATE \"\"\"{{{{- if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{- end }}}}
{{{{- range .Messages }}}}
<|im_start|>{{{{ .Role }}}}
{{{{ .Content }}}}<|im_end|>
{{{{- end }}}}
<|im_start|>assistant
\"\"\"

PARAMETER temperature 0.2
PARAMETER top_p 0.95
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|endoftext|>"
"""
    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info("📝 Generated Ollama Modelfile at '%s'", modelfile_path)
    logger.info("👉 To load into Ollama: ollama create cyberagent-dpo -f %s", modelfile_path)


def main():
    parser = argparse.ArgumentParser(description="CyberAgent DPO Fine-Tuning Engine with Unsloth")
    parser.add_argument("--dataset", "-d", default="data/datasets/test_cyberagent_dpo.json", help="Path to input DPO dataset JSON")
    parser.add_argument("--model", "-m", default="unsloth/Qwen3.5-0.8B", help="Base HuggingFace / Unsloth model name")
    parser.add_argument("--output", "-o", default="weights/cyberagent-dpo-qwen3.5-0.8b", help="Output directory for adapters and weights")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Maximum sequence length (default: 2048)")
    parser.add_argument("--lora-rank", "-r", type=int, default=16, help="LoRA rank dimension")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha scaling factor")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO loss beta temperature")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device train batch size (default: 1 for VRAM safety)")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps (default: 8, effective batch size: 8)")
    parser.add_argument("--load-in-4bit", action="store_true", default=False, help="Enable 4-bit QLoRA mode if running very low on VRAM (default: False, 16-bit)")
    parser.add_argument("--eval-split", type=float, default=0.1, help="Fraction of dataset reserved for validation/test (0.0 to 0.5)")
    parser.add_argument("--exclude-scenarios", type=str, default=None, help="Comma-separated scenario names to exclude from training and reserve for evaluation (e.g. 'scenario4_auto')")
    parser.add_argument("--eval-output", type=str, default=None, help="Output path to save evaluation dataset with ground truth (defaults to <output>/eval_dataset.json)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--export-gguf", choices=["q8_0", "f16", "q4_k_m", "none"], default="q8_0", help="GGUF quantization format for Ollama export")
    parser.add_argument("--task-filter", choices=["all", "query_generation", "triage_synthesis"], default="all", help="Filter specific task type in dataset")

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    eval_out_path = args.eval_output or os.path.join(args.output, "eval_dataset.json")
    excluded_scenarios_list = [s.strip() for s in args.exclude_scenarios.split(",") if s.strip()] if args.exclude_scenarios else None

    print("\n" + "=" * 65)
    print("🚀 CyberAgent DPO Fine-Tuning with Unsloth")
    print(f"  • Base Model        : {args.model}")
    print(f"  • Dataset Source    : {args.dataset}")
    print(f"  • Target Device     : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  • Precision / Mode  : {'4-bit QLoRA (Memory-Safe)' if args.load_in_4bit else '16-bit Full Precision'}")
    print(f"  • Max Seq Length    : {args.max_seq_length}")
    print(f"  • LoRA Config       : r={args.lora_rank}, alpha={args.lora_alpha}")
    print(f"  • DPO Beta (Margin) : {args.beta}")
    print(f"  • Batch Size (Eff.) : {args.batch_size * args.grad_accum} (batch={args.batch_size}, accum={args.grad_accum})")
    print(f"  • Eval Split Ratio  : {args.eval_split * 100:.0f}% validation")
    if excluded_scenarios_list:
        print(f"  • Excluded Scenarios: {excluded_scenarios_list}")
    print(f"  • Eval Output Path  : {eval_out_path}")
    print(f"  • Epochs / LR       : {args.epochs} epochs @ {args.lr}")
    print(f"  • GGUF Export Target: {args.export_gguf.upper()}")
    print("=" * 65 + "\n")

    # 1. Load Model & Tokenizer via Unsloth FastLanguageModel
    logger.info("Loading base model '%s' (4-bit=%s)...", args.model, args.load_in_4bit)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=None,  # Auto detection (float16/bfloat16)
        load_in_4bit=args.load_in_4bit,
    )

    # 2. Attach LoRA Adapters
    """
    logger.info("Attaching LoRA adapters (r=%d, alpha=%d)...", args.lora_rank, args.lora_alpha)
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=3407,
    )
    """
    # 2. Attach LoRA Adapters (skip if already loaded from SFT checkpoint)
    if hasattr(model, "peft_config") and model.peft_config:
        logger.info("Model already has LoRA adapters attached from SFT. Enabling training mode.")
        FastLanguageModel.for_training(model)
    else:
        logger.info("Attaching LoRA adapters (r=%d, alpha=%d)...", args.lora_rank, args.lora_alpha)
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=args.lora_alpha,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing=True,
            random_state=3407,
        )

    # 3. Format Dataset & Train/Test Split
    train_dataset, eval_dataset, n_train, n_eval = format_dataset_for_dpo(
        dataset_path=args.dataset,
        tokenizer=tokenizer,
        task_filter=args.task_filter,
        exclude_scenarios=excluded_scenarios_list,
        eval_split=args.eval_split,
        eval_output_path=eval_out_path
    )

    # 4. Configure DPO Training
    logger.info("Initializing DPOTrainer...")
    max_target_len = 512
    max_prompt_len = max(512, args.max_seq_length - max_target_len)

    training_args = DPOConfig(
        output_dir=args.output,
        beta=args.beta,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_length=args.max_seq_length,
        max_prompt_length=max_prompt_len,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        eval_strategy="epoch" if eval_dataset is not None else "no",
        logging_steps=10,
        save_strategy="epoch",
        optim="adamw_8bit",
        seed=3407,
        report_to="none"
    )

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Unsloth handles implicit reference model without duplicate VRAM
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    # 5. Execute Training
    logger.info("🔥 Starting DPO optimization loop...")
    train_result = dpo_trainer.train()

    logger.info("Training completed in %.2f seconds! Final Train Loss: %.4f",
                train_result.metrics.get("train_runtime", 0),
                train_result.metrics.get("train_loss", 0.0))

    if eval_dataset is not None:
        eval_metrics = dpo_trainer.evaluate()
        logger.info("📊 Validation Metrics: %s", eval_metrics)

    # 6. Save LoRA Adapters
    adapter_path = os.path.join(args.output, "lora_adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    logger.info("💾 Saved LoRA adapters to '%s'", adapter_path)

    # 7. Export GGUF for Ollama
    if args.export_gguf != "none":
        gguf_dir = os.path.join(args.output, f"gguf_{args.export_gguf}")
        logger.info("📦 Exporting merged model to GGUF format ('%s')...", args.export_gguf)
        try:
            model.save_pretrained_gguf(
                gguf_dir,
                tokenizer,
                quantization_method=args.export_gguf
            )
            # Find the exported .gguf file
            exported_gguf = None
            for fname in os.listdir(gguf_dir):
                if fname.endswith(".gguf"):
                    exported_gguf = os.path.join(gguf_dir, fname)
                    break

            if exported_gguf:
                logger.info("✅ Successfully exported GGUF model: '%s'", exported_gguf)
                generate_ollama_modelfile(args.output, exported_gguf, args.model)
            else:
                logger.info("GGUF directory: '%s'", gguf_dir)
        except Exception as e:
            logger.warning("GGUF export encountered an issue: %s", e)
            logger.info("LoRA adapters remain safely stored at '%s'", adapter_path)

    print("\n" + "=" * 65)
    print("🎉 CyberAgent DPO Fine-Tuning Complete!")
    print(f"  • LoRA Weights : {adapter_path}")
    print(f"  • Modelfile    : {os.path.join(args.output, 'Modelfile')}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
