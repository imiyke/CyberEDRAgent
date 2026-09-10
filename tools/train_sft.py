#!/usr/bin/env python3
"""
CyberAgent SFT Fine-Tuning Pipeline
===================================
Supervised Fine-Tuning (SFT) for student LLM (e.g., Qwen3.5-0.8B) on chosen
expert reasoning (<think>) and strict JSON output formatting.

Uses Unsloth + LoRA + TRL (SFTTrainer) with response-only loss masking.

Outputs:
1. LoRA adapter weights (safetensors)
2. Merged model or direct Q8_0 / FP16 GGUF export for Ollama
3. Auto-generated Ollama Modelfile for immediate deployment.
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, Any, List, Optional, Tuple

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

# Unsloth acceleration
from unsloth import FastLanguageModel, train_on_responses_only
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("train_sft")


def format_dataset_for_sft(
    dataset_path: str,
    tokenizer: Any,
    task_filter: str = "all",
    exclude_scenarios: Optional[List[str]] = None,
    eval_split: float = 0.1,
    eval_output_path: Optional[str] = None
) -> Tuple[Dataset, Optional[Dataset], int, int]:
    """
    Loads CyberAgent dataset and prepares conversational SFT texts:
    <|im_start|>system...<|im_end|>
    <|im_start|>user...<|im_end|>
    <|im_start|>assistant
    <think>
    1. log_source: ...
    2. target_entities: ...
    3. observed_intent: ...
    4. threat_assessment: ...
    </think>
    {
      "status": ...
    }<|im_end|>
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
        messages = entry.get("messages", [])

        if not chosen:
            skipped += 1
            continue

        eos = tokenizer.eos_token or "<|im_end|>"
        if not chosen.endswith(eos):
            chosen_clean = f"{chosen}{eos}"
        else:
            chosen_clean = chosen

        # Format full conversation text
        if messages:
            # Reconstruct conversational turns: system + human + chosen assistant
            formatted_messages = []
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                if role in ["system", "user"]:
                    formatted_messages.append({"role": role, "content": content})
            formatted_messages.append({"role": "assistant", "content": chosen_clean})

            full_text = tokenizer.apply_chat_template(
                formatted_messages,
                tokenize=False,
                add_generation_prompt=False
            )
        else:
            prompt = entry.get("prompt", "").strip()
            full_text = f"{prompt}{chosen_clean}"

        formatted_item = {
            "text": full_text,
            "scenario": scenario,
            "task": task,
            "raw_entry": entry
        }

        if scenario in excluded_set:
            excluded_rows.append(formatted_item)
        else:
            train_candidates.append(formatted_item)

    logger.info("Processed dataset: %d candidates, %d excluded (holdout), %d skipped",
                len(train_candidates), len(excluded_rows), skipped)

    eval_rows: List[Dict[str, Any]] = []
    if excluded_rows:
        eval_rows = excluded_rows
        train_rows = train_candidates
        logger.info("Using %d holdout scenario rows for evaluation.", len(eval_rows))
    elif eval_split > 0.0 and len(train_candidates) > 10:
        import random
        random.seed(3407)
        shuffled = list(train_candidates)
        random.shuffle(shuffled)
        n_eval = int(len(shuffled) * eval_split)
        eval_rows = shuffled[:n_eval]
        train_rows = shuffled[n_eval:]
        logger.info("Random split: %d train rows, %d eval rows (%.1f%%)",
                    len(train_rows), len(eval_rows), eval_split * 100)
    else:
        train_rows = train_candidates
        eval_rows = []

    # Export eval dataset with ground truth metadata
    if eval_output_path and eval_rows:
        os.makedirs(os.path.dirname(os.path.abspath(eval_output_path)), exist_ok=True)
        eval_export_data = []
        for r in eval_rows:
            raw_e = r["raw_entry"]
            eval_export_data.append({
                "id": raw_e.get("chunk_id", raw_e.get("id", f"eval_{len(eval_export_data)}")),
                "scenario": raw_e.get("scenario", "unknown"),
                "chunk_id": raw_e.get("chunk_id", ""),
                "task": raw_e.get("task", ""),
                "expected_status": raw_e.get("expected_status", "clear"),
                "expected_category": raw_e.get("expected_category", "noise"),
                "expected_mitre_id": raw_e.get("expected_mitre_id"),
                "raw_logs": raw_e.get("raw_logs", ""),
                "prompt": raw_e.get("prompt", ""),
                "messages": raw_e.get("messages", []),
                "ground_truth_chosen": raw_e.get("chosen", "")
            })
        with open(eval_output_path, "w", encoding="utf-8") as f:
            json.dump(eval_export_data, f, indent=2, ensure_ascii=False)
        logger.info("Saved %d evaluation items with ground truth to '%s'",
                    len(eval_export_data), eval_output_path)

    train_hf = Dataset.from_list([{"text": r["text"]} for r in train_rows])
    eval_hf = Dataset.from_list([{"text": r["text"]} for r in eval_rows]) if eval_rows else None

    return train_hf, eval_hf, len(train_rows), len(eval_rows)


def generate_ollama_modelfile(output_dir: str, gguf_path: str, base_model_name: str):
    """Creates a clean Modelfile for Ollama deployment."""
    modelfile_path = os.path.join(output_dir, "Modelfile")
    rel_gguf = os.path.abspath(gguf_path)

    content = f"""# CyberAgent SFT Fine-Tuned Model
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

PARAMETER temperature 0.1
PARAMETER top_p 0.95
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|endoftext|>"
"""
    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info("📝 Generated Ollama Modelfile at '%s'", modelfile_path)
    logger.info("👉 To load into Ollama: ollama create cyberagent-sft -f %s", modelfile_path)


def main():
    parser = argparse.ArgumentParser(description="CyberAgent Supervised Fine-Tuning (SFT) with Unsloth")
    parser.add_argument("--dataset", "-d", default="data/datasets/test_cyberagent_dpo.json", help="Path to input dataset JSON")
    parser.add_argument("--model", "-m", default="unsloth/Qwen3.5-0.8B", help="Base HuggingFace / Unsloth model name")
    parser.add_argument("--output", "-o", default="weights/cyberagent-sft-qwen3.5-0.8b", help="Output directory for adapters and weights")
    parser.add_argument("--max-seq-length", type=int, default=1536, help="Maximum sequence length (default: 1536)")
    parser.add_argument("--lora-rank", "-r", type=int, default=16, help="LoRA rank dimension")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha scaling factor")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device train batch size (default: 4)")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps (default: 2, effective batch: 8)")
    parser.add_argument("--load-in-4bit", action="store_true", default=False, help="Enable 4-bit QLoRA mode (default: False for 24GB GPUs)")
    parser.add_argument("--eval-split", type=float, default=0.1, help="Fraction of dataset reserved for validation (0.0 to 0.5)")
    parser.add_argument("--exclude-scenarios", type=str, default=None, help="Comma-separated scenario names to exclude (e.g. 'scenario4_auto')")
    parser.add_argument("--eval-output", type=str, default=None, help="Output path for eval dataset JSON")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (default: 3)")
    parser.add_argument("--export-gguf", choices=["q8_0", "f16", "q4_k_m", "none"], default="q8_0", help="GGUF export format")
    parser.add_argument("--task-filter", choices=["all", "query_generation", "triage_synthesis"], default="all", help="Task filter")

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    eval_out_path = args.eval_output or os.path.join(args.output, "eval_dataset.json")
    excluded_scenarios_list = [s.strip() for s in args.exclude_scenarios.split(",") if s.strip()] if args.exclude_scenarios else None

    print("\n" + "=" * 65)
    print("🚀 CyberAgent Supervised Fine-Tuning (SFT) with Unsloth")
    print(f"  • Base Model        : {args.model}")
    print(f"  • Dataset Source    : {args.dataset}")
    print(f"  • Target Device     : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  • Precision / Mode  : {'4-bit QLoRA' if args.load_in_4bit else '16-bit Full Precision (BF16)'}")
    print(f"  • Max Seq Length    : {args.max_seq_length}")
    print(f"  • LoRA Config       : r={args.lora_rank}, alpha={args.lora_alpha}")
    print(f"  • Batch Size (Eff.) : {args.batch_size * args.grad_accum} (batch={args.batch_size}, accum={args.grad_accum})")
    print(f"  • Eval Split Ratio  : {args.eval_split * 100:.0f}% validation")
    if excluded_scenarios_list:
        print(f"  • Excluded Scenarios: {excluded_scenarios_list}")
    print(f"  • Eval Output Path  : {eval_out_path}")
    print(f"  • Epochs / LR       : {args.epochs} epochs @ {args.lr}")
    print(f"  • GGUF Export Target: {args.export_gguf.upper()}")
    print("=" * 65 + "\n")

    # 1. Load Model & Tokenizer
    logger.info("Loading base model '%s' (4-bit=%s)...", args.model, args.load_in_4bit)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )

    # 2. Attach LoRA
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

    # 3. Format Dataset
    train_dataset, eval_dataset, n_train, n_eval = format_dataset_for_sft(
        dataset_path=args.dataset,
        tokenizer=tokenizer,
        task_filter=args.task_filter,
        exclude_scenarios=excluded_scenarios_list,
        eval_split=args.eval_split,
        eval_output_path=eval_out_path
    )

    # 4. Configure SFT Training
    logger.info("Initializing SFTTrainer...")
    training_args = SFTConfig(
        output_dir=args.output,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        eval_strategy="epoch" if eval_dataset is not None else "no",
        logging_steps=10,
        save_strategy="epoch",
        optim="adamw_8bit",
        seed=3407,
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    # Apply response-only loss masking for ChatML
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    # 5. Execute Training
    logger.info("🔥 Starting SFT optimization loop...")
    train_result = trainer.train()

    logger.info("Training completed in %.2f seconds! Final Train Loss: %.4f",
                train_result.metrics.get("train_runtime", 0),
                train_result.metrics.get("train_loss", 0.0))

    if eval_dataset is not None:
        eval_metrics = trainer.evaluate()
        logger.info("📊 Validation Metrics: %s", eval_metrics)

    # 6. Save LoRA Adapters
    adapter_path = os.path.join(args.output, "lora_adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    logger.info("💾 Saved LoRA adapters to '%s'", adapter_path)

    # 7. Export GGUF
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
            if os.path.exists(gguf_dir):
                for fname in os.listdir(gguf_dir):
                    if fname.endswith(".gguf") and not fname.endswith("mmproj.gguf"):
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
    print("🎉 CyberAgent SFT Fine-Tuning Complete!")
    print(f"  • LoRA Weights : {adapter_path}")
    print(f"  • Modelfile    : {os.path.join(args.output, 'Modelfile')}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
