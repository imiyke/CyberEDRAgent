#!/usr/bin/env python3
"""
CyberAgent SOC Evaluation & Benchmark Engine
============================================
Evaluates fine-tuned or baseline LLM models against ground-truth security datasets:
1. Computes Confusion Matrix (Alert / Suspicious / Clear)
2. Calculates Key SOC Metrics (FPR on benign, Threat Recall, MITRE accuracy)
3. Verifies Chain-of-Thought (<think>) and JSON Schema compliance
4. Generates a clean Scorecard report.
"""

import os
import sys
import json
import logging
import argparse
import time
from typing import Dict, Any, List, Optional, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from config.llm_factory import get_llm, load_config
from tools.dpo_engine import load_scenario_chunks, dict_list_to_messages, SYNTHESIS_SYSTEM
from tools.utils import parse_llm_response

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("eval_benchmark")


def load_evaluation_targets(
    eval_file: Optional[str] = None,
    scenario_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Loads evaluation items from either an eval_dataset.json file or a scenario folder."""
    targets = []

    if eval_file and os.path.exists(eval_file):
        with open(eval_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            if item.get("task") == "triage_synthesis" or "expected_status" in item:
                targets.append({
                    "id": item.get("id", item.get("chunk_id", "sample")),
                    "scenario": item.get("scenario", "eval_split"),
                    "chunk_id": item.get("chunk_id", ""),
                    "expected_status": item.get("expected_status", "clear").lower(),
                    "expected_category": item.get("expected_category", "noise"),
                    "expected_mitre_id": item.get("expected_mitre_id"),
                    "messages": item.get("messages", []),
                    "prompt": item.get("prompt", ""),
                    "raw_logs": item.get("raw_logs", "")
                })
        logger.info("Loaded %d evaluation items from '%s'", len(targets), eval_file)

    elif scenario_path and os.path.exists(scenario_path):
        chunks = load_scenario_chunks(scenario_path)
        for c in chunks:
            targets.append({
                "id": c["chunk_id"],
                "scenario": c["scenario_name"],
                "chunk_id": c["chunk_id"],
                "expected_status": c["expected_status"].lower(),
                "expected_category": c["expected_category"],
                "expected_mitre_id": c["expected_mitre_id"],
                "messages": [],
                "prompt": "",
                "raw_logs": c["raw_logs"]
            })
        logger.info("Loaded %d evaluation items from scenario '%s'", len(targets), scenario_path)

    else:
        raise ValueError("Must provide a valid --eval-file or --scenario path.")

    return targets


def run_benchmark(
    targets: List[Dict[str, Any]],
    model_name: str,
    config: Optional[dict] = None,
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """Runs model evaluation against all targets and computes SOC metrics."""
    cfg = config or load_config()
    
    # Configure model override
    cfg_eval = dict(cfg)
    if "llms" not in cfg_eval:
        cfg_eval["llms"] = {}
    cfg_eval["llms"]["eval"] = {"provider": "ollama", "model": model_name}

    llm = get_llm("eval", cfg_eval)

    if limit and limit > 0:
        targets = targets[:limit]

    logger.info("🚀 Starting Benchmark Evaluation for model '%s' (%d samples)...", model_name, len(targets))

    # Confusion matrix: [actual][predicted]
    # Statuses: alert, suspicious, clear
    cm = {
        "alert": {"alert": 0, "suspicious": 0, "clear": 0, "invalid": 0},
        "suspicious": {"alert": 0, "suspicious": 0, "clear": 0, "invalid": 0},
        "clear": {"alert": 0, "suspicious": 0, "clear": 0, "invalid": 0}
    }

    metrics = {
        "total": len(targets),
        "cot_compliance": 0,
        "json_compliance": 0,
        "mitre_matches": 0,
        "mitre_evaluable": 0,
        "latencies": []
    }

    results_log = []

    for idx, target in enumerate(targets, 1):
        target_id = target["id"]
        expected_status = target["expected_status"]
        expected_mitre = target["expected_mitre_id"]

        logger.info("[%d/%d] Testing %s (Expected: %s)...", idx, len(targets), target_id, expected_status)

        # Prepare clean single-turn messages (System + Target raw logs)
        if target.get("raw_logs"):
            user_text = f"Raw Logs:
{target['raw_logs']}"
        elif target.get("messages"):
            human_msgs = [m["content"] for m in target["messages"] if m.get("role") in ["user", "human"]]
            user_text = human_msgs[-1] if human_msgs else ""
        else:
            user_text = target.get("prompt", "")

        messages = [
            SystemMessage(content=SYNTHESIS_SYSTEM),
            HumanMessage(content=user_text)
        ]
        t0 = time.time()
        try:
            resp = llm.invoke(messages)
            latency = time.time() - t0
            metrics["latencies"].append(latency)
            parsed = parse_llm_response(resp)
        except Exception as e:
            logger.warning("Invocation error on %s: %s", target_id, e)
            cm[expected_status]["invalid"] += 1
            continue

        thinking = parsed.get("thinking", "")
        output_str = parsed.get("output", "").strip()

        # Check CoT presence
        has_cot = bool(thinking and len(thinking.strip()) > 10)
        if has_cot:
            metrics["cot_compliance"] += 1

        # Check JSON parsing
        pred_status = "invalid"
        pred_mitre = None
        is_valid_json = False

        try:
            import re
            json_match = re.search(r"\{.*\}", output_str, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                if "status" in data:
                    pred_status = str(data["status"]).lower()
                    pred_mitre = data.get("mitre_id")
                    is_valid_json = True
                    metrics["json_compliance"] += 1
        except Exception:
            pass

        if pred_status not in ["alert", "suspicious", "clear"]:
            pred_status = "invalid"

        cm[expected_status][pred_status] += 1

        # Check MITRE match if expected
        if expected_mitre:
            metrics["mitre_evaluable"] += 1
            if pred_mitre and (pred_mitre.strip().upper() == expected_mitre.strip().upper()):
                metrics["mitre_matches"] += 1

        results_log.append({
            "id": target_id,
            "expected_status": expected_status,
            "predicted_status": pred_status,
            "expected_mitre": expected_mitre,
            "predicted_mitre": pred_mitre,
            "has_cot": has_cot,
            "valid_json": is_valid_json,
            "latency": latency
        })

    # Compute high-level summary
    total_eval = len(targets)
    benign_total = sum(cm["clear"].values())
    benign_fp = cm["clear"]["alert"] + cm["clear"]["suspicious"]
    fp_rate = (benign_fp / max(1, benign_total)) * 100

    threat_total = sum(cm["alert"].values()) + sum(cm["suspicious"].values())
    threat_caught = (cm["alert"]["alert"] + cm["alert"]["suspicious"] +
                     cm["suspicious"]["alert"] + cm["suspicious"]["suspicious"])
    recall_rate = (threat_caught / max(1, threat_total)) * 100

    mitre_acc = (metrics["mitre_matches"] / max(1, metrics["mitre_evaluable"])) * 100
    cot_rate = (metrics["cot_compliance"] / max(1, total_eval)) * 100
    json_rate = (metrics["json_compliance"] / max(1, total_eval)) * 100
    avg_latency = sum(metrics["latencies"]) / max(1, len(metrics["latencies"]))

    summary = {
        "model_name": model_name,
        "total_samples": total_eval,
        "confusion_matrix": cm,
        "threat_recall_pct": round(recall_rate, 2),
        "false_positive_rate_pct": round(fp_rate, 2),
        "mitre_accuracy_pct": round(mitre_acc, 2),
        "cot_compliance_pct": round(cot_rate, 2),
        "json_compliance_pct": round(json_rate, 2),
        "avg_latency_sec": round(avg_latency, 3),
        "details": results_log
    }

    print_scorecard(summary)
    return summary


def print_scorecard(summary: Dict[str, Any]):
    cm = summary["confusion_matrix"]
    print("\n" + "=" * 68)
    print(f"📊 SOC Triage Scorecard: {summary['model_name']}")
    print("=" * 68)
    print("Confusion Matrix:")
    print("                  Predicted Alert | Predicted Susp | Predicted Clear | Invalid")
    print(f"  Actual Alert        : {cm['alert']['alert']:<15} | {cm['alert']['suspicious']:<14} | {cm['alert']['clear']:<15} | {cm['alert']['invalid']}")
    print(f"  Actual Suspicious   : {cm['suspicious']['alert']:<15} | {cm['suspicious']['suspicious']:<14} | {cm['suspicious']['clear']:<15} | {cm['suspicious']['invalid']}")
    print(f"  Actual Clear        : {cm['clear']['alert']:<15} | {cm['clear']['suspicious']:<14} | {cm['clear']['clear']:<15} | {cm['clear']['invalid']}")
    print("-" * 68)
    print(f"  🎯 Threat Recall (Catch Rate)  : {summary['threat_recall_pct']}% (Alert + Suspicious captured)")
    print(f"  🚨 False Positive Rate (Benign): {summary['false_positive_rate_pct']}% (Benign flagged as threats)")
    print(f"  🏷️  MITRE ATT&CK Mapping Acc.   : {summary['mitre_accuracy_pct']}%")
    print(f"  🧠 CoT <think> Compliance     : {summary['cot_compliance_pct']}%")
    print(f"  📋 Strict JSON Schema Validity : {summary['json_compliance_pct']}%")
    print(f"  ⏱️  Average Latency            : {summary['avg_latency_sec']}s / sample")
    print("=" * 68 + "\n")


def main():
    parser = argparse.ArgumentParser(description="CyberAgent SOC Model Benchmark & Evaluation")
    parser.add_argument("--model", "-m", default="cyberagent-dpo", help="Model name in Ollama (e.g. 'cyberagent-dpo' or 'qwen3.5:0.8b')")
    parser.add_argument("--eval-file", "-f", default=None, help="Path to evaluation JSON dataset (e.g. 'weights/.../eval_dataset.json')")
    parser.add_argument("--scenario", "-s", default=None, help="Path to scenario directory (e.g. 'data/logs_training_data/scenario4_auto')")
    parser.add_argument("--limit", "-l", type=int, default=None, help="Optional limit on number of samples to evaluate")
    parser.add_argument("--output-report", "-o", default=None, help="Optional path to save full evaluation JSON report")

    args = parser.parse_args()

    # Default fallback
    if not args.eval_file and not args.scenario:
        default_eval = "weights/cyberagent-dpo-qwen3.5-0.8b/eval_dataset.json"
        if os.path.exists(default_eval):
            args.eval_file = default_eval
        else:
            args.scenario = "data/logs_training_data/scenario4_auto"

    targets = load_evaluation_targets(eval_file=args.eval_file, scenario_path=args.scenario)
    summary = run_benchmark(targets=targets, model_name=args.model, limit=args.limit)

    if args.output_report:
        with open(args.output_report, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info("Report saved to '%s'", args.output_report)


if __name__ == "__main__":
    main()
