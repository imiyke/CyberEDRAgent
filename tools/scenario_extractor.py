#!/usr/bin/env python3
"""
Generic Scenario Log Extractor & Ground Truth Generator
Parses multi-host simulation scenarios (attackmate.json + host system logs),
extracts structured log chunks into 3 classes (offensive/alert, suspicious, benign/clear),
guarantees strict anti-overlap filtering, and exports ground_truth.json.
"""

import os
import re
import json
import shlex
import argparse
from datetime import datetime, timezone
from collections import defaultdict


# ==============================================================================
# 1. Suspicious Patterns & Anti-Overlap Registry
# ==============================================================================

# Keywords and regex patterns for ambiguous / suspicious administrative activities
SUSPICIOUS_PATTERNS = [
    # 1. Privileged / remote session creations
    (r"Accepted (?:publickey|password) for root from", "T1078", "Valid Accounts: Root SSH Login", "privileged_remote_ssh_login"),
    (r"sudo:\s+pam_unix\(sudo:session\):\s+session opened for user root", "T1078", "Valid Accounts: Privilege Elevation", "interactive_privilege_elevation"),
    (r"Connection closed by [0-9.]+ port \d+ \[preauth\]", None, None, "ssh_unauthenticated_connection_probe"),
    
    # 2. Account & Policy Discovery
    (r"chage\s+-l", "T1201", "Password Policy Discovery", "password_policy_enumeration"),
    (r"cat\s+/etc/pam\.d/common-password", "T1201", "Password Policy Discovery", "pam_configuration_read"),
    (r"cat\s+/etc/passwd", "T1087.001", "Account Discovery: Local Account", "local_user_enumeration"),
    (r"T1201_Password_Policy_Discovery", "T1201", "Password Policy Discovery", "password_policy_discovery"),
    (r"T1087_Account_Discovery", "T1087", "Account Discovery", "account_discovery"),

    # 3. Bulk log collections & Local staging
    (r"rsync\s+--server\s+--sender.*\/var\/log", "T1005", "Data from Local System", "bulk_log_collection"),
    (r"gcore\s+-o", "T1003", "OS Credential Dumping / Memory Extraction", "process_core_dumping"),
    
    # 4. Anti-Forensics / File deletion in temporary dirs
    (r"T1070_Indicator_Removal_on_Host", "T1070.004", "Indicator Removal: File Deletion", "temporary_artifact_deletion"),
    (r"unlinkat.*\/tmp\/", "T1070.004", "Indicator Removal: File Deletion", "temporary_file_unlinking"),
    
    # 5. Service Restarts & Daemon Anomalies
    (r"Restarting capture daemon for", None, None, "service_watchdog_restart_anomaly"),
    (r"SURICATA STREAM pkt seen on wrong thread", None, None, "network_stream_sequence_anomaly"),
]

# Strict token exclusion list for the Benign filter:
# Any candidate benign log containing these terms is REJECTED from the benign dataset.
BENIGN_EXCLUSION_KEYWORDS = [
    # Explicit attacks / tools
    "zmcontroller", "superimplant", "selinux.so.3", "ld.so.preload", "rootjail",
    "r.tar.gz", "glibc_rootjail", "faaacebook.com", "nmap", "/dev/tcp", "reverse shell",
    # Suspicious primitives
    "chage", "common-password", "passwd", "shadow", "gcore", "rsync --server",
    "Accepted publickey for root", "Accepted password for root", "[preauth]",
    "unlinkat", "T1070", "T1201", "T1087", "T1005", "T1046", "T1560", "T1547",
    "SURICATA STREAM"
]


# ==============================================================================
# 2. Auditd Parser & Hex Decoder
# ==============================================================================

def decode_hex_string(hex_str: str) -> str:
    """Decodes hex-encoded auditd strings (e.g. 2F746D70... -> /tmp/...)."""
    if len(hex_str) > 2 and len(hex_str) % 2 == 0 and re.fullmatch(r"[0-9A-Fa-f]+", hex_str):
        try:
            decoded = bytes.fromhex(hex_str).decode("utf-8", errors="ignore")
            if any(c.isprintable() for c in decoded):
                return decoded.replace("\x00", " ")
        except Exception:
            pass
    return hex_str


def parse_auditd_blocks(audit_log_path: str):
    """
    Groups multi-line auditd records by audit event ID: audit(timestamp.msec:event_id).
    Returns list of event dictionaries: { 'id': ..., 'timestamp': ..., 'lines': [...], 'raw_text': ... }
    """
    if not os.path.exists(audit_log_path):
        return []

    events = defaultdict(list)
    event_order = []

    with open(audit_log_path, "r", errors="ignore") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            # Match audit(1759315501.465:5668)
            m = re.search(r"audit\((\d+\.\d+):(\d+)\)", line_str)
            if m:
                event_key = f"{m.group(1)}:{m.group(2)}"
                if event_key not in events:
                    event_order.append(event_key)
                events[event_key].append(line_str)

    parsed_events = []
    for k in event_order:
        lines = events[k]
        epoch_str = k.split(":")[0]
        epoch = float(epoch_str)
        
        # Combine lines and decode hex proctitle/execve
        decoded_lines = []
        for l in lines:
            if "proctitle=" in l:
                parts = l.split("proctitle=")
                raw_hex = parts[1].strip().strip('"')
                decoded = decode_hex_string(raw_hex)
                decoded_lines.append(f'{parts[0]}proctitle="{decoded}"')
            elif "a1=" in l or "cmd=" in l:
                decoded_l = re.sub(
                    r'(a\d+|cmd)=([0-9A-Fa-f]{4,})',
                    lambda match: f'{match.group(1)}="{decode_hex_string(match.group(2))}"',
                    l
                )
                decoded_lines.append(decoded_l)
            else:
                decoded_lines.append(l)

        parsed_events.append({
            "id": k,
            "event_id": k,
            "epoch": epoch,
            "datetime": datetime.fromtimestamp(epoch, tz=timezone.utc),
            "lines": decoded_lines,
            "raw_text": "\n".join(decoded_lines)
        })

    return parsed_events


# ==============================================================================
# 3. Attackmate Parser
# ==============================================================================

class AttackmateParser:
    """Parses attackmate.json to extract the offensive execution manifest."""

    def __init__(self, attackmate_path: str):
        self.attackmate_path = attackmate_path
        self.commands = []
        self.offensive_keywords = set()
        self.attack_start_time = None
        self.attack_end_time = None
        self._load()

    def _load(self):
        if not os.path.exists(self.attackmate_path):
            raise FileNotFoundError(f"Attackmate file not found: {self.attackmate_path}")

        with open(self.attackmate_path, "r", errors="ignore") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    entry = json.loads(line_str)
                    dt_str = entry.get("start-datetime")
                    if dt_str:
                        dt = datetime.fromisoformat(dt_str)
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        entry["dt"] = dt
                        if self.attack_start_time is None or dt < self.attack_start_time:
                            self.attack_start_time = dt
                        if self.attack_end_time is None or dt > self.attack_end_time:
                            self.attack_end_time = dt
                    self.commands.append(entry)

                    # Extract offensive keywords (ignoring common system commands & dictionary words)
                    STOPWORDS = {
                        "sudo", "shell", "true", "false", "null", "root", "execute", "download", "upload",
                        "service", "start", "stop", "restart", "system", "status", "cat", "tar", "grep",
                        "echo", "mkdir", "rm", "mv", "cp", "sed", "awk", "find", "chmod", "chown", "ps",
                        "kill", "netstat", "ip", "ifconfig", "hostname", "whoami", "id", "which", "whereis",
                        "ls", "dir", "cd", "etc", "bin", "usr", "var", "tmp", "run", "log", "lib", "opt",
                        "dev", "sys", "proc", "mount", "unmount", "user", "group", "password", "pam", "login",
                        "ssh", "bash", "sh", "init", "socket", "target", "default", "clean", "deactivated",
                        "reached", "listening", "succeeded", "stopping", "stopped", "started", "mounting", "session"
                    }
                    cmd = entry.get("cmd", "")
                    params = entry.get("parameters") or {}
                    for field in [params.get("remote_path"), params.get("name"), params.get("session")]:
                        if field and isinstance(field, str):
                            for token in re.findall(r"[A-Za-z0-9_\-\.]{4,}", field):
                                t_low = token.lower()
                                if t_low not in STOPWORDS and not t_low.startswith((".", "-")):
                                    self.offensive_keywords.add(t_low)

                except Exception as e:
                    pass

        print(f"[*] Loaded {len(self.commands)} commands from Attackmate.")
        print(f"[*] Attack Window: {self.attack_start_time} to {self.attack_end_time}")
        print(f"[*] Extracted {len(self.offensive_keywords)} offensive keyword tokens: {sorted(list(self.offensive_keywords))[:10]}...")


# ==============================================================================
# 4. Scenario Extractor Engine
# ==============================================================================

class ScenarioExtractor:
    """Generic multi-host scenario extractor with anti-overlap guarantees."""

    def __init__(self, scenario_dir: str, output_dir: str):
        self.scenario_dir = os.path.abspath(scenario_dir)
        self.scenario_name = os.path.basename(self.scenario_dir.rstrip("/"))
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        attackmate_json = os.path.join(self.scenario_dir, "attacker", "logs", "attackmate.json")
        self.parser = AttackmateParser(attackmate_json)

        self.offensive_chunks = []
        self.suspicious_chunks = []
        self.benign_chunks = []
        self.ground_truth = {}

    def _get_host_log_path(self, host: str, rel_path: str) -> str:
        return os.path.join(self.scenario_dir, host, "logs", "log", rel_path)

    def extract_offensive(self):
        """Extracts confirmed offensive chunks (alert) corresponding to attack actions."""
        print("[*] Extracting Offensive (Alert) chunks...")
        
        # We search primary victim host (videoserver or host found in logs)
        victim_hosts = [d for d in os.listdir(self.scenario_dir) if d not in ["attacker", "wazuh"] and os.path.isdir(os.path.join(self.scenario_dir, d))]
        
        seen_event_ids = set()
        chunk_idx = 1
        for cmd_entry in self.parser.commands:
            meta = (cmd_entry.get("parameters") or {}).get("metadata")
            if not meta:
                continue

            technique_id = meta.get("techniques")
            technique_name = meta.get("technique_name") or meta.get("tactics", "Offensive Action")
            cmd_str = (cmd_entry.get("cmd") or "").strip()
            params = cmd_entry.get("parameters") or {}
            
            # Filter out dual-use/pure discovery, interactive editor keystrokes, and shell navigation
            INTERACTIVE_IGNORE = [
                "hostname", "id", "whoami", "ifconfig", "netstat", "ps", "ls", "03", "sleep", "cd",
                ":wq", ":w", ":q", ":q!", "o", "i", "a", "1B", "clear", "reset"
            ]
            if cmd_str in INTERACTIVE_IGNORE or cmd_str.startswith((":%s", ":g/", "/invoke")):
                continue

            req_exe = params.get("exe")
            req_args = params.get("args") or []
            if not req_exe and cmd_str:
                try:
                    parts = [p.strip() for p in (shlex.split(cmd_str) if "\n" not in cmd_str else cmd_str.split()) if p.strip()]
                    if parts:
                        first_tok = os.path.basename(parts[0])
                        if first_tok not in ["sudo", "sh", "bash"]:
                            req_exe = first_tok
                            if not req_args:
                                req_args = parts[1:]
                        elif len(parts) > 1:
                            req_exe = os.path.basename(parts[1])
                            if not req_args:
                                req_args = parts[2:]
                except Exception:
                    pass

            req_path = params.get("remote_path")
            cmd_dt = cmd_entry.get("dt")

            GENERIC_VERBS = {
                "stop", "start", "restart", "status", "enable", "disable", "reload", "update", "upgrade",
                "install", "remove", "purge", "add", "del", "delete", "create", "list", "show", "get", "set",
                "apply", "check", "test", "rows", "columns", "color", "auto", "no-block"
            }
            GENERIC_PATH_PREFIXES = ("/lib64/ld-linux", "/lib/x86_64", "/usr/lib", "/lib/", "/usr/share")
            GENERIC_EXECUTABLES = {
                "/bin/bash", "/bin/sh", "/bin/dash", "/usr/bin/bash", "/usr/bin/sh", "/usr/bin/dash",
                "/usr/bin/python3", "/usr/bin/python", "/usr/bin/perl", "/usr/sbin/sshd", "/usr/bin/sudo"
            }

            # Identify candidate log lines across victim hosts
            found_chunk = None
            for host in victim_hosts:
                # 1. Check auditd parsed events
                audit_path = self._get_host_log_path(host, "audit/audit.log")
                if os.path.exists(audit_path):
                    audit_events = parse_auditd_blocks(audit_path)
                    
                    for ev in audit_events:
                        if ev["id"] in seen_event_ids:
                            continue

                        raw = ev["raw_text"]
                        
                        # Timestamp proximity check
                        if cmd_dt:
                            time_diff = abs((ev["datetime"] - cmd_dt).total_seconds())
                            max_window = 3600 if ("sed" in cmd_str or "cron" in cmd_str or "service" in cmd_str) else 180
                            if time_diff > max_window:
                                continue

                        # Exclude routine daemons / monitoring agents from standalone offensive matches
                        is_daemon = any(d in raw for d in [
                            'comm="cron"', 'comm="CRON"', 'comm="systemd"', 'comm="e2scrub', 'GID="wazuh"',
                            'comm="wazuh-agent"', 'comm="logrotate"', 'exe="/usr/sbin/cron"', 'comm="sshd"'
                        ])

                        matched = False
                        
                        # 1. Explicit executable match with word boundaries and argument verification
                        if req_exe and not is_daemon:
                            exe_pat = r'(?:comm="?' + re.escape(req_exe) + r'"?|exe="?[^"\s]*/' + re.escape(req_exe) + r'"?)'
                            if re.search(exe_pat, raw):
                                if req_args:
                                    meaningful_args = [a for a in req_args if len(a) > 2 and not a.startswith("-") and a.lower() not in GENERIC_VERBS]
                                    if meaningful_args:
                                        if any(a in raw for a in meaningful_args):
                                            matched = True
                                    else:
                                        # When only generic verbs exist (e.g. "apt update"), verify exe and interactive or exact match
                                        if any(a in raw for a in req_args if len(a) > 2 and not a.startswith("-")):
                                            matched = True
                                else:
                                    matched = True

                        # 2. File deletion / removal match
                        if not matched and cmd_str in ["rm", "delete"]:
                            if req_path and os.path.basename(req_path) in raw:
                                if "nametype=DELETE" in raw or "SYSCALL=unlink" in raw or "unlinkat" in raw:
                                    matched = True

                        # 3. Specific remote file transfer/execution (download, upload)
                        if not matched and req_path and not req_exe:
                            base_p = os.path.basename(req_path)
                            if base_p in raw:
                                req_session = params.get("session")
                                if req_session and req_session in raw:
                                    matched = True
                                elif any(p in raw for p in ['comm="sftp-server"', 'comm="scp"', 'comm="wget"', 'comm="curl"']):
                                    matched = True

                        # 4. Command strings / Persistence / Script modifications (dynamic full path tokens)
                        if not matched and cmd_str and not is_daemon and not req_exe:
                            cmd_paths = re.findall(r'/[A-Za-z0-9_\.\-]+(?:/[A-Za-z0-9_\.\-]+)+', cmd_str)
                            clean_paths = [p for p in cmd_paths if p not in GENERIC_EXECUTABLES and not any(p.startswith(gp) for gp in GENERIC_PATH_PREFIXES)]
                            if clean_paths and any(p in raw for p in clean_paths):
                                matched = True

                        if matched:
                            seen_event_ids.add(ev["id"])
                            found_chunk = {
                                "host": host,
                                "log_type": "auditd",
                                "raw_logs": raw,
                                "technique_id": technique_id,
                                "technique_name": technique_name,
                                "cmd": cmd_str,
                                "entities": [req_exe] if req_exe else ([req_path] if req_path else [cmd_str])
                            }
                            break

                if found_chunk:
                    break

            if found_chunk:
                chunk_id = f"{self.scenario_name}_off_{chunk_idx:03d}"
                self.offensive_chunks.append({
                    "id": chunk_id,
                    "host": found_chunk["host"],
                    "log_type": found_chunk["log_type"],
                    "raw_logs": found_chunk["raw_logs"],
                    "technique_id": found_chunk["technique_id"],
                    "technique_name": found_chunk["technique_name"],
                    "intent": f"offensive_{found_chunk['technique_name'].lower().replace(' ', '_')[:30]}",
                    "summary": f"Observed offensive execution for {found_chunk['cmd']} ({found_chunk['technique_id']})."
                })
                chunk_idx += 1

        print(f"[+] Extracted {len(self.offensive_chunks)} Offensive chunks.")

    def extract_suspicious(self, max_per_intent: int = 5):
        """Extracts contextual / ambiguous logs matching the Suspicious Pattern Registry."""
        print(f"[*] Extracting Suspicious chunks (max {max_per_intent} per intent)...")
        hosts = [d for d in os.listdir(self.scenario_dir) if os.path.isdir(os.path.join(self.scenario_dir, d)) and d != "wazuh"]

        chunk_idx = 1
        seen_texts = set()
        intent_counts = defaultdict(int)

        for host in hosts:
            for log_rel in ["auth.log", "syslog", "suricata/fast.log", "audit/audit.log"]:
                log_path = self._get_host_log_path(host, log_rel)
                if not os.path.exists(log_path):
                    continue

                if log_rel.endswith("audit.log"):
                    # Use parsed audit events
                    audit_events = parse_auditd_blocks(log_path)
                    for ev in audit_events:
                        raw = ev["raw_text"]
                        for pattern, mitre_id, tech_name, intent in SUSPICIOUS_PATTERNS:
                            if intent_counts[intent] >= max_per_intent:
                                continue
                            if re.search(pattern, raw, re.IGNORECASE):
                                if raw not in seen_texts:
                                    seen_texts.add(raw)
                                    intent_counts[intent] += 1
                                    chunk_id = f"{self.scenario_name}_susp_{chunk_idx:03d}"
                                    self.suspicious_chunks.append({
                                        "id": chunk_id,
                                        "host": host,
                                        "log_type": "auditd",
                                        "raw_logs": raw,
                                        "technique_id": mitre_id,
                                        "technique_name": tech_name,
                                        "intent": intent,
                                        "summary": f"Suspicious activity detected on {host}: {intent} ({mitre_id or 'System Anomaly'})."
                                    })
                                    chunk_idx += 1
                                break
                else:
                    # Line-based logs (auth.log, syslog, fast.log)
                    with open(log_path, "r", errors="ignore") as f:
                        lines = f.readlines()
                    
                    i = 0
                    while i < len(lines):
                        line_str = lines[i].strip()
                        for pattern, mitre_id, tech_name, intent in SUSPICIOUS_PATTERNS:
                            if intent_counts[intent] >= max_per_intent:
                                continue
                            if re.search(pattern, line_str, re.IGNORECASE):
                                start = max(0, i - 1)
                                end = min(len(lines), i + 4)
                                chunk_text = "".join(lines[start:end]).strip()
                                
                                if chunk_text not in seen_texts:
                                    seen_texts.add(chunk_text)
                                    intent_counts[intent] += 1
                                    chunk_id = f"{self.scenario_name}_susp_{chunk_idx:03d}"
                                    self.suspicious_chunks.append({
                                        "id": chunk_id,
                                        "host": host,
                                        "log_type": os.path.basename(log_rel),
                                        "raw_logs": chunk_text,
                                        "technique_id": mitre_id,
                                        "technique_name": tech_name,
                                        "intent": intent,
                                        "summary": f"Suspicious activity on {host}: {intent} ({mitre_id or 'System Anomaly'})."
                                    })
                                    chunk_idx += 1
                                i = end
                                break
                        i += 1

        print(f"[+] Extracted {len(self.suspicious_chunks)} Suspicious chunks across {len(intent_counts)} intents.")

    def extract_benign(self, max_per_intent: int = 5):
        """
        Extracts routine benign logs from baseline hosts & pre-attack periods.
        Applies STRICT ANTI-OVERLAP negative filtering.
        """
        print(f"[*] Extracting Benign (Clear) chunks with Anti-Overlap Filter (max {max_per_intent} per intent)...")
        hosts = [d for d in os.listdir(self.scenario_dir) if os.path.isdir(os.path.join(self.scenario_dir, d)) and d != "wazuh"]

        chunk_idx = 1
        seen_texts = set()
        intent_counts = defaultdict(int)

        # Enriched routine benign signals
        BENIGN_SIGNALS = [
            (r"CRON\[\d+\]:\s+pam_unix\(cron:session\):\s+session opened for user root", "routine_cron_session_maintenance"),
            (r"CRON\[\d+\]:\s+\(root\)\s+CMD\s+\(.*?popularity-contest", "routine_popularity_contest_cron"),
            (r"CRON\[\d+\]:\s+\(root\)\s+CMD\s+\(.*?e2scrub", "routine_e2scrub_cron"),
            (r"cloud-final\.service:\s+Deactivated successfully", "cloud_init_boot_routine"),
            (r"ET POLICY GNU/Linux APT User-Agent", "routine_package_manager_network_check"),
            (r"systemd\[\d+\]:\s+Stopped User Manager for UID", "systemd_user_session_cleanup"),
            (r"Linux version \d+\.\d+", "kernel_boot_hardware_telemetry"),
            (r"unattended-upgrade.*No packages found that can be upgraded", "automated_patch_check"),
            (r"systemd-fsck.*clean", "filesystem_integrity_check"),
            (r"systemd\[1\]:\s+Started\s+OpenSSH server daemon", "sshd_service_startup"),
            (r"systemd\[1\]:\s+Started\s+Daily apt download activities", "systemd_timer_apt_activity"),
            (r"systemd\[1\]:\s+Started\s+Periodic Command Scheduler", "systemd_cron_service_start"),
            (r"systemd\[1\]:\s+Finished\s+Clean up any mess left by 0dns-up", "systemd_network_cleanup_finish"),
            (r"systemd\[1\]:\s+Started\s+Collectd statistics daemon", "collectd_daemon_startup"),
            (r"systemd-logind\[\d+\]:\s+Watching system buttons", "systemd_logind_service_idle"),
            (r"systemd\[1\]:\s+Reached target Network is Online", "systemd_target_network_online"),
            (r"systemd\[1\]:\s+Reached target Basic System", "systemd_target_basic_system"),
            (r"systemd\[1\]:\s+Listening on D-Bus System Message Bus Socket", "systemd_dbus_socket_listening"),
            (r"systemd\[1\]:\s+Created slice User Slice of UID \d+", "systemd_user_slice_created"),
            (r"systemd\[1\]:\s+Started Session \d+ of User", "systemd_user_session_started"),
            (r"dbus-daemon\[\d+\]:\s+\[system\] Successfully activated service", "dbus_service_activation"),
            (r"systemd-timesyncd\[\d+\]:\s+Synchronized to time server", "ntp_time_sync"),
            (r"sshd\[\d+\]:\s+pam_unix\(sshd:session\):\s+session closed for user", "sshd_normal_session_close"),
            (r"kernel:\s+\[.*?\]\s+EXT4-fs.*mounted filesystem", "kernel_filesystem_mount"),
            (r"rsyslogd:.*action 'action-.*' resumed", "rsyslog_action_resumed"),
            (r"systemd\[1\]:\s+Mounted Mount unit for lxd", "systemd_lxd_mount"),
            (r"systemd\[1\]:\s+Mounted FUSE Control File System", "systemd_fuse_mount"),
            (r"systemd\[1\]:\s+Started\s+Dispatch Password Requests to Console", "systemd_password_dispatch"),
            (r"systemd\[1\]:\s+Reached target Local Encrypted Volumes", "systemd_cryptsetup_target")
        ]

        for host in hosts:
            for log_rel in ["syslog", "auth.log", "suricata/fast.log"]:
                log_path = self._get_host_log_path(host, log_rel)
                if not os.path.exists(log_path):
                    continue

                with open(log_path, "r", errors="ignore") as f:
                    lines = f.readlines()

                i = 0
                while i < len(lines):
                    line_str = lines[i].strip()
                    for pattern, intent in BENIGN_SIGNALS:
                        if intent_counts[intent] >= max_per_intent:
                            continue
                        if re.search(pattern, line_str, re.IGNORECASE):
                            start = max(0, i - 1)
                            end = min(len(lines), i + 4)
                            chunk_text = "".join(lines[start:end]).strip()

                            # STRICT ANTI-OVERLAP FILTER:
                            # 1. Must not match any offensive token (word boundary)
                            # 2. Must not match any suspicious keyword
                            has_offensive_leak = any(
                                re.search(r'\b' + re.escape(k) + r'\b', chunk_text, re.IGNORECASE)
                                for k in self.parser.offensive_keywords
                            )
                            has_suspicious_leak = any(
                                re.search(r'\b' + re.escape(k) + r'\b', chunk_text, re.IGNORECASE)
                                for k in BENIGN_EXCLUSION_KEYWORDS
                            )

                            if not has_offensive_leak and not has_suspicious_leak:
                                if chunk_text not in seen_texts:
                                    seen_texts.add(chunk_text)
                                    intent_counts[intent] += 1
                                    chunk_id = f"{self.scenario_name}_benign_{chunk_idx:03d}"
                                    self.benign_chunks.append({
                                        "id": chunk_id,
                                        "host": host,
                                        "log_type": os.path.basename(log_rel),
                                        "raw_logs": chunk_text,
                                        "technique_id": None,
                                        "technique_name": None,
                                        "intent": intent,
                                        "summary": f"Routine benign system maintenance on {host}: {intent}."
                                    })
                                    chunk_idx += 1
                            i = end
                            break
                    i += 1

        print(f"[+] Extracted {len(self.benign_chunks)} Benign chunks (Passed Anti-Overlap Filter).")

    def export(self):
        """Exports the 3 .log files and ground_truth.json."""
        ground_truth = {}

        # 1. Offensive File
        off_file = f"{self.scenario_name}_offensive.log"
        off_path = os.path.join(self.output_dir, off_file)
        with open(off_path, "w", encoding="utf-8") as f:
            f.write(f"# Auto-extracted Offensive Logs: {off_file}\n# Total chunks: {len(self.offensive_chunks)}\n\n")
            for ch in self.offensive_chunks:
                f.write(f"# --- START CHUNK: {ch['id']} (ALERT) ---\n{ch['raw_logs']}\n# --- END CHUNK: {ch['id']} ---\n\n")
                ground_truth[ch["id"]] = {
                    "log_file": off_file,
                    "source_host": ch["host"],
                    "log_type": ch["log_type"],
                    "status": "alert",
                    "category": "security",
                    "mitre_id": ch["technique_id"],
                    "mitre_technique_name": ch["technique_name"],
                    "observed_intent": ch["intent"],
                    "confidence": "high",
                    "summary": ch["summary"]
                }
        print(f"[*] Exported {off_path} ({len(self.offensive_chunks)} chunks)")

        # 2. Suspicious File
        susp_file = f"{self.scenario_name}_suspicious.log"
        susp_path = os.path.join(self.output_dir, susp_file)
        with open(susp_path, "w", encoding="utf-8") as f:
            f.write(f"# Auto-extracted Suspicious Logs: {susp_file}\n# Total chunks: {len(self.suspicious_chunks)}\n\n")
            for ch in self.suspicious_chunks:
                f.write(f"# --- START CHUNK: {ch['id']} (SUSPICIOUS) ---\n{ch['raw_logs']}\n# --- END CHUNK: {ch['id']} ---\n\n")
                ground_truth[ch["id"]] = {
                    "log_file": susp_file,
                    "source_host": ch["host"],
                    "log_type": ch["log_type"],
                    "status": "suspicious",
                    "category": "system_warning" if ch["technique_id"] is None else "security",
                    "mitre_id": ch["technique_id"],
                    "mitre_technique_name": ch["technique_name"],
                    "observed_intent": ch["intent"],
                    "confidence": "medium",
                    "summary": ch["summary"]
                }
        print(f"[*] Exported {susp_path} ({len(self.suspicious_chunks)} chunks)")

        # 3. Benign File
        benign_file = f"{self.scenario_name}_benign.log"
        benign_path = os.path.join(self.output_dir, benign_file)
        with open(benign_path, "w", encoding="utf-8") as f:
            f.write(f"# Auto-extracted Benign Logs: {benign_file}\n# Total chunks: {len(self.benign_chunks)}\n\n")
            for ch in self.benign_chunks:
                f.write(f"# --- START CHUNK: {ch['id']} (CLEAR) ---\n{ch['raw_logs']}\n# --- END CHUNK: {ch['id']} ---\n\n")
                ground_truth[ch["id"]] = {
                    "log_file": benign_file,
                    "source_host": ch["host"],
                    "log_type": ch["log_type"],
                    "status": "clear",
                    "category": "noise",
                    "mitre_id": None,
                    "mitre_technique_name": None,
                    "observed_intent": ch["intent"],
                    "confidence": "high",
                    "summary": ch["summary"]
                }
        print(f"[*] Exported {benign_path} ({len(self.benign_chunks)} chunks)")

        # 4. Ground Truth JSON
        gt_path = os.path.join(self.output_dir, "ground_truth.json")
        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump(ground_truth, f, indent=2)
        print(f"[*] Exported {gt_path} ({len(ground_truth)} total entries)")


# ==============================================================================
# 5. CLI Entrypoint
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Auto Scenario Log Extractor & Ground Truth Builder")
    parser.add_argument("--scenario_dir", required=True, help="Path to scenario directory (e.g. data/logs_training_data/scenario_2_cron)")
    parser.add_argument("--output_dir", required=True, help="Output directory for curated files")
    parser.add_argument("--max_per_intent", type=int, default=5, help="Maximum chunks per unique intent pattern (default: 5)")
    args = parser.parse_args()

    extractor = ScenarioExtractor(args.scenario_dir, args.output_dir)
    extractor.extract_offensive()
    extractor.extract_suspicious(max_per_intent=args.max_per_intent)
    extractor.extract_benign(max_per_intent=args.max_per_intent)
    extractor.export()


if __name__ == "__main__":
    main()
