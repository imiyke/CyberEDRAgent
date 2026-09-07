#!/usr/bin/env python3
"""
Scenario 2 Log Extractor & Ground Truth Generator
Extracts raw log chunks from scenario_2_cron for Offensive (Alert), Suspicious, and Benign (Clear) classes.
Generates scenario2_cron_{offensive,suspicious,benign}.log and ground_truth.json.
"""

import os
import json

BASE_DIR = "data/logs_training_data/scenario_2_cron"
OUTPUT_DIR = "data/logs_training_data/scenario2_cron_curated"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Definition of the Curated Dataset chunks
DATASET_ENTRIES = {
    # ==========================================
    # 1. OFFENSIVE LOGS (status: alert)
    # ==========================================
    "scenario2_off_001_cron_trojan": {
        "file": "scenario2_cron_offensive.log",
        "source_host": "videoserver",
        "log_type": "auditd / syslog",
        "status": "alert",
        "category": "security",
        "mitre_id": "T1547.001",
        "mitre_technique_name": "Boot or Logon Autostart Execution: Cron",
        "target_entities": ["/usr/share/awffull/awffull", "/usr/bin/zmcontroller", "cron", "pid=3216"],
        "observed_intent": "persistence_script_trojanization",
        "confidence": "high",
        "summary": "Scheduled cron execution of trojanized awffull script spawning the unauthorized /usr/bin/zmcontroller binary in background.",
        "raw_logs": """Oct  1 10:30:01 videoserver CRON[2974]: (root) CMD (/usr/share/awffull/awffull)
type=SYSCALL msg=audit(1759315501.465:5668): arch=c000003e syscall=59 success=yes exit=0 a0=55c64b0b7868 a1=55c64b0b78a0 a2=55c64b0b78b0 a3=8 items=4 ppid=3214 pid=3216 auid=0 uid=0 gid=0 euid=0 tty=(none) ses=16 comm="sh" exe="/bin/dash"
type=EXECVE msg=audit(1759315501.465:5668): argc=2 a0="/bin/sh" a1="/usr/share/awffull/awffull"
type=PATH msg=audit(1759315501.465:5668): item=0 name="/usr/share/awffull/awffull" inode=531981 dev=fe:01 mode=0100755 ouid=33 ogid=0
type=EXECVE msg=audit(1759315501.469:5670): argc=1 a0="/usr/bin/zmcontroller"
type=PATH msg=audit(1759315501.469:5670): item=0 name="/usr/bin/zmcontroller" inode=30329 dev=fe:01 mode=0100755 ouid=0 ogid=0
type=PROCTITLE msg=audit(1759315501.469:5670): proctitle="/usr/bin/zmcontroller" """
    },

    "scenario2_off_002_implant_c2_beacon": {
        "file": "scenario2_cron_offensive.log",
        "source_host": "videoserver",
        "log_type": "auditd",
        "status": "alert",
        "category": "security",
        "mitre_id": "T1071.001",
        "mitre_technique_name": "Application Layer Protocol: Web Protocols",
        "target_entities": ["/usr/bin/zmcontroller", "pid=3217", "syscall=42", "c000092bac"],
        "observed_intent": "c2_network_beaconing",
        "confidence": "high",
        "summary": "Implant zmcontroller invoking socket connect (syscall 42) to establish persistent C2 communications.",
        "raw_logs": """type=SYSCALL msg=audit(1759315501.485:5673): arch=c000003e syscall=42 success=yes exit=0 a0=3 a1=c000092bac a2=10 a3=0 items=0 ppid=1 pid=3217 auid=0 uid=0 gid=0 euid=0 comm="zmcontroller" exe="/usr/bin/zmcontroller"
type=PROCTITLE msg=audit(1759315501.485:5673): proctitle="/usr/bin/zmcontroller"
type=SYSCALL msg=audit(1759315501.485:5674): arch=c000003e syscall=42 success=yes exit=0 a0=7 a1=c000092bcc a2=10 a3=0 items=0 ppid=1 pid=3217 auid=0 uid=0 gid=0 euid=0 comm="zmcontroller" exe="/usr/bin/zmcontroller"
type=PROCTITLE msg=audit(1759315501.485:5674): proctitle="/usr/bin/zmcontroller"
type=SYSCALL msg=audit(1759315501.513:5680): arch=c000003e syscall=42 success=yes exit=0 a0=7 a1=c00009370c a2=10 a3=0 items=0 ppid=1 pid=3217 auid=0 uid=0 gid=0 euid=0 comm="zmcontroller" exe="/usr/bin/zmcontroller"
type=PROCTITLE msg=audit(1759315501.513:5680): proctitle="/usr/bin/zmcontroller" """
    },

    "scenario2_off_003_shadow_credential_access": {
        "file": "scenario2_cron_offensive.log",
        "source_host": "videoserver",
        "log_type": "auditd",
        "status": "alert",
        "category": "security",
        "mitre_id": "T1003.008",
        "mitre_technique_name": "OS Credential Dumping: /etc/passwd and /etc/shadow",
        "target_entities": ["/etc/shadow", "pid=3257", "syscall=257 (openat)", "root"],
        "observed_intent": "credential_theft_shadow_dump",
        "confidence": "high",
        "summary": "Direct access and exfiltration read attempt against /etc/shadow credential database.",
        "raw_logs": """type=SYSCALL msg=audit(1759315504.849:5753): arch=c000003e syscall=257 success=yes exit=5 a0=ffffff9c a1=7f58e995a176 a2=80000 a3=0 items=1 ppid=690 pid=3257 auid=4294967295 uid=0 gid=0 euid=0 suid=0 fsuid=0 egid=0 comm="sshd" exe="/usr/sbin/sshd" key="T1087_Account_Discovery"
type=PATH msg=audit(1759315504.849:5753): item=0 name="/etc/shadow" inode=30247 dev=fe:01 mode=0100640 ouid=0 ogid=42 rdev=00:00 nametype=NORMAL OUID="root" OGID="shadow"
type=SYSCALL msg=audit(1759315504.861:5763): arch=c000003e syscall=257 success=yes exit=5 a0=ffffff9c a1=7f58e995a176 a2=80000 a3=0 items=1 ppid=690 pid=3257 auid=1001 uid=0 gid=0 euid=0 suid=0 fsuid=0 egid=0 comm="sshd" exe="/usr/sbin/sshd" key="T1087_Account_Discovery"
type=PATH msg=audit(1759315504.861:5763): item=0 name="/etc/shadow" inode=30247 dev=fe:01 mode=0100640 ouid=0 ogid=42 rdev=00:00 nametype=NORMAL OUID="root" OGID="shadow" """
    },

    "scenario2_off_004_root_archive_staging": {
        "file": "scenario2_cron_offensive.log",
        "source_host": "videoserver",
        "log_type": "auditd",
        "status": "alert",
        "category": "security",
        "mitre_id": "T1560.001",
        "mitre_technique_name": "Archive Collected Data: Archive via Utility",
        "target_entities": ["/usr/bin/tar", "/tmp/r.tar.gz", "/root", "tar"],
        "observed_intent": "data_staging_and_compression",
        "confidence": "high",
        "summary": "Execution of tar archiving the entire /root directory into a temporary archive /tmp/r.tar.gz for exfiltration staging.",
        "raw_logs": """type=SYSCALL msg=audit(1759315538.107:5810): arch=c000003e syscall=59 success=yes exit=0 a0=55d21a08e120 a1=55d21a08e150 a2=55d21a08e170 a3=8 items=2 ppid=3217 pid=3280 auid=0 uid=0 gid=0 euid=0 comm="tar" exe="/usr/bin/tar" key="T1005_Data_from_Local_System"
type=EXECVE msg=audit(1759315538.107:5810): argc=4 a0="/usr/bin/tar" a1="cvzf" a2="/tmp/r.tar.gz" a3="/root"
type=PATH msg=audit(1759315538.107:5810): item=0 name="/usr/bin/tar" inode=30412 dev=fe:01 mode=0100755 ouid=0 ogid=0
type=PATH msg=audit(1759315538.107:5810): item=1 name="/tmp/r.tar.gz" inode=14022 dev=fe:01 mode=0100644 ouid=0 ogid=0
type=PROCTITLE msg=audit(1759315538.107:5810): proctitle=2F7573722F62696E2F7461720063767A66002F746D702F722E7461722E677A002F726F6F74 """
    },

    "scenario2_off_005_nmap_internal_recon": {
        "file": "scenario2_cron_offensive.log",
        "source_host": "videoserver",
        "log_type": "auditd",
        "status": "alert",
        "category": "security",
        "mitre_id": "T1046",
        "mitre_technique_name": "Network Service Discovery",
        "target_entities": ["/tmp/nmap", "172.17.100.0/24", "pid=2977", "chmod"],
        "observed_intent": "internal_network_scanning",
        "confidence": "high",
        "summary": "Deployment and execution of rogue /tmp/nmap binary scanning internal subnet 172.17.100.0/24.",
        "raw_logs": """type=SYSCALL msg=audit(1759314889.136:5275): arch=c000003e syscall=268 success=yes exit=0 a0=ffffff9c a1=c000355600 a2=1ed a3=0 items=1 ppid=1 pid=2977 auid=0 uid=0 gid=0 euid=0 comm="zmcontroller" exe="/usr/bin/zmcontroller" key="T1070_Indicator_Removal_on_Host"
type=PATH msg=audit(1759314889.136:5275): item=0 name="/tmp/nmap" inode=13988 dev=fe:01 mode=0100755 ouid=0 ogid=0
type=SYSCALL msg=audit(1759314904.237:5282): arch=c000003e syscall=59 success=yes exit=0 a0=55c64b0b7868 a1=55c64b0b78a0 a2=55c64b0b78b0 a3=8 items=2 ppid=2977 pid=3044 auid=0 uid=0 gid=0 euid=0 comm="nmap" exe="/tmp/nmap" key="T1046_Network_Service_Scanning"
type=EXECVE msg=audit(1759314904.237:5282): argc=2 a0="/tmp/nmap" a1="172.17.100.0/24"
type=PROCTITLE msg=audit(1759314904.237:5282): proctitle=2F746D702F6E6D6170003137322E31372E3130302E302F3234 """
    },

    "scenario2_off_006_sftp_ingress_implant": {
        "file": "scenario2_cron_offensive.log",
        "source_host": "videoserver",
        "log_type": "auth.log / auditd",
        "status": "alert",
        "category": "security",
        "mitre_id": "T1105",
        "mitre_technique_name": "Ingress Tool Transfer",
        "target_entities": ["root", "192.42.1.174", "/usr/bin/zmcontroller", "sshd"],
        "observed_intent": "remote_tool_drop_sftp",
        "confidence": "high",
        "summary": "External SSH login as root dropping the malicious payload into /usr/bin/zmcontroller.",
        "raw_logs": """Oct  1 10:14:15 videoserver sshd[2802]: Accepted publickey for root from 192.42.1.174 port 34856 ssh2: RSA SHA256:VbW9KNEmQ5+AUYJxP2VWEd9r7V
Oct  1 10:14:15 videoserver sshd[2802]: pam_unix(sshd:session): session opened for user root(uid=0) by (uid=0)
Oct  1 10:14:15 videoserver systemd-logind[557]: New session 6 of user root.
type=SYSCALL msg=audit(1759314931.002:5120): arch=c000003e syscall=257 success=yes exit=3 a0=ffffff9c a1=55c64b0c1230 a2=241 a3=1b6 items=2 ppid=2802 pid=2820 auid=0 uid=0 gid=0 euid=0 comm="sftp-server" exe="/usr/lib/openssh/sftp-server" key="T1105_remote_file_copy"
type=PATH msg=audit(1759314931.002:5120): item=0 name="/usr/bin/zmcontroller" inode=30329 dev=fe:01 mode=0100755 ouid=0 ogid=0 """
    },

    # ==========================================
    # 2. SUSPICIOUS LOGS (status: suspicious)
    # ==========================================
    "scenario2_susp_001_password_policy_discovery": {
        "file": "scenario2_cron_suspicious.log",
        "source_host": "videoserver",
        "log_type": "auditd",
        "status": "suspicious",
        "category": "security",
        "mitre_id": "T1201",
        "mitre_technique_name": "Password Policy Discovery",
        "target_entities": ["/etc/pam.d/common-password", "chage", "aecid", "cat"],
        "observed_intent": "account_password_policy_enumeration",
        "confidence": "medium",
        "summary": "User executing cat on common-password PAM config and querying account password expiration policy using chage.",
        "raw_logs": """type=SYSCALL msg=audit(1759315548.367:5825): arch=c000003e syscall=59 success=yes exit=0 a0=55d21a08e120 a1=55d21a08e150 a2=55d21a08e170 a3=8 items=2 ppid=3217 pid=3285 auid=0 uid=0 gid=0 euid=0 comm="cat" exe="/bin/cat" key="T1201_Password_Policy_Discovery"
type=EXECVE msg=audit(1759315548.367:5825): argc=2 a0="cat" a1="/etc/pam.d/common-password"
type=PATH msg=audit(1759315548.367:5825): item=0 name="/etc/pam.d/common-password" inode=30248 dev=fe:01 mode=0100644 ouid=0 ogid=0
type=SYSCALL msg=audit(1759315558.490:5830): arch=c000003e syscall=59 success=yes exit=0 a0=55d21a08e120 a1=55d21a08e150 a2=55d21a08e170 a3=8 items=2 ppid=3217 pid=3290 auid=0 uid=0 gid=0 euid=0 comm="chage" exe="/usr/bin/chage" key="T1201_Password_Policy_Discovery"
type=EXECVE msg=audit(1759315558.490:5830): argc=3 a0="chage" a1="-l" a2="aecid" """
    },

    "scenario2_susp_002_tmp_archive_deletion": {
        "file": "scenario2_cron_suspicious.log",
        "source_host": "videoserver",
        "log_type": "auditd",
        "status": "suspicious",
        "category": "security",
        "mitre_id": "T1070.004",
        "mitre_technique_name": "Indicator Removal: File Deletion",
        "target_entities": ["/tmp/r.tar.gz", "syscall=263 (unlinkat)", "pid=3217", "zmcontroller"],
        "observed_intent": "temporary_archive_cleanup_or_antiforensics",
        "confidence": "medium",
        "summary": "Process deleting temporary archive /tmp/r.tar.gz immediately after file creation.",
        "raw_logs": """type=SYSCALL msg=audit(1759315518.253:5798): arch=c000003e syscall=263 success=yes exit=0 a0=ffffff9c a1=c000355600 a2=0 a3=0 items=2 ppid=1 pid=3217 auid=0 uid=0 gid=0 euid=0 comm="zmcontroller" exe="/usr/bin/zmcontroller" key="T1070_Indicator_Removal_on_Host"
type=PATH msg=audit(1759315518.253:5798): item=0 name="/tmp" inode=14016 dev=fe:01 mode=041777 ouid=0 ogid=0
type=PATH msg=audit(1759315518.253:5798): item=1 name="/tmp/r.tar.gz" inode=14022 dev=fe:01 mode=0100644 ouid=0 ogid=0
type=PROCTITLE msg=audit(1759315518.253:5798): proctitle="/usr/bin/zmcontroller" """
    },

    "scenario2_susp_003_zoneminder_daemon_fail_restart": {
        "file": "scenario2_cron_suspicious.log",
        "source_host": "videoserver",
        "log_type": "syslog",
        "status": "suspicious",
        "category": "system_warning",
        "mitre_id": None,
        "mitre_technique_name": None,
        "target_entities": ["zmc_m1", "zmwatch", "rtsp://172.17.100.80:8554/mystream", "pid=3012"],
        "observed_intent": "service_instability_or_video_stream_interruption",
        "confidence": "low",
        "summary": "ZoneMinder watchdog repeatedly restarting video capture daemon due to 404 HTTP/RTSP error and lack of video frames.",
        "raw_logs": """Oct  1 10:34:21 videoserver zmc_m1[3012]: ERR [zmc_m1] [Unable to open input rtsp://172.17.100.80:8554/mystream due to: Server returned 404 Not Found]
Oct  1 10:34:21 videoserver zmc_m1[3012]: ERR [zmc_m1] [Failed to prime capture of initial monitor]
Oct  1 10:34:33 videoserver zmwatch[875]: WAR [Restarting capture daemon for 1 cam-1, no image since startup. Startup time was 1759314825 - now 1759314873 > 45]
Oct  1 10:34:34 videoserver zmc_m1[3012]: ERR [zmc_m1] [Unable to open input rtsp://172.17.100.80:8554/mystream due to: Immediate exit requested]
Oct  1 10:34:35 videoserver zmc_m1[3033]: ERR [zmc_m1] [Unable to open input rtsp://172.17.100.80:8554/mystream due to: Server returned 404 Not Found] """
    },

    "scenario2_susp_004_sudo_rsync_log_collection": {
        "file": "scenario2_cron_suspicious.log",
        "source_host": "inetfw",
        "log_type": "auth.log / auditd",
        "status": "suspicious",
        "category": "system_warning",
        "mitre_id": "T1005",
        "mitre_technique_name": "Data from Local System",
        "target_entities": ["aecid", "sudo", "/usr/bin/rsync", "/var/log", "172.17.100.201"],
        "observed_intent": "bulk_log_sync_or_data_collection",
        "confidence": "medium",
        "summary": "User aecid logging in via SSH and executing rsync with sudo over /var/log directory.",
        "raw_logs": """Oct  1 10:38:02 inetfw sshd[4587]: Accepted publickey for aecid from 172.17.100.201 port 58402 ssh2: RSA SHA256:yxAJcMgl5WTSlj4ZPvj7TSgUbIlx
Oct  1 10:38:02 inetfw sshd[4587]: pam_unix(sshd:session): session opened for user aecid(uid=1001) by (uid=0)
Oct  1 10:38:02 inetfw sudo:    aecid : PWD=/home/aecid ; USER=root ; COMMAND=/usr/bin/rsync --server --sender -lLogDtrze.LsfxCIvu . /var/log
Oct  1 10:38:02 inetfw sudo: pam_unix(sudo:session): session opened for user root(uid=0) by (uid=1001)
type=USER_CMD msg=audit(1759315082.688:7090): pid=4587 uid=1001 auid=1001 ses=5 msg='cwd="/home/aecid" cmd="rsync --server --sender -lLogDtrze.LsfxCIvu . /var/log" exe="/usr/bin/sudo" terminal=? res=success' """
    },

    "scenario2_susp_005_suricata_packet_thread_anomaly": {
        "file": "scenario2_cron_suspicious.log",
        "source_host": "inetfw",
        "log_type": "suricata fast.log",
        "status": "suspicious",
        "category": "system_warning",
        "mitre_id": None,
        "mitre_technique_name": None,
        "target_entities": ["172.17.100.121", "192.168.100.130", "port 1515", "TCP"],
        "observed_intent": "atypical_network_stream_pattern",
        "confidence": "low",
        "summary": "NIDS engine detecting out-of-sequence stream packets between internal host 172.17.100.121 and Wazuh manager port 1515.",
        "raw_logs": """10/01/2025-09:57:18.200098  [**] [1:2210059:1] SURICATA STREAM pkt seen on wrong thread [**] [Classification: (null)] [Priority: 3] {TCP} 172.17.100.121:35559 -> 192.168.100.130:1515
10/01/2025-09:57:53.208128  [**] [1:2210059:1] SURICATA STREAM pkt seen on wrong thread [**] [Classification: (null)] [Priority: 3] {TCP} 172.17.100.121:45659 -> 192.168.100.130:1515
10/01/2025-09:58:13.234022  [**] [1:2210059:1] SURICATA STREAM pkt seen on wrong thread [**] [Classification: (null)] [Priority: 3] {TCP} 172.17.100.121:58399 -> 192.168.100.130:1515
10/01/2025-09:58:17.384421  [**] [1:2210059:1] SURICATA STREAM pkt seen on wrong thread [**] [Classification: (null)] [Priority: 3] {TCP} 172.17.100.121:60805 -> 192.168.100.130:1515 """
    },

    "scenario2_susp_006_interactive_passwd_enum": {
        "file": "scenario2_cron_suspicious.log",
        "source_host": "videoserver",
        "log_type": "auditd",
        "status": "suspicious",
        "category": "security",
        "mitre_id": "T1003.008",
        "mitre_technique_name": "OS Credential Dumping: /etc/passwd and /etc/shadow",
        "target_entities": ["/etc/passwd", "pid=3288", "cat", "root"],
        "observed_intent": "local_user_enumeration",
        "confidence": "medium",
        "summary": "Root execution of cat on /etc/passwd to enumerate system accounts.",
        "raw_logs": """type=SYSCALL msg=audit(1759315553.431:5828): arch=c000003e syscall=59 success=yes exit=0 a0=55d21a08e120 a1=55d21a08e150 a2=55d21a08e170 a3=8 items=2 ppid=3217 pid=3288 auid=0 uid=0 gid=0 euid=0 comm="cat" exe="/bin/cat" key="T1087_Account_Discovery"
type=EXECVE msg=audit(1759315553.431:5828): argc=2 a0="cat" a1="/etc/passwd"
type=PATH msg=audit(1759315553.431:5828): item=0 name="/bin/cat" inode=30401 dev=fe:01 mode=0100755 ouid=0 ogid=0
type=PATH msg=audit(1759315553.431:5828): item=1 name="/etc/passwd" inode=30246 dev=fe:01 mode=0100644 ouid=0 ogid=0 """
    },

    # ==========================================
    # 3. BENIGN LOGS (status: clear)
    # ==========================================
    "scenario2_benign_001_routine_cron_session": {
        "file": "scenario2_cron_benign.log",
        "source_host": "corpdns",
        "log_type": "auth.log / syslog",
        "status": "clear",
        "category": "noise",
        "mitre_id": None,
        "mitre_technique_name": None,
        "target_entities": ["CRON[3020]", "root", "pam_unix"],
        "observed_intent": "admin_scheduled_task_maintenance",
        "confidence": "high",
        "summary": "Standard scheduled system cron job session open and close for root user on corpdns.",
        "raw_logs": """Oct  1 10:17:01 corpdns CRON[3020]: pam_unix(cron:session): session opened for user root(uid=0) by (uid=0)
Oct  1 10:17:01 corpdns CRON[3020]: (root) CMD (   test -x /etc/cron.daily/popularity-contest && /etc/cron.daily/popularity-contest --quiet)
Oct  1 10:17:01 corpdns CRON[3020]: pam_unix(cron:session): session closed for user root """
    },

    "scenario2_benign_002_systemd_boot_cloud_init": {
        "file": "scenario2_cron_benign.log",
        "source_host": "corpdns",
        "log_type": "syslog",
        "status": "clear",
        "category": "noise",
        "mitre_id": None,
        "mitre_technique_name": None,
        "target_entities": ["systemd[1]", "cloud-final.service", "cloudimg-rootfs"],
        "observed_intent": "routine_system_initialization",
        "confidence": "high",
        "summary": "Routine systemd service deactivations and filesystem check during VM cloud-init completion.",
        "raw_logs": """Sep 17 11:19:28 atb-corpdns systemd[1]: cloud-final.service: Deactivated successfully.
Sep 17 11:19:28 atb-corpdns systemd[1]: Stopped Execute cloud user/final scripts.
Sep 17 11:19:28 atb-corpdns systemd[1]: Stopped target Multi-User System.
Sep 17 11:19:28 atb-corpdns systemd[1]: cloud-config.service: Deactivated successfully.
Oct  1 09:53:01 corpdns systemd-fsck[397]: cloudimg-rootfs: clean, 84382/3225600 files, 814770/6525179 blocks """
    },

    "scenario2_benign_003_suricata_apt_outbound_policy": {
        "file": "scenario2_cron_benign.log",
        "source_host": "inetfw",
        "log_type": "suricata fast.log",
        "status": "clear",
        "category": "noise",
        "mitre_id": None,
        "mitre_technique_name": None,
        "target_entities": ["192.168.100.130", "185.125.190.95", "port 80", "APT User-Agent"],
        "observed_intent": "routine_package_manager_update",
        "confidence": "high",
        "summary": "Suricata policy signature for routine outbound Ubuntu APT package manager update check.",
        "raw_logs": """10/01/2025-09:57:35.156598  [**] [1:2013504:6] ET POLICY GNU/Linux APT User-Agent Outbound likely related to package management [**] [Classification: Not Suspicious Traffic] [Priority: 3] {TCP} 192.168.100.130:43234 -> 185.125.190.95:80
10/01/2025-09:57:35.175995  [**] [1:2013504:6] ET POLICY GNU/Linux APT User-Agent Outbound likely related to package management [**] [Classification: Not Suspicious Traffic] [Priority: 3] {TCP} 192.168.100.130:45978 -> 185.125.190.82:80
10/01/2025-09:57:36.130156  [**] [1:2013504:6] ET POLICY GNU/Linux APT User-Agent Outbound likely related to package management [**] [Classification: Not Suspicious Traffic] [Priority: 3] {TCP} 192.168.100.130:43234 -> 185.125.190.95:80 """
    },

    "scenario2_benign_004_systemd_user_manager_exit": {
        "file": "scenario2_cron_benign.log",
        "source_host": "videoserver",
        "log_type": "syslog",
        "status": "clear",
        "category": "noise",
        "mitre_id": None,
        "mitre_technique_name": None,
        "target_entities": ["systemd[2805]", "user@0.service", "UID 0"],
        "observed_intent": "system_session_logout_cleanup",
        "confidence": "high",
        "summary": "Normal systemd unit shutdown sequence upon root user logout.",
        "raw_logs": """Oct  1 10:35:46 videoserver systemd[2805]: Stopped target Main User Target.
Oct  1 10:35:46 videoserver systemd[2805]: Stopped target Basic System.
Oct  1 10:35:46 videoserver systemd[2805]: dirmngr.socket: Succeeded.
Oct  1 10:35:46 videoserver systemd[2805]: gpg-agent.socket: Succeeded.
Oct  1 10:35:46 videoserver systemd[1]: user@0.service: Succeeded.
Oct  1 10:35:46 videoserver systemd[1]: Stopped User Manager for UID 0.
Oct  1 10:35:46 videoserver systemd[1]: Stopped User Runtime Directory /run/user/0. """
    },

    "scenario2_benign_005_kernel_hardware_enumeration": {
        "file": "scenario2_cron_benign.log",
        "source_host": "corpdns",
        "log_type": "dmesg / syslog",
        "status": "clear",
        "category": "noise",
        "mitre_id": None,
        "mitre_technique_name": None,
        "target_entities": ["kernel", "x86_64", "Intel GenuineIntel"],
        "observed_intent": "hardware_initialization",
        "confidence": "high",
        "summary": "Kernel boot message reporting CPU features and microcode initialization.",
        "raw_logs": """Oct  1 09:53:01 corpdns kernel: [    0.000000] Linux version 5.15.0-113-generic (buildd@lcy02-amd64-072) (gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0, GNU ld (GNU Binutils for Ubuntu) 2.38) #123-Ubuntu SMP Mon Jun 10 08:16:17 UTC 2024
Oct  1 09:53:01 corpdns kernel: [    0.000000] Command line: BOOT_IMAGE=/boot/vmlinuz-5.15.0-113-generic root=LABEL=cloudimg-rootfs ro console=tty1 console=ttyS0
Oct  1 09:53:01 corpdns kernel: [    0.000000] KERNEL supported cpus:
Oct  1 09:53:01 corpdns kernel: [    0.000000]   Intel GenuineIntel
Oct  1 09:53:01 corpdns kernel: [    0.000000]   AMD AuthenticAMD """
    },

    "scenario2_benign_006_unattended_upgrades_check": {
        "file": "scenario2_cron_benign.log",
        "source_host": "videoserver",
        "log_type": "syslog",
        "status": "clear",
        "category": "noise",
        "mitre_id": None,
        "mitre_technique_name": None,
        "target_entities": ["unattended-upgrade", "dpkg", "apt"],
        "observed_intent": "automated_system_patching_check",
        "confidence": "high",
        "summary": "Scheduled unattended-upgrades service verifying no pending security updates before cleanly stopping.",
        "raw_logs": """Oct  1 09:54:12 videoserver systemd[1]: Starting Unattended Upgrades Shutdown...
Oct  1 09:54:12 videoserver unattended-upgrade[780]: Initializing unattended-upgrades
Oct  1 09:54:12 videoserver unattended-upgrade[780]: No packages found that can be upgraded unattended and no pending auto-removals
Oct  1 09:54:12 videoserver systemd[1]: unattended-upgrades.service: Deactivated successfully.
Oct  1 09:54:12 videoserver systemd[1]: Finished Unattended Upgrades Shutdown. """
    }
}


def build_curated_logs():
    files_content = {
        "scenario2_cron_offensive.log": [],
        "scenario2_cron_suspicious.log": [],
        "scenario2_cron_benign.log": []
    }

    ground_truth = {}

    for chunk_id, entry in DATASET_ENTRIES.items():
        filename = entry["file"]
        raw = entry["raw_logs"].strip()
        
        # Format the chunk in the .log file
        chunk_block = f"# --- START CHUNK: {chunk_id} ({entry['status'].upper()}) ---\n{raw}\n# --- END CHUNK: {chunk_id} ---\n"
        files_content[filename].append(chunk_block)

        # Store in ground truth dictionary (without duplicating heavy raw_logs text)
        ground_truth[chunk_id] = {
            "log_file": entry["file"],
            "source_host": entry["source_host"],
            "log_type": entry["log_type"],
            "status": entry["status"],
            "category": entry["category"],
            "mitre_id": entry["mitre_id"],
            "mitre_technique_name": entry["mitre_technique_name"],
            "target_entities": entry["target_entities"],
            "observed_intent": entry["observed_intent"],
            "confidence": entry["confidence"],
            "summary": entry["summary"]
        }

    # Write the 3 .log files
    for filename, chunks in files_content.items():
        out_path = os.path.join(OUTPUT_DIR, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"# Curated dataset for Scenario 2: {filename}\n# Total chunks: {len(chunks)}\n\n")
            f.write("\n".join(chunks))
        print(f"[*] Generated {out_path} ({len(chunks)} chunks)")

    # Write ground_truth.json
    gt_path = os.path.join(OUTPUT_DIR, "ground_truth.json")
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2)
    print(f"[*] Generated {gt_path} ({len(ground_truth)} entries)")


if __name__ == "__main__":
    build_curated_logs()
