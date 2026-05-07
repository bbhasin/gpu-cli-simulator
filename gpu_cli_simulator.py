#!/usr/bin/env python3
"""
GPU Cluster CLI Simulator — Run:ai / Slurm / Kubernetes
========================================================
A practice environment for the NVIDIA NCP-AIIOL exam.  Type real-shape
commands and get real-shape output, against a persistent simulated
cluster of 16 nodes × 8 H100 GPUs.

ONE-COMMAND RUN
---------------
    python3 gpu_cli_simulator.py

You'll land in a REPL prompt:

    (cluster) >

…where you can type any of:

    runai list jobs
    runai submit my-train -p ml-team -g 8 -i nvcr.io/nvidia/pytorch:24.03-py3 -- python train.py
    sinfo
    squeue
    sbatch train.sbatch
    scontrol show node node-001
    scontrol update NodeName=node-007 State=DRAIN Reason=PSU
    kubectl get nodes
    kubectl get pods -n ml-team
    kubectl describe node node-007
    kubectl exec -it pod-x -- nvidia-smi
    help            # list every simulated command
    scenarios       # exam-style practice scenarios
    reset           # rebuild fresh cluster state

State persists in `gpu_cli_state.json` next to this file, so jobs
you submit stay in the queue and nodes you drain stay drained
across REPL sessions.

ONE-SHOT MODE
-------------
    python3 gpu_cli_simulator.py runai list jobs
    python3 gpu_cli_simulator.py kubectl get nodes

OPTIONAL: SHELL ALIASES so you can type `runai …` etc. directly
---------------------------------------------------------------
Add these to ~/.zshrc or ~/.bashrc:

    SIM=~/path/to/gpu_cli_simulator.py
    alias runai='python3 $SIM runai'
    alias squeue='python3 $SIM squeue'
    alias sinfo='python3 $SIM sinfo'
    alias sbatch='python3 $SIM sbatch'
    alias scancel='python3 $SIM scancel'
    alias scontrol='python3 $SIM scontrol'
    alias sacct='python3 $SIM sacct'
    alias ksim='python3 $SIM kubectl'        # use ksim, not kubectl,
                                              # so you don't shadow the real one

No external dependencies — pure stdlib.
"""
from __future__ import annotations

import json
import os
import random
import re
import shlex
import time
import sys
import textwrap
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional


# ===========================================================================
# Persistent state
# ===========================================================================
STATE_FILE = Path(__file__).resolve().parent / "gpu_cli_state.json"
NOW = lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")


def fresh_state() -> dict:
    """Build a believable cluster mirroring the GPU Fleet platform."""
    # (lab-mode session state — see Section 21+ for usage)
    nodes = []
    for i in range(16):
        node_id = f"node-{i:03d}"
        # node-007 is the bad-PSU one (matches gpu_fleet_platform.py)
        state = "drain" if i == 7 else "idle" if i % 5 == 0 else "alloc"
        nodes.append({
            "name": node_id,
            "partition": "gpu" if i < 14 else "interactive",
            "state": state,
            "cpus": 192,
            "gpus": 8,
            "memory_gb": 1024,
            "gres": "gpu:h100:8",
            "reason": "PSU-investigation" if i == 7 else None,
            "k8s_taints": [],
        })

    projects = [
        {"name": "ml-research",   "department": "core-ai",     "gpu_quota": 64,
         "gpu_used": 48, "fairshare": 0.30, "guarantee": 32},
        {"name": "vision-team",   "department": "applied-ai",  "gpu_quota": 32,
         "gpu_used": 24, "fairshare": 0.20, "guarantee": 16},
        {"name": "nlp-team",      "department": "core-ai",     "gpu_quota": 24,
         "gpu_used": 16, "fairshare": 0.20, "guarantee": 12},
        {"name": "robotics",      "department": "applied-ai",  "gpu_quota": 16,
         "gpu_used": 4,  "fairshare": 0.15, "guarantee": 8},
        {"name": "infra-shared",  "department": "platform",    "gpu_quota": 8,
         "gpu_used": 0,  "fairshare": 0.15, "guarantee": 0},
    ]

    runai_jobs = [
        {"name": "bert-pretrain",        "project": "ml-research",
         "user": "alice",  "type": "Train",     "status": "Running",
         "gpus_requested": 16, "image": "nvcr.io/nvidia/pytorch:24.03-py3",
         "node": "node-001,node-002", "age_min": 142, "command": "python train.py"},
        {"name": "yolo-finetune",        "project": "vision-team",
         "user": "bob",    "type": "Train",     "status": "Running",
         "gpus_requested": 8,  "image": "nvcr.io/nvidia/pytorch:24.03-py3",
         "node": "node-003", "age_min": 67, "command": "python finetune.py"},
        {"name": "llama-eval-jupyter",   "project": "nlp-team",
         "user": "carol",  "type": "Interactive", "status": "Running",
         "gpus_requested": 1, "image": "nvcr.io/nvidia/pytorch:24.03-py3",
         "node": "node-014", "age_min": 312, "command": "jupyter lab"},
        {"name": "embeddings-batch",     "project": "ml-research",
         "user": "dave",   "type": "Train",     "status": "Pending",
         "gpus_requested": 32, "image": "nvcr.io/nvidia/pytorch:24.03-py3",
         "node": "", "age_min": 4, "command": "python embed.py",
         "pending_reason": "Resources"},
    ]

    slurm_jobs = [
        {"jobid": 12471, "user": "alice",  "partition": "gpu",
         "name": "bert-pre",   "state": "RUNNING", "time": "02:22:14",
         "nodes": 2, "nodelist": "node-[001-002]", "gres": "gpu:8"},
        {"jobid": 12472, "user": "bob",    "partition": "gpu",
         "name": "yolo-ft",    "state": "RUNNING", "time": "01:07:48",
         "nodes": 1, "nodelist": "node-003",       "gres": "gpu:8"},
        {"jobid": 12473, "user": "dave",   "partition": "gpu",
         "name": "embed-batch","state": "PENDING", "time": "0:00",
         "nodes": 4, "nodelist": "(Resources)",    "gres": "gpu:8"},
    ]

    pods = [
        {"name": "bert-pretrain-0-0", "namespace": "ml-team",
         "ready": "1/1", "status": "Running", "restarts": 0, "age": "2h22m",
         "node": "node-001", "ip": "10.244.1.45",
         "labels": {"app": "bert-pretrain", "runai/project": "ml-research"}},
        {"name": "bert-pretrain-0-1", "namespace": "ml-team",
         "ready": "1/1", "status": "Running", "restarts": 0, "age": "2h22m",
         "node": "node-002", "ip": "10.244.2.18"},
        {"name": "yolo-finetune-0-0", "namespace": "vision-team",
         "ready": "1/1", "status": "Running", "restarts": 0, "age": "1h7m",
         "node": "node-003", "ip": "10.244.3.71"},
        {"name": "embeddings-batch-0-0","namespace": "ml-team",
         "ready": "0/1", "status": "Pending", "restarts": 0, "age": "4m",
         "node": "<none>", "ip": "<none>"},
        {"name": "gpu-operator-7c8d", "namespace": "gpu-operator",
         "ready": "1/1", "status": "Running", "restarts": 0, "age": "12d",
         "node": "node-000", "ip": "10.244.0.5"},
        {"name": "nvidia-dcgm-exporter-9bf2", "namespace": "gpu-operator",
         "ready": "1/1", "status": "Running", "restarts": 0, "age": "12d",
         "node": "node-001"},
    ]

    return {
        "user": os.environ.get("USER", "student"),
        "host": "bcm",                # who-am-i for the prompt; flips on su/ssh
        "is_root": True,               # affects # vs $ in prompt
        "loaded_modules": [],          # `module load` accumulates here
        "dcgm_groups": [               # populated/edited by `dcgmi group` cmds
            {"id": 0, "name": "default",        "entities": "0,1,2,3,4,5,6,7"},
            {"id": 1, "name": "default_nvswitch","entities": "nvswitch:8,9,10,11,12,13"},
        ],
        "next_dcgm_group_id": 2,
        "k8s_user_added": False,       # toggled by cm-kubernetes-setup --add-user
        "wlm_setup_done": False,
        "k8s_setup_done": False,
        # Lab-2 (NVIDIA Container Toolkit + Docker + Driver labs)
        "driver_installed": True,      # `apt-get remove --purge ^nvidia` flips off
        "container_toolkit_installed": False,
        "docker_images": [],           # appended by `docker pull`
        "in_container": False,         # inside a `docker run -it` shell
        "container_id": "",
        "container_image": "",
        "container_cpu_only": False,   # True when --gpus is omitted
        "training_done": False,        # set by `python train.py`
        "bmc_logged_in": False,
        # BCM Admin Lab (Practices 0-5)
        "chroot_active": False,        # cm-chroot-sw-img engaged
        "chroot_image": "",            # name of the chroot image
        "chroot_files": [],            # files created via touch in chroot
        "rshell_node": "",             # rshell <node> session active
        "default_image_has_testtxt": False,
        "soundcore_added": False,      # `add soundcore` in kernelmodules
        "node001_kernel_synced": False,# True after reboot post-kernel change
        "cloned_images": [],           # softwareimage clone <src> <dst>
        "cloned_categories": [         # category clone — pre-seeded list
            {"name": "default", "image": "default-image"},
        ],
        "node_category": {             # per-node category override
            "node001": "default", "node002": "default",
            "node003": "default", "node004": "default",
        },
        "node_softwareimage": {},      # per-node sw image override (empty = inherit)
        "bcm_users": [                 # `cmsh user` state
            {"name": "cmsupport", "id": 1000, "primary": "cmsupport",
             "secondary": "", "password": ""},
        ],
        "bcm_groups": [
            {"name": "cmsupport", "id": 1000, "members": "cmsupport"},
        ],
        "next_user_id": 1001,
        "next_group_id": 1001,
        # Monitoring (Practice 5)
        "monitoring_data_producers": [],
        "monitoring_actions": [],
        "monitoring_triggers": [],
        "monitoring_dashboards": [],
        "current_project": "ml-research",
        "current_namespace": "ml-team",
        "nodes": nodes,
        "projects": projects,
        "departments": ["core-ai", "applied-ai", "platform"],
        "runai_jobs": runai_jobs,
        "slurm_jobs": slurm_jobs,
        "next_jobid": 12474,
        "pods": pods,
        "events": [
            {"ts": NOW(), "msg": "cluster initialised"},
        ],
    }


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            loaded = json.loads(STATE_FILE.read_text())
            # Migrate: backfill any keys added by newer fresh_state()
            template = fresh_state()
            for key, default_value in template.items():
                if key not in loaded:
                    loaded[key] = default_value
            save_state(loaded)
            return loaded
        except Exception:
            pass
    s = fresh_state()
    save_state(s)
    return s


def save_state(s: dict) -> None:
    STATE_FILE.write_text(json.dumps(s, indent=2))


# ===========================================================================
# Output helpers
# ===========================================================================
def table(headers: list[str], rows: list[list[Any]], pad: int = 3) -> str:
    if not rows:
        widths = [len(h) for h in headers]
    else:
        widths = [max(len(str(r[i])) if i < len(r) else 0 for r in rows + [headers])
                  for i in range(len(headers))]
    fmt = (" " * pad).join(f"{{:<{w}}}" for w in widths)
    out = [fmt.format(*headers)]
    for r in rows:
        cells = [str(r[i]) if i < len(r) else "" for i in range(len(headers))]
        out.append(fmt.format(*cells))
    return "\n".join(out)


def err(msg: str) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return 1


def info(msg: str) -> None:
    print(msg)


# ===========================================================================
# Run:ai commands
# ===========================================================================
RUNAI_HELP = """\
Usage: runai <command> [options]

Project / department:
  list projects                  list configured Run:ai projects
  list departments               list departments
  config project <name>          set the active project
  whoami                         show current user / project / cluster
  cluster info                   show cluster summary

Jobs (workloads):
  list jobs [-p <project>]       list jobs
  submit <name> -p <proj> -g <n> [-i <image>] [--interactive] -- <cmd...>
  describe job <name>            show full job details
  logs <name>                    print fake stdout for the job
  delete job <name>              delete a job
  bash <name>                    open shell into job (simulated)
  top job                        show running-job utilization

Examples:
  runai list jobs -p vision-team
  runai submit bert-test -p ml-research -g 8 -i nvcr.io/nvidia/pytorch:24.03-py3 -- python train.py
  runai submit jupyter-bob -p nlp-team -g 1 --interactive -i nvcr.io/nvidia/pytorch:24.03-py3
  runai describe job bert-pretrain
  runai delete job yolo-finetune
"""


def parse_flags(tokens: list[str], short: dict[str, str]) -> tuple[dict, list[str]]:
    """Tiny argparse — short maps `-p`->`project`. Returns (flags, positional)."""
    out: dict[str, Any] = {}
    pos: list[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == "--":
            pos.extend(tokens[i + 1:])
            return out, pos
        if t.startswith("--"):
            key = t[2:]
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                out[key] = tokens[i + 1]; i += 2; continue
            out[key] = True; i += 1; continue
        if t.startswith("-") and len(t) >= 2:
            key = short.get(t, t[1:])
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                out[key] = tokens[i + 1]; i += 2; continue
            out[key] = True; i += 1; continue
        pos.append(t); i += 1
    return out, pos


def cmd_runai(args: list[str], s: dict) -> int:
    if not args or args[0] in ("-h", "--help", "help"):
        print(RUNAI_HELP); return 0
    # Try the extras dispatcher first (workspaces, submit-mpi, update project,
    # create/delete project, exec, port-forward) — added in section 21+.
    extras_rc = cmd_runai_extras(args, s) if "cmd_runai_extras" in globals() else None
    if extras_rc is not None:
        return extras_rc
    sub = args[0]
    rest = args[1:]

    if sub == "whoami":
        info(f"user:    {s['user']}\n"
             f"project: {s['current_project']}\n"
             f"cluster: phoenix-prod-1")
        return 0

    if sub == "cluster":
        if rest and rest[0] == "info":
            tot_g = sum(n["gpus"] for n in s["nodes"])
            used = sum(p["gpu_used"] for p in s["projects"])
            info(f"NAME              phoenix-prod-1\n"
                 f"VERSION           2.18.x\n"
                 f"NODES             {len(s['nodes'])}\n"
                 f"GPUS_TOTAL        {tot_g}\n"
                 f"GPUS_ALLOCATED    {used}\n"
                 f"UTIL_PCT          {round(used/tot_g*100,1)}%")
            return 0

    if sub == "list":
        what = rest[0] if rest else ""
        if what == "projects":
            rows = [[p["name"], p["department"], p["gpu_quota"], p["gpu_used"],
                     p["guarantee"], f"{p['fairshare']:.2f}"] for p in s["projects"]]
            print(table(["NAME", "DEPARTMENT", "QUOTA", "USED",
                         "GUARANTEED", "FAIRSHARE"], rows))
            return 0
        if what == "departments":
            print(table(["DEPARTMENT", "PROJECTS", "QUOTA"],
                        [[d, sum(1 for p in s["projects"] if p["department"] == d),
                          sum(p["gpu_quota"] for p in s["projects"]
                              if p["department"] == d)]
                         for d in s["departments"]]))
            return 0
        if what == "jobs":
            flags, _ = parse_flags(rest[1:], {"-p": "project"})
            proj = flags.get("project")
            jobs = [j for j in s["runai_jobs"]
                    if not proj or j["project"] == proj]
            rows = [[j["name"], j["project"], j["user"], j["type"],
                     j["status"], j["gpus_requested"],
                     j.get("node", "") or "<pending>",
                     f"{j['age_min']}m"] for j in jobs]
            print(table(["NAME", "PROJECT", "USER", "TYPE", "STATUS",
                         "GPUs", "NODE", "AGE"], rows))
            return 0
        return err(f"runai list: unknown resource '{what}'")

    if sub == "config":
        if len(rest) >= 2 and rest[0] == "project":
            target = rest[1]
            if not any(p["name"] == target for p in s["projects"]):
                return err(f"project '{target}' not found")
            s["current_project"] = target
            save_state(s)
            info(f"Project '{target}' set as active.")
            return 0
        return err("usage: runai config project <name>")

    if sub == "submit":
        if not rest:
            return err("usage: runai submit <name> -p <proj> -g <n> "
                       "[-i <image>] [--interactive] -- <cmd...>")
        name = rest[0]
        flags, pos = parse_flags(rest[1:], {"-p": "project", "-g": "gpu",
                                            "-i": "image"})
        proj = flags.get("project") or s["current_project"]
        if not any(p["name"] == proj for p in s["projects"]):
            return err(f"project '{proj}' not found")
        gpus = int(flags.get("gpu", 1))
        image = flags.get("image", "nvcr.io/nvidia/pytorch:24.03-py3")
        interactive = bool(flags.get("interactive"))
        cmd_str = " ".join(pos) if pos else ("jupyter lab" if interactive else "sleep infinity")

        # quota check
        proj_obj = next(p for p in s["projects"] if p["name"] == proj)
        if proj_obj["gpu_used"] + gpus > proj_obj["gpu_quota"]:
            info(f"WARNING: Job '{name}' exceeds project quota "
                 f"({proj_obj['gpu_used']}+{gpus} > {proj_obj['gpu_quota']}). "
                 f"Will run as over-quota using fairshare.")

        # placement
        free_node = next((n for n in s["nodes"]
                         if n["state"] == "idle"), None)
        node = free_node["name"] if free_node and gpus <= 8 else ""
        status = "Running" if node else "Pending"

        s["runai_jobs"].append({
            "name": name, "project": proj, "user": s["user"],
            "type": "Interactive" if interactive else "Train",
            "status": status, "gpus_requested": gpus,
            "image": image, "node": node, "age_min": 0,
            "command": cmd_str,
            "pending_reason": "" if status == "Running" else "Resources",
        })
        if node:
            free_node["state"] = "alloc"
            proj_obj["gpu_used"] += gpus
        save_state(s)

        info(f"INFO[{NOW()}] Job submitted successfully")
        info(f"INFO[{NOW()}] You can run 'runai describe job {name} -p {proj}' "
             f"to check the job status")
        return 0

    if sub == "describe":
        if len(rest) < 2 or rest[0] != "job":
            return err("usage: runai describe job <name>")
        name = rest[1]
        j = next((x for x in s["runai_jobs"] if x["name"] == name), None)
        if not j:
            return err(f"job '{name}' not found")
        info(f"Name:           {j['name']}")
        info(f"Project:        {j['project']}")
        info(f"User:           {j['user']}")
        info(f"Type:           {j['type']}")
        info(f"Status:         {j['status']}")
        info(f"Image:          {j['image']}")
        info(f"GPUs:           {j['gpus_requested']}")
        info(f"Node:           {j.get('node') or '<pending>'}")
        info(f"Command:        {j.get('command','')}")
        info(f"Age:            {j['age_min']}m")
        if j["status"] == "Pending":
            info(f"Pending reason: {j.get('pending_reason','Resources')}")
        info("\nPod events:")
        info(f"  {j['age_min']}m ago  Scheduled  Successfully assigned to {j.get('node')}")
        info(f"  {j['age_min']}m ago  Pulled     Container image '{j['image']}'")
        info(f"  {j['age_min']}m ago  Started    Container ml-job")
        return 0

    if sub == "logs":
        if not rest:
            return err("usage: runai logs <job>")
        name = rest[0]
        j = next((x for x in s["runai_jobs"] if x["name"] == name), None)
        if not j:
            return err(f"job '{name}' not found")
        if j["status"] == "Pending":
            return err(f"job '{name}' is Pending — no logs yet")
        # Fake plausible training output
        epochs = max(1, j["age_min"] // 12)
        info(f"[rank0]: starting training: {j['command']}")
        info(f"[rank0]: torch=2.3.0  cuda=12.4  nccl=2.20")
        info(f"[rank0]: world_size={j['gpus_requested']}  "
             f"backend=nccl  ifname=eth0")
        for e in range(epochs):
            loss = round(2.4 * (0.92 ** e) + random.uniform(-0.02, 0.02), 4)
            tps = int(2400 + random.uniform(-80, 80))
            info(f"[rank0]: epoch={e:03d}  loss={loss}  tokens/s/gpu={tps}")
        return 0

    if sub == "delete":
        if len(rest) < 2 or rest[0] != "job":
            return err("usage: runai delete job <name>")
        name = rest[1]
        before = len(s["runai_jobs"])
        s["runai_jobs"] = [j for j in s["runai_jobs"] if j["name"] != name]
        if len(s["runai_jobs"]) == before:
            return err(f"job '{name}' not found")
        save_state(s)
        info(f"Job '{name}' deleted")
        return 0

    if sub == "bash":
        if not rest:
            return err("usage: runai bash <job>")
        info(f"[simulated] would attach interactive shell to {rest[0]} (Ctrl-D to detach)")
        return 0

    if sub == "top" and rest and rest[0] == "job":
        rows = []
        for j in s["runai_jobs"]:
            if j["status"] != "Running":
                continue
            util = random.randint(60, 95) if j["type"] == "Train" else random.randint(2, 25)
            rows.append([j["name"], j["project"], j["gpus_requested"],
                         f"{util}%", f"{random.randint(20,75)}GB"])
        print(table(["NAME", "PROJECT", "GPUs", "GPU_UTIL", "MEM"], rows))
        return 0

    return err(f"runai: unknown command '{sub}'.  try 'runai help'")


# ===========================================================================
# Slurm commands
# ===========================================================================
SLURM_HELP = """\
Slurm commands simulated:

  sinfo                        partition / node summary
  sinfo -N                     per-node detail
  squeue                       running + pending jobs
  squeue -u <user>             filter by user
  sbatch <script.sh>           submit a batch script (parses #SBATCH)
  srun -N <n> --gres=gpu:<g> <cmd>   run interactively (simulated)
  scancel <jobid>              cancel a job
  scontrol show node <name>    full node detail
  scontrol show job <jobid>
  scontrol update NodeName=<n> State={DRAIN|RESUME} Reason=<text>
  sacct                        job-accounting history
"""


def slurm_node_state(n: dict) -> str:
    if n["state"] == "drain": return "drain"
    if n["state"] == "alloc": return "alloc"
    return "idle"


def cmd_sinfo(args: list[str], s: dict) -> int:
    flags, _ = parse_flags(args, {})
    if flags.get("N") or flags.get("Node"):
        rows = [[n["name"], 1, n["partition"], slurm_node_state(n)]
                for n in s["nodes"]]
        print(table(["NODELIST", "NODES", "PARTITION", "STATE"], rows))
        return 0
    # default partition view
    parts: dict[str, list[dict]] = {}
    for n in s["nodes"]:
        parts.setdefault(n["partition"], []).append(n)
    rows = []
    for p, ns in parts.items():
        by_state: dict[str, list[str]] = {}
        for n in ns:
            by_state.setdefault(slurm_node_state(n), []).append(n["name"])
        for st, names in by_state.items():
            rows.append([p + "*", "infinite", len(names), st,
                         compact_nodelist(names)])
    print(table(["PARTITION", "TIMELIMIT", "NODES", "STATE", "NODELIST"], rows))
    return 0


def compact_nodelist(names: list[str]) -> str:
    """Turn ['node-001','node-002','node-003','node-005']
       into 'node-[001-003,005]' — Slurm-style."""
    nums = sorted(int(n.split("-")[1]) for n in names)
    if not nums: return ""
    ranges: list[str] = []
    start = prev = nums[0]
    for v in nums[1:]:
        if v == prev + 1:
            prev = v; continue
        ranges.append(f"{start:03d}" if start == prev else f"{start:03d}-{prev:03d}")
        start = prev = v
    ranges.append(f"{start:03d}" if start == prev else f"{start:03d}-{prev:03d}")
    return f"node-[{','.join(ranges)}]"


def cmd_squeue(args: list[str], s: dict) -> int:
    flags, _ = parse_flags(args, {"-u": "user"})
    user = flags.get("user")
    rows = []
    for j in s["slurm_jobs"]:
        if user and j["user"] != user: continue
        rows.append([j["jobid"], j["partition"], j["name"], j["user"],
                     j["state"][:2] if j["state"] != "PENDING" else "PD",
                     j["time"], j["nodes"], j["nodelist"]])
    print(table(["JOBID", "PARTITION", "NAME", "USER", "ST",
                 "TIME", "NODES", "NODELIST(REASON)"], rows))
    return 0


def cmd_sbatch(args: list[str], s: dict) -> int:
    if not args:
        return err("usage: sbatch <script.sh>")
    script_path = Path(args[0])
    name, gpus, nodes_n, partition, time_str = "job", 1, 1, "gpu", "01:00:00"
    if script_path.exists():
        for line in script_path.read_text().splitlines():
            if not line.startswith("#SBATCH"): continue
            tok = line.replace("#SBATCH", "").strip()
            m = re.match(r"-J\s+(\S+)|--job-name=(\S+)", tok)
            if m: name = m.group(1) or m.group(2)
            m = re.match(r"-p\s+(\S+)|--partition=(\S+)", tok)
            if m: partition = m.group(1) or m.group(2)
            m = re.match(r"-N\s+(\d+)|--nodes=(\d+)", tok)
            if m: nodes_n = int(m.group(1) or m.group(2))
            m = re.match(r"--gres=gpu:(\d+)", tok)
            if m: gpus = int(m.group(1)) * nodes_n
            m = re.match(r"-t\s+(\S+)|--time=(\S+)", tok)
            if m: time_str = m.group(1) or m.group(2)
    else:
        info(f"NOTE: '{script_path}' not found on disk — using defaults.")

    jid = s["next_jobid"]; s["next_jobid"] += 1
    free_nodes = [n for n in s["nodes"]
                  if slurm_node_state(n) == "idle"
                  and n["partition"] == partition][:nodes_n]
    if len(free_nodes) >= nodes_n:
        for n in free_nodes: n["state"] = "alloc"
        nodelist = compact_nodelist([n["name"] for n in free_nodes])
        state = "RUNNING"
    else:
        nodelist = "(Resources)"
        state = "PENDING"

    s["slurm_jobs"].append({"jobid": jid, "user": s["user"],
                            "partition": partition, "name": name,
                            "state": state, "time": "0:00",
                            "nodes": nodes_n, "nodelist": nodelist,
                            "gres": f"gpu:{gpus // max(1,nodes_n)}"})
    save_state(s)
    info(f"Submitted batch job {jid}")
    return 0


def cmd_scancel(args: list[str], s: dict) -> int:
    if not args:
        return err("usage: scancel <jobid>")
    try:
        jid = int(args[0])
    except ValueError:
        return err("jobid must be an integer")
    j = next((x for x in s["slurm_jobs"] if x["jobid"] == jid), None)
    if not j:
        return err(f"scancel: error: Kill job error on job id {jid}: Invalid job id specified")
    s["slurm_jobs"] = [x for x in s["slurm_jobs"] if x["jobid"] != jid]
    save_state(s)
    return 0  # scancel is silent on success


def cmd_scontrol(args: list[str], s: dict) -> int:
    if not args:
        return err("usage: scontrol show node|job ...  | scontrol update NodeName=...")
    sub = args[0]
    if sub == "show":
        what = args[1] if len(args) > 1 else ""
        if what == "node" and len(args) >= 3:
            n = next((x for x in s["nodes"] if x["name"] == args[2]), None)
            if not n: return err(f"node {args[2]} not found")
            info(f"NodeName={n['name']} CoresPerSocket=48 CPUs={n['cpus']}")
            info(f"   Partitions={n['partition']}")
            info(f"   Gres={n['gres']}")
            info(f"   State={slurm_node_state(n).upper()} ThreadsPerCore=2")
            info(f"   RealMemory={n['memory_gb']*1024}")
            if n["reason"]:
                info(f"   Reason={n['reason']} [{NOW()}]")
            return 0
        if what == "job" and len(args) >= 3:
            try: jid = int(args[2])
            except ValueError: return err("jobid must be int")
            j = next((x for x in s["slurm_jobs"] if x["jobid"] == jid), None)
            if not j: return err(f"job {jid} not found")
            info(f"JobId={j['jobid']} JobName={j['name']}")
            info(f"   UserId={j['user']}(1000) GroupId=staff(1000)")
            info(f"   Partition={j['partition']} NumNodes={j['nodes']}")
            info(f"   JobState={j['state']} NodeList={j['nodelist']}")
            info(f"   Gres={j['gres']}")
            return 0
        return err("scontrol show: try 'scontrol show node <n>' or 'scontrol show job <id>'")

    if sub == "update":
        kv = {}
        for tok in args[1:]:
            if "=" in tok:
                k, v = tok.split("=", 1); kv[k] = v
        node_name = kv.get("NodeName")
        new_state = kv.get("State")
        reason = kv.get("Reason", "operator-action")
        if not node_name or not new_state:
            return err("usage: scontrol update NodeName=<n> State={DRAIN|RESUME} Reason=<text>")
        n = next((x for x in s["nodes"] if x["name"] == node_name), None)
        if not n: return err(f"node {node_name} not found")
        if new_state.upper() == "DRAIN":
            n["state"] = "drain"; n["reason"] = reason
        elif new_state.upper() in ("RESUME", "IDLE"):
            n["state"] = "idle"; n["reason"] = None
        else:
            return err(f"unsupported state '{new_state}'")
        save_state(s)
        info(f"Updated {node_name}: state={n['state']} reason={n['reason']}")
        return 0

    return err(f"scontrol: unknown subcommand '{sub}'")


def cmd_sacct(args: list[str], s: dict) -> int:
    flags, _ = parse_flags(args, {"-u": "user"})
    user = flags.get("user")
    rows = []
    # current jobs
    for j in s["slurm_jobs"]:
        if user and j["user"] != user: continue
        rows.append([j["jobid"], j["name"], j["partition"], j["user"],
                     j["state"], j["time"]])
    # plus 6 fake historical
    for i in range(6):
        rows.append([12000 + i, f"old-job-{i}", "gpu",
                     s["user"], "COMPLETED",
                     f"{random.randint(1,8):02d}:00:00"])
    print(table(["JobID", "JobName", "Partition", "User", "State", "Elapsed"], rows))
    return 0


# ===========================================================================
# kubectl commands
# ===========================================================================
KUBECTL_HELP = """\
kubectl commands simulated:

  kubectl get nodes [-o wide]
  kubectl get pods [-n <ns>] [-o wide] [--all-namespaces]
  kubectl get namespaces
  kubectl get clusterpolicy           # NVIDIA GPU operator
  kubectl get pvc [-n <ns>]
  kubectl describe node <name>
  kubectl describe pod <name> [-n <ns>]
  kubectl logs <pod> [-n <ns>]
  kubectl exec <pod> [-n <ns>] -- <cmd...>      # try '-- nvidia-smi'
  kubectl top node
  kubectl top pod [-n <ns>]
  kubectl cordon <node>      |  kubectl uncordon <node>
  kubectl drain <node> --ignore-daemonsets --delete-emptydir-data
  kubectl delete pod <name> [-n <ns>]
  kubectl apply -f <yaml>    # parses kind/metadata.name from a real-ish file
"""


def cmd_kubectl(args: list[str], s: dict) -> int:
    if not args or args[0] in ("-h", "--help", "help"):
        print(KUBECTL_HELP); return 0
    sub = args[0]
    rest = args[1:]

    if sub == "get":
        return kubectl_get(rest, s)
    if sub == "describe":
        return kubectl_describe(rest, s)
    if sub == "logs":
        if not rest: return err("usage: kubectl logs <pod>")
        # Lab-flow special case: gpu-pod's stdout was nvidia-smi
        if rest[0] == "gpu-pod":
            print(GPU_POD_LOG.rstrip()); return 0
        return cmd_runai(["logs", rest[0].rsplit("-", 2)[0]], s)
    if sub == "exec":
        flags, pos = parse_flags(rest, {"-n": "namespace"})
        if "-it" in rest:
            pos = [t for t in pos if t != "-it"]
        if not pos: return err("usage: kubectl exec <pod> -- <cmd>")
        pod = pos[0]; rest_cmd = pos[1:]
        if rest_cmd and rest_cmd[0] == "nvidia-smi":
            return print_fake_nvidia_smi(pod)
        info(f"[simulated exec into {pod}] ran: {' '.join(rest_cmd) or '<no command>'}")
        return 0
    if sub == "top":
        return kubectl_top(rest, s)
    if sub == "cordon":
        if not rest: return err("usage: kubectl cordon <node>")
        n = next((x for x in s["nodes"] if x["name"] == rest[0]), None)
        if not n: return err(f"node {rest[0]} not found")
        n["state"] = "drain"; save_state(s)
        info(f"node/{rest[0]} cordoned"); return 0
    if sub == "uncordon":
        if not rest: return err("usage: kubectl uncordon <node>")
        n = next((x for x in s["nodes"] if x["name"] == rest[0]), None)
        if not n: return err(f"node {rest[0]} not found")
        n["state"] = "idle"; n["reason"] = None; save_state(s)
        info(f"node/{rest[0]} uncordoned"); return 0
    if sub == "drain":
        if not rest: return err("usage: kubectl drain <node> [--ignore-daemonsets]")
        node = rest[0]
        n = next((x for x in s["nodes"] if x["name"] == node), None)
        if not n: return err(f"node {node} not found")
        n["state"] = "drain"; n["reason"] = "kubectl drain"
        evicted = [p["name"] for p in s["pods"] if p["node"] == node]
        s["pods"] = [p for p in s["pods"] if p["node"] != node
                     or "daemonset" in str(p).lower()]
        save_state(s)
        info(f"node/{node} cordoned")
        for e in evicted:
            info(f"evicting pod default/{e}")
        info(f"node/{node} drained")
        return 0
    if sub == "delete":
        if len(rest) >= 2 and rest[0] == "pod":
            pod = rest[1]
            before = len(s["pods"])
            s["pods"] = [p for p in s["pods"] if p["name"] != pod]
            if len(s["pods"]) == before:
                return err(f"pod \"{pod}\" not found")
            save_state(s)
            info(f"pod \"{pod}\" deleted"); return 0
        return err("usage: kubectl delete pod <name>")
    if sub == "apply":
        flags, _ = parse_flags(rest, {"-f": "file"})
        f = flags.get("file") or (rest[1] if len(rest) > 1 and rest[0] == "-f" else None)
        if not f: return err("usage: kubectl apply -f <yaml>")
        # Lab-flow: gpu.yaml in /cm/shared/apps/lp/ creates a single gpu-pod
        if "gpu.yaml" in f or "gpu-pod" in f:
            existing = next((p for p in s["pods"] if p["name"] == "gpu-pod"), None)
            if not existing:
                s["pods"].append({
                    "name": "gpu-pod", "namespace": "default",
                    "ready": "0/1", "status": "ContainerCreating",
                    "restarts": 0, "age": "0s",
                    "node": "k8s-worker-01", "ip": "<none>",
                    "check_count": 0,
                })
                save_state(s)
            info("pod/gpu-pod created"); return 0
        info(f"deployment.apps/{Path(f).stem} created")
        return 0
    # Extensions added in section 21+:
    if sub == "scale":        return kubectl_scale(rest, s)
    if sub == "rollout":      return kubectl_rollout(rest, s)
    if sub == "label":        return kubectl_label(rest, s)
    if sub == "taint":        return kubectl_taint(rest, s)
    if sub == "run":          return kubectl_run(rest, s)
    if sub == "create":       return kubectl_create(rest, s)
    if sub in ("port-forward", "pf"): return kubectl_port_forward(rest, s)
    if sub == "edit":         return kubectl_edit(rest, s)
    if sub == "explain":      return kubectl_explain(rest, s)

    return err(f"kubectl: unknown command '{sub}'")


def kubectl_get(args: list[str], s: dict) -> int:
    if not args: return err("usage: kubectl get <resource>")
    res = args[0]
    flags, _ = parse_flags(args[1:], {"-n": "namespace", "-o": "output"})
    wide = flags.get("output") == "wide"
    all_ns = bool(flags.get("all-namespaces") or flags.get("A"))
    ns = flags.get("namespace", s["current_namespace"])

    if res == "nodes":
        rows = []
        for n in s["nodes"]:
            status = "Ready,SchedulingDisabled" if n["state"] == "drain" else "Ready"
            row = [n["name"], status, "control-plane,worker" if n["name"]=="node-000" else "worker",
                   "12d", "v1.28.5"]
            if wide: row += ["10.0.0." + n["name"].split("-")[1],
                             "Ubuntu 22.04.4 LTS", "5.15.0", "containerd://1.7.13"]
            rows.append(row)
        headers = ["NAME", "STATUS", "ROLES", "AGE", "VERSION"]
        if wide: headers += ["INTERNAL-IP", "OS-IMAGE", "KERNEL", "CONTAINER-RUNTIME"]
        print(table(headers, rows)); return 0

    if res == "namespaces" or res == "ns":
        nss = sorted({p["namespace"] for p in s["pods"]} |
                     {"default", "kube-system", "gpu-operator", "runai"})
        print(table(["NAME", "STATUS", "AGE"],
                    [[n, "Active", "12d"] for n in nss])); return 0

    if res == "pods" or res == "po":
        # Lab-mode: progressively transition gpu-pod ContainerCreating -> Completed
        gpu_pod = next((p for p in s["pods"] if p["name"] == "gpu-pod"), None)
        if gpu_pod:
            checks = gpu_pod.get("check_count", 0) + 1
            gpu_pod["check_count"] = checks
            if checks == 1:
                gpu_pod["status"] = "ContainerCreating"
                gpu_pod["ready"] = "0/1"; gpu_pod["age"] = "5s"
            elif checks == 2:
                gpu_pod["status"] = "Running"
                gpu_pod["ready"] = "1/1"; gpu_pod["age"] = "12s"
            else:
                gpu_pod["status"] = "Completed"
                gpu_pod["ready"] = "0/1"; gpu_pod["age"] = "20s"
            save_state(s)
        pods = s["pods"] if all_ns else [p for p in s["pods"] if p["namespace"] == ns]
        rows = []
        for p in pods:
            row = [p["name"], p["ready"], p["status"],
                   p["restarts"], p["age"]]
            if all_ns: row.insert(0, p["namespace"])
            if wide: row += [p.get("ip","<none>"), p.get("node","<none>"), "<none>", "<none>"]
            rows.append(row)
        headers = (["NAMESPACE"] if all_ns else []) + \
                  ["NAME", "READY", "STATUS", "RESTARTS", "AGE"] + \
                  (["IP", "NODE", "NOMINATED", "READINESS GATES"] if wide else [])
        print(table(headers, rows)); return 0

    if res == "clusterpolicy":
        info("NAME              STATUS   AGE")
        info("cluster-policy    ready    12d")
        return 0

    if res == "pvc":
        rows = [["model-cache", "Bound", "pvc-1234", "500Gi", "RWX", "vast-rwx", "12d"],
                ["scratch-data", "Bound", "pvc-5678", "10Ti",  "RWX", "vast-rwx", "9d"]]
        print(table(["NAME","STATUS","VOLUME","CAPACITY","ACCESS MODES",
                     "STORAGECLASS","AGE"], rows))
        return 0

    return err(f"kubectl get: unknown resource '{res}'")


def kubectl_describe(args: list[str], s: dict) -> int:
    if len(args) < 2: return err("usage: kubectl describe <resource> <name>")
    res, name = args[0], args[1]
    if res == "node":
        n = next((x for x in s["nodes"] if x["name"] == name), None)
        if not n: return err(f"node {name} not found")
        status = "True" if n["state"] != "drain" else "Unschedulable"
        info(f"Name:               {n['name']}")
        info(f"Roles:              {'control-plane,worker' if n['name']=='node-000' else 'worker'}")
        info(f"Labels:             nvidia.com/gpu.product=H100-SXM5-80GB")
        info(f"                    nvidia.com/gpu.count={n['gpus']}")
        info(f"                    nvidia.com/mig.strategy=single")
        info(f"Taints:             <none>")
        info(f"Unschedulable:      {'true' if n['state']=='drain' else 'false'}")
        info(f"Conditions:")
        info(f"  Ready             {status}   KubeletReady    kubelet is posting ready status")
        info(f"Capacity:")
        info(f"  cpu:                {n['cpus']}")
        info(f"  memory:             {n['memory_gb']*1024}Mi")
        info(f"  nvidia.com/gpu:     {n['gpus']}")
        if n["reason"]: info(f"Drain reason:       {n['reason']}")
        return 0
    if res == "pod":
        p = next((x for x in s["pods"] if x["name"] == name), None)
        if not p: return err(f"pod {name} not found")
        info(f"Name:           {p['name']}")
        info(f"Namespace:      {p['namespace']}")
        info(f"Node:           {p.get('node')}")
        info(f"Status:         {p['status']}")
        info(f"IP:             {p.get('ip','<none>')}")
        info(f"Containers:")
        info(f"  ml-job:")
        info(f"    Image:        nvcr.io/nvidia/pytorch:24.03-py3")
        info(f"    Limits:")
        info(f"      nvidia.com/gpu: 1")
        info(f"Events:")
        info(f"  Type    Reason     Age   From               Message")
        info(f"  Normal  Scheduled  2h    default-scheduler  Successfully assigned to {p.get('node')}")
        info(f"  Normal  Pulled     2h    kubelet            Container image already present")
        info(f"  Normal  Started    2h    kubelet            Started container ml-job")
        return 0
    return err(f"kubectl describe: unknown resource '{res}'")


def kubectl_top(args: list[str], s: dict) -> int:
    if not args: return err("usage: kubectl top {node|pod}")
    if args[0] == "node":
        rows = []
        for n in s["nodes"]:
            cpu_pct = random.randint(15, 85) if n["state"] == "alloc" else random.randint(2, 15)
            mem_pct = random.randint(20, 70) if n["state"] == "alloc" else random.randint(5, 20)
            rows.append([n["name"], f"{int(n['cpus']*cpu_pct/100)}m", f"{cpu_pct}%",
                         f"{int(n['memory_gb']*mem_pct*10)}Mi", f"{mem_pct}%"])
        print(table(["NAME","CPU(cores)","CPU%","MEMORY(bytes)","MEMORY%"], rows))
        return 0
    if args[0] == "pod":
        rows = []
        for p in s["pods"]:
            if p["status"] != "Running": continue
            rows.append([p["name"], f"{random.randint(50,2000)}m",
                         f"{random.randint(100,8000)}Mi"])
        print(table(["NAME","CPU(cores)","MEMORY(bytes)"], rows))
        return 0
    return err("kubectl top: unknown subresource")


def print_fake_nvidia_smi(pod: str) -> int:
    print(textwrap.dedent(f"""\
    Thu Apr 30 14:32:11 2026
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 550.54.14    Driver Version: 550.54.14    CUDA Version: 12.4     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name                Bus-Id        Disp.A | Volatile Uncorr. ECC       |
    | Fan  Temp Perf Pwr:Usage/Cap |  Memory-Usage    | GPU-Util  Compute M.     |
    |===============================+======================+======================|
    |   0  NVIDIA H100 80GB HBM3   00000000:1B:00.0 Off |                    0   |
    | N/A   62C    P0   421W / 700W |  64512MiB / 81920MiB |     87%   Default   |
    +-------------------------------+----------------------+----------------------+
    Processes:
     GPU   PID   Type   Process name                           GPU Memory
       0  24193  C      python train.py (in {pod})             64500MiB
    +-----------------------------------------------------------------------------+
    """))
    return 0


# ===========================================================================
# Base Command Manager (BCM) — cmsh and cm-* utilities
# ===========================================================================
CMSH_HELP = """\
cmsh — Bright/NVIDIA Base Command Manager shell.  Supported forms:

  cmsh -c "device list"
  cmsh -c "device status node-007"
  cmsh -c "category list"
  cmsh -c "softwareimage list"
  cmsh -c "monitoring metrics gpu_temp"
  cmsh -c "monitoring alerts"
  cmsh -c "wlm; status"
  cmsh device list                # flat form (simulator extension)

Modes: device | category | softwareimage | monitoring | wlm | main

The real cmsh is an interactive nested shell.  This simulator accepts
the -c "<command>" form and a flat form so you can practice the same
verbs without a live BCM head node.
"""

BCM_IMAGES = [
    {"name": "default-image",       "kernel": "5.15.0-91-generic",
     "size_gb": 18, "category": "default"},
    {"name": "h100-cuda12.4-image", "kernel": "5.15.0-91-generic",
     "size_gb": 24, "category": "gpu-h100"},
    {"name": "interactive-jupyter", "kernel": "5.15.0-91-generic",
     "size_gb": 22, "category": "interactive"},
]

BCM_CATEGORIES = [
    {"name": "default",     "image": "default-image",       "nodes": 0},
    {"name": "gpu-h100",    "image": "h100-cuda12.4-image", "nodes": 14},
    {"name": "interactive", "image": "interactive-jupyter", "nodes": 2},
    {"name": "head",        "image": "default-image",       "nodes": 1},
]


def cmd_cmsh(args: list[str], s: dict) -> int:
    if not args:
        # No args -> launch the stateful interactive cmsh REPL (NVIDIA lab style)
        return cmsh_interactive(s)
    if args[0] in ("-h", "--help", "help"):
        print(CMSH_HELP); return 0
    if args[0] == "-c":
        if len(args) < 2:
            return err("cmsh -c requires a command string")
        # cmsh chains semicolon-separated commands within a *mode* context:
        # 'wlm; status' means: enter wlm mode, then run 'status' inside it.
        chunks = [c.strip() for c in args[1].split(";") if c.strip()]
        current_mode = None
        for chunk in chunks:
            tokens = shlex.split(chunk)
            if not tokens: continue
            VALID_MODES = {"device", "category", "softwareimage",
                           "monitoring", "wlm", "main"}
            # If the first token is a known mode, switch context.
            # Only print if the chunk has a sub-command (mode alone =
            # just enter-mode, like real cmsh, no output).
            if tokens[0] in VALID_MODES:
                current_mode = tokens[0]
                if len(tokens) == 1:
                    rc = 0  # bare mode switch — no output
                else:
                    rc = cmsh_dispatch(tokens, s)
            elif current_mode:
                rc = cmsh_dispatch([current_mode] + tokens, s)
            else:
                rc = cmsh_dispatch(tokens, s)
            if rc != 0: return rc
        return 0
    return cmsh_dispatch(args, s)


def cmsh_dispatch(tokens: list[str], s: dict) -> int:
    if not tokens: return 0
    mode, rest = tokens[0], tokens[1:]
    # Bare verbs that work in every mode (BCM cmsh chunk-form)
    if mode == "commit" or (rest and rest[0] == "commit"):
        info("(commit accepted)")
        return 0
    if mode == "device":         return cmsh_device_v2(rest, s)
    if mode == "category":       return cmsh_category_v2(rest, s)
    if mode == "softwareimage":  return cmsh_image_v2(rest, s)
    if mode == "monitoring":     return cmsh_monitoring(rest, s)
    if mode == "wlm":            return cmsh_wlm(rest, s)
    if mode == "main":           return cmsh_main(rest, s)
    if mode == "user":           return cmd_bcmuser(rest, s)
    if mode == "group":          return cmd_bcmgroup(rest, s)
    if mode in ("quit", "exit"): return 0
    return err(f"cmsh: unknown mode '{mode}'")


def cmsh_device(args: list[str], s: dict) -> int:
    cmd = args[0] if args else "list"
    if cmd == "list":
        rows = []
        for n in s["nodes"]:
            cat = ("head" if n["name"] == "node-000"
                   else "gpu-h100" if n["partition"] == "gpu"
                   else "interactive")
            up = "UP" if n["state"] != "drain" else "DOWN"
            rows.append([n["name"], "physicalnode", cat, up,
                         "10.0.0." + n["name"].split("-")[1]])
        print(table(["NAME", "TYPE", "CATEGORY", "STATUS", "IP"], rows))
        return 0
    if cmd == "status":
        target = args[1] if len(args) > 1 else None
        nodes = [n for n in s["nodes"] if not target or n["name"] == target]
        if target and not nodes: return err(f"node {target} not found")
        rows = []
        for n in nodes:
            health = "OK" if n["state"] != "drain" else "FAIL"
            rows.append([n["name"], n["state"].upper(), health,
                         (n.get("reason") or "-")])
        print(table(["NAME", "STATE", "HEALTH", "REASON"], rows))
        return 0
    if cmd == "use":
        if len(args) < 2: return err("usage: device use <node>")
        info(f"[cluster->device[{args[1]}]]   (interactive context simulated)")
        return 0
    if cmd == "reboot":
        if len(args) < 2: return err("usage: device reboot <node>")
        info(f"Reboot of {args[1]} initiated."); return 0
    if cmd == "power":
        if len(args) < 3: return err("usage: device power {on|off|reset} <node>")
        info(f"Power {args[1]} of {args[2]} accepted."); return 0
    if cmd == "open":
        if len(args) < 2: return err("usage: device open <node>")
        info(f"[cluster->device[{args[1]}]>   "); return 0
    if cmd == "ping":
        if len(args) < 2: return err("usage: device ping <node>")
        info(f"PING {args[1]}: 0% packet loss, avg 0.21ms"); return 0
    return err(f"cmsh device: unknown command '{cmd}'")


def cmsh_category(args: list[str], s: dict) -> int:
    cmd = args[0] if args else "list"
    if cmd == "list":
        print(table(["NAME", "SOFTWAREIMAGE", "NODES"],
                    [[c["name"], c["image"], c["nodes"]]
                     for c in BCM_CATEGORIES]))
        return 0
    if cmd == "use":
        if len(args) < 2: return err("usage: category use <name>")
        info(f"[cluster->category[{args[1]}]]"); return 0
    return err(f"cmsh category: unknown '{cmd}'")


def cmsh_image(args: list[str], s: dict) -> int:
    cmd = args[0] if args else "list"
    if cmd == "list":
        print(table(["NAME", "KERNEL", "SIZE_GB", "CATEGORY"],
                    [[i["name"], i["kernel"], i["size_gb"], i["category"]]
                     for i in BCM_IMAGES]))
        return 0
    if cmd == "clone":
        if len(args) < 3: return err("usage: softwareimage clone <src> <dst>")
        info(f"Image '{args[2]}' created (cloned from '{args[1]}')."); return 0
    if cmd == "update":
        info("Image update queued.  ~12 minutes typical."); return 0
    if cmd == "commit":
        info("Image changes committed."); return 0
    return err(f"cmsh softwareimage: unknown '{cmd}'")


def cmsh_monitoring(args: list[str], s: dict) -> int:
    cmd = args[0] if args else "metrics"
    if cmd == "metrics":
        metric = args[1] if len(args) > 1 else "gpu_util"
        rows = []
        for n in s["nodes"][:8]:
            v = (random.uniform(60, 90) if n["state"] != "drain" else 0.0)
            rows.append([n["name"], metric, f"{v:.1f}", NOW()])
        print(table(["DEVICE", "METRIC", "VALUE", "TIMESTAMP"], rows))
        return 0
    if cmd == "alerts":
        info(table(
            ["ALERT_ID", "SEVERITY", "DEVICE",   "MESSAGE"],
            [["psu-007", "CRITICAL", "node-007", "12V rail 11.55V — under spec"],
             ["ecc-007", "CRITICAL", "node-007", "Uncorrectable ECC on GPU 0"],
             ["fec-12",  "WARNING",  "tor-01",   "FEC corrections > 1k/s on Eth1/12"]]
        ))
        return 0
    if cmd == "thresholds":
        info("METRIC          WARN     CRITICAL")
        info("gpu_temp_c      82       88")
        info("psu_12v         11.8     11.6")
        info("ecc_dbe         >0       >0")
        return 0
    return err(f"cmsh monitoring: unknown '{cmd}'")


def cmsh_wlm(args: list[str], s: dict) -> int:
    cmd = args[0] if args else "status"
    if cmd in ("status", "list"):
        running = sum(1 for j in s["slurm_jobs"] if j["state"] == "RUNNING")
        pending = sum(1 for j in s["slurm_jobs"] if j["state"] == "PENDING")
        info(table(
            ["WLM", "STATUS", "JOBS_RUNNING", "JOBS_PENDING"],
            [["slurm", "active", running, pending]]
        ))
        return 0
    return err(f"cmsh wlm: unknown '{cmd}'")


def cmsh_main(args: list[str], s: dict) -> int:
    info(f"[cluster]   phoenix-prod-1   BCM 10.24.6   "
         f"nodes={len(s['nodes'])}   uptime=12d"); return 0


# cm-* utilities (top-level binaries)
def cmd_cm_info(args: list[str], s: dict) -> int:
    info("Cluster:        phoenix-prod-1")
    info("BCM Version:    10.24.6")
    info("Head node:      node-000")
    info(f"Node count:     {len(s['nodes'])}   "
         f"(1 head, {len(s['nodes'])-1} compute)")
    info("Workload mgr:   slurm 23.11.x")
    info("GPU:            128 × NVIDIA H100 SXM5")
    info("Storage:        VAST C-Series, NFS/GDS")
    return 0


def cmd_cm_version(args: list[str], s: dict) -> int:
    info("Bright Cluster Manager 10.24.6")
    info("cmd version 10.24.6  (build 4321, 2026-03-12)")
    return 0


def cmd_cmha_status(args: list[str], s: dict) -> int:
    info("Cluster HA Status:    OK")
    info("Active head:          node-000  (master)")
    info("Passive head:         node-000-ha  (standby)")
    info("Last failover:        never")
    info("DRBD replication:     UpToDate / UpToDate")
    return 0


def cmd_mhcheck(args: list[str], s: dict) -> int:
    info("Running BCM cluster health checks ...")
    checks = [("DNS resolution", "OK"),
              ("LDAP/auth", "OK"),
              ("cmd daemon", "OK"),
              ("slurmctld", "OK"),
              ("node-007 PSU voltage", "FAIL  (11.55V < 11.6V threshold)"),
              ("Storage NFS mounts", "OK"),
              ("Time sync (chrony)", "OK")]
    for name, result in checks:
        tag = "[ OK  ]" if result == "OK" else "[ FAIL]"
        info(f"{tag} {name}: {result}")
    info(f"\n{sum(1 for _,r in checks if r=='OK')}/{len(checks)} checks passed")
    return 0


def cmd_ndlist(args: list[str], s: dict) -> int:
    rows = []
    for n in s["nodes"]:
        idx = n["name"].split("-")[1]
        rows.append([n["name"], "10.0.0." + idx,
                     f"00:25:90:1a:bc:{idx}",
                     "INSTALLED" if n["state"] != "drain" else "FAIL"])
    print(table(["NAME", "IP", "MAC", "STATUS"], rows))
    return 0


def cmd_node_installer_status(args: list[str], s: dict) -> int:
    rows = [[n["name"], "INSTALLED", "12d ago",
             ("default-image" if n["name"] == "node-000"
              else "h100-cuda12.4-image")]
            for n in s["nodes"]]
    print(table(["NODE", "STATE", "LAST_INSTALL", "IMAGE"], rows))
    return 0


# ===========================================================================
# Slurm extensions — srun / salloc / sshare / sprio / seff / sreport / sacctmgr
# ===========================================================================
def cmd_srun(args: list[str], s: dict) -> int:
    flags, pos = parse_flags(args, {"-N": "nodes", "-n": "ntasks"})
    nodes = int(flags.get("nodes", 1))
    cmd = " ".join(pos) if pos else "hostname"
    info(f"srun: launching '{cmd}' on {nodes} node(s) "
         f"with {flags.get('gres','gpu:8')} ...")
    if cmd == "hostname":
        for i in range(min(nodes, 4)):
            info(f"node-{i+1:03d}")
    elif "nccl" in cmd or "all_reduce" in cmd:
        info("# nThread 1 nGpus 8 minBytes 8 maxBytes 8589934592 step: 2(factor)")
        info("#       size  algbw  busbw   error    time")
        info("           8B   0.05   0.04    0e+00  142us")
        info("       1.05GB  85.40  85.39    0e+00   24ms")
        info("       8.39GB  87.21  87.18    0e+00   192ms")
    return 0


def cmd_salloc(args: list[str], s: dict) -> int:
    jid = s["next_jobid"]; s["next_jobid"] += 1
    save_state(s)
    info(f"salloc: Granted job allocation {jid}")
    info(f"salloc: Nodes node-001 are ready for job")
    info("(simulated — type 'exit' to release in real Slurm)")
    return 0


def cmd_sshare(args: list[str], s: dict) -> int:
    rows = []
    for p in s["projects"]:
        used_pct = p["gpu_used"] / max(1, p["gpu_quota"])
        rows.append(["root", p["name"], 1, f"{p['fairshare']:.4f}",
                     p["gpu_used"] * 100, f"{used_pct:.4f}"])
    print(table(
        ["Account", "User", "RawShares", "NormShares",
         "RawUsage", "EffectvUsage"], rows))
    return 0


def cmd_sprio(args: list[str], s: dict) -> int:
    rows = []
    for j in s["slurm_jobs"]:
        if j["state"] != "PENDING": continue
        rows.append([j["jobid"], j["partition"], j["user"],
                     random.randint(1000, 9999), 0,
                     random.randint(50, 500),
                     random.randint(100, 800),
                     random.randint(0, 200)])
    if not rows:
        info("(no pending jobs — sprio output is empty)")
        return 0
    print(table(["JOBID", "PARTITION", "USER", "PRIORITY",
                 "AGE", "FAIRSHARE", "JOBSIZE", "QOS"], rows))
    return 0


def cmd_seff(args: list[str], s: dict) -> int:
    if not args: return err("usage: seff <jobid>")
    info(f"Job ID: {args[0]}")
    info("Cluster: phoenix-prod-1")
    info("User/Group: alice/staff")
    info("State: COMPLETED (exit code 0)")
    info("Cores: 192")
    info(f"CPU Utilized: 02:14:08")
    cpu_eff = round(random.uniform(60, 95), 2)
    info(f"CPU Efficiency: {cpu_eff}% of 02:30:00 core-walltime")
    info("Job Wall-clock time: 02:30:00")
    mem = random.randint(80, 240)
    mem_eff = round(mem / 1024 * 100, 2)
    info(f"Memory Utilized: {mem} GB")
    info(f"Memory Efficiency: {mem_eff}% of 1.00 TB")
    return 0


def cmd_sreport(args: list[str], s: dict) -> int:
    info("--------------------------------------------------------")
    info("Cluster Utilization 2026-04-01T00:00:00 - 2026-04-30T23:59:59")
    info("--------------------------------------------------------")
    print(table(
        ["Cluster", "Allocated", "Down", "Idle", "Reserved", "Reported"],
        [["phoenix-prod-1", "73.2%", "2.4%", "18.9%", "1.0%", "100.0%"]]
    ))
    return 0


def cmd_sacctmgr(args: list[str], s: dict) -> int:
    if not args:
        return err("usage: sacctmgr {list|add|modify} {account|user|qos|cluster}")
    if args[0] == "list":
        what = args[1] if len(args) > 1 else ""
        if what in ("account", "accounts"):
            print(table(["Account", "Description", "Organization"],
                        [[d, f"{d} dept", "Acme"] for d in s["departments"]]))
            return 0
        if what in ("user", "users"):
            users = [("alice", "ml-research"), ("bob", "vision-team"),
                     ("carol", "nlp-team"), ("dave", "ml-research")]
            print(table(["User", "DefAccount", "AdminLevel"],
                        [[u, a, "None"] for u, a in users]))
            return 0
        if what == "qos":
            print(table(
                ["Name", "Priority", "MaxJobs", "MaxWall"],
                [["normal", "100",  "100", "48:00:00"],
                 ["high",   "500",  "20",  "12:00:00"],
                 ["debug",  "1000", "5",   "00:30:00"]]))
            return 0
        if what == "cluster":
            print(table(["Cluster", "ControlHost"],
                        [["phoenix-prod-1", "node-000"]]))
            return 0
    info(f"(sacctmgr: '{' '.join(args)}' accepted — simulator ack)")
    return 0


# ===========================================================================
# kubectl extensions — scale, rollout, label, taint, run, create, port-forward
# ===========================================================================
def kubectl_scale(args: list[str], s: dict) -> int:
    flags, pos = parse_flags(args, {"-n": "namespace"})
    if not pos: return err("usage: kubectl scale deployment <name> --replicas=N")
    target = pos[-1]
    info(f"deployment.apps/{target} scaled")
    return 0


def kubectl_rollout(args: list[str], s: dict) -> int:
    if not args:
        return err("usage: kubectl rollout {status|restart|history|undo} ...")
    sub = args[0]
    target = args[-1] if len(args) > 1 else "<name>"
    if sub == "status":
        info(f'deployment "{target}" successfully rolled out'); return 0
    if sub == "restart":
        info(f"deployment.apps/{target} restarted"); return 0
    if sub == "history":
        print(table(["REVISION", "CHANGE-CAUSE"],
                    [[1, "<none>"], [2, "kubectl set image ..."]]))
        return 0
    if sub == "undo":
        info(f"deployment.apps/{target} rolled back"); return 0
    return err(f"rollout: unknown subcommand '{sub}'")


def kubectl_label(args: list[str], s: dict) -> int:
    if len(args) < 3: return err("usage: kubectl label <resource> <name> key=value")
    info(f"{args[0]}/{args[1]} labeled"); return 0


def kubectl_taint(args: list[str], s: dict) -> int:
    if len(args) < 3: return err("usage: kubectl taint nodes <name> key=value:effect")
    info(f"node/{args[1]} tainted"); return 0


def kubectl_run(args: list[str], s: dict) -> int:
    flags, pos = parse_flags(args, {"-n": "namespace", "--image": "image"})
    name = pos[0] if pos else "test-pod"
    s["pods"].append({
        "name": name + "-abc", "namespace": flags.get("namespace", "default"),
        "ready": "1/1", "status": "Running", "restarts": 0, "age": "0s",
        "node": "node-001", "ip": "10.244.1.99",
    })
    save_state(s)
    info(f"pod/{name}-abc created"); return 0


def kubectl_create(args: list[str], s: dict) -> int:
    if not args:
        return err("usage: kubectl create {namespace|deployment|configmap|secret} <name>")
    kind = args[0]
    name = args[1] if len(args) > 1 else "<name>"
    if kind == "namespace":   info(f"namespace/{name} created"); return 0
    if kind == "deployment":  info(f"deployment.apps/{name} created"); return 0
    if kind == "configmap":   info(f"configmap/{name} created"); return 0
    if kind == "secret":      info(f"secret/{name} created"); return 0
    info(f"{kind}/{name} created"); return 0


def kubectl_port_forward(args: list[str], s: dict) -> int:
    if len(args) < 2:
        return err("usage: kubectl port-forward <pod> <local:remote>")
    spec = args[1]
    local = spec.split(":")[0]
    remote = spec.split(":")[-1]
    info(f"Forwarding from 127.0.0.1:{local} -> {remote}")
    info(f"Forwarding from [::1]:{local} -> {remote}")
    info("(simulated — would block until Ctrl-C)")
    return 0


def kubectl_edit(args: list[str], s: dict) -> int:
    if len(args) < 2: return err("usage: kubectl edit <resource> <name>")
    info(f'{args[0]}/{args[1]} edited (no changes — simulator does not open $EDITOR)')
    return 0


def kubectl_explain(args: list[str], s: dict) -> int:
    if not args: return err("usage: kubectl explain <resource>")
    info(f"KIND:     {args[0].title()}")
    info("VERSION:  v1\n")
    info("DESCRIPTION:")
    info(f"  {args[0]} is a Kubernetes resource.  See real `kubectl explain` for details.")
    return 0


# ===========================================================================
# Helm
# ===========================================================================
HELM_RELEASES = [
    {"name": "gpu-operator", "namespace": "gpu-operator",
     "revision": 5, "updated": "2026-04-15", "status": "deployed",
     "chart": "gpu-operator-v24.3.0", "app_version": "v24.3.0"},
    {"name": "runai-cluster", "namespace": "runai",
     "revision": 12, "updated": "2026-04-22", "status": "deployed",
     "chart": "runai-cluster-2.18.5", "app_version": "2.18.5"},
    {"name": "kube-prometheus", "namespace": "monitoring",
     "revision": 3, "updated": "2026-03-30", "status": "deployed",
     "chart": "kube-prometheus-stack-58.0.0", "app_version": "v0.73"},
    {"name": "vast-csi", "namespace": "vast-system",
     "revision": 2, "updated": "2026-02-10", "status": "deployed",
     "chart": "vast-csi-2.4.1", "app_version": "2.4.1"},
]


def cmd_helm(args: list[str], s: dict) -> int:
    if not args or args[0] in ("-h", "--help", "help"):
        info("helm {list|install|upgrade|status|repo|uninstall|history|rollback}")
        return 0
    sub = args[0]
    if sub == "list":
        flags, _ = parse_flags(args[1:], {"-n": "namespace", "-A": "all"})
        rels = HELM_RELEASES
        if not flags.get("all") and flags.get("namespace"):
            rels = [r for r in rels if r["namespace"] == flags["namespace"]]
        rows = [[r["name"], r["namespace"], r["revision"], r["updated"],
                 r["status"], r["chart"], r["app_version"]] for r in rels]
        print(table(["NAME", "NAMESPACE", "REVISION", "UPDATED",
                     "STATUS", "CHART", "APP VERSION"], rows))
        return 0
    if sub == "install":
        if len(args) < 3: return err("usage: helm install <release> <chart>")
        info(f"NAME: {args[1]}")
        info(f"LAST DEPLOYED: {NOW()}")
        info("NAMESPACE: default")
        info("STATUS: deployed")
        info("REVISION: 1")
        info(f"CHART: {args[2]}")
        return 0
    if sub == "upgrade":
        if len(args) < 3: return err("usage: helm upgrade <release> <chart>")
        info(f'Release "{args[1]}" has been upgraded. Happy Helming!')
        return 0
    if sub == "status":
        rel = next((r for r in HELM_RELEASES
                    if r["name"] == (args[1] if len(args) > 1 else "")), None)
        if not rel: return err("release not found")
        info(f"NAME: {rel['name']}")
        info(f"LAST DEPLOYED: {rel['updated']}")
        info(f"STATUS: {rel['status']}")
        info(f"REVISION: {rel['revision']}")
        return 0
    if sub == "repo":
        op = args[1] if len(args) > 1 else "list"
        if op == "list":
            info("NAME      URL")
            info("nvidia    https://helm.ngc.nvidia.com/nvidia")
            info("runai     https://run-ai-charts.storage.googleapis.com")
            info("prom      https://prometheus-community.github.io/helm-charts")
        elif op == "add":
            info(f"\"{args[2] if len(args)>2 else 'repo'}\" has been added to your repositories")
        elif op == "update":
            info("Hang tight while we grab the latest from your chart repositories...")
            info("...Successfully got an update from all chart repositories")
        return 0
    if sub == "uninstall":
        if len(args) < 2: return err("usage: helm uninstall <release>")
        info(f"release \"{args[1]}\" uninstalled"); return 0
    if sub == "history":
        rows = [[1, "2026-04-01", "deployed", "Initial install"],
                [2, "2026-04-15", "deployed", "Upgrade to v24.3.0"]]
        print(table(["REVISION", "UPDATED", "STATUS", "DESCRIPTION"], rows))
        return 0
    if sub == "rollback":
        info(f"Rollback was a success! Happy Helming!"); return 0
    return err(f"helm: unknown subcommand '{sub}'")


# ===========================================================================
# Run:ai extensions — workspaces, departments, MPI submit, project admin
# ===========================================================================
def cmd_runai_extras(args: list[str], s: dict) -> Optional[int]:
    """Returns int rc if it handled the command, else None."""
    if not args: return None
    sub = args[0]; rest = args[1:]

    # runai list workspaces
    if sub == "list" and rest and rest[0] == "workspaces":
        ws = [j for j in s["runai_jobs"] if j["type"] == "Interactive"]
        rows = [[j["name"], j["project"], j["user"], j["status"],
                 j["gpus_requested"], j.get("node", "<pending>"),
                 f"{j['age_min']}m"] for j in ws]
        print(table(["NAME", "PROJECT", "USER", "STATUS",
                     "GPUs", "NODE", "AGE"], rows))
        return 0

    # runai submit-mpi
    if sub == "submit-mpi":
        if not rest:
            return err("usage: runai submit-mpi <name> -p <proj> -g <n> "
                       "--workers <w> -- <cmd>")
        name = rest[0]
        flags, pos = parse_flags(rest[1:], {"-p": "project", "-g": "gpu",
                                            "-i": "image"})
        proj = flags.get("project") or s["current_project"]
        workers = int(flags.get("workers", 2))
        gpus = int(flags.get("gpu", 1)) * workers
        cmd_str = " ".join(pos) if pos else "mpirun -np 16 python train.py"
        s["runai_jobs"].append({
            "name": name, "project": proj, "user": s["user"],
            "type": "MPI", "status": "Running",
            "gpus_requested": gpus,
            "image": flags.get("image", "nvcr.io/nvidia/pytorch:24.03-py3"),
            "node": f"node-001..node-{workers:03d}", "age_min": 0,
            "command": cmd_str,
        })
        save_state(s)
        info(f"INFO[{NOW()}] MPI job '{name}' submitted "
             f"({workers} workers × {gpus // workers} GPUs)")
        return 0

    # runai update project <name> --gpu-quota N
    if sub == "update" and rest and rest[0] == "project":
        if len(rest) < 2: return err("usage: runai update project <name> --gpu-quota N")
        name = rest[1]
        flags, _ = parse_flags(rest[2:], {})
        proj = next((p for p in s["projects"] if p["name"] == name), None)
        if not proj: return err(f"project '{name}' not found")
        if "gpu-quota" in flags:
            proj["gpu_quota"] = int(flags["gpu-quota"])
        if "gpu-guarantee" in flags:
            proj["guarantee"] = int(flags["gpu-guarantee"])
        save_state(s)
        info(f"Project '{name}' updated.")
        return 0

    # runai create project / department
    if sub == "create" and rest:
        kind = rest[0]
        if kind == "project" and len(rest) > 1:
            s["projects"].append({"name": rest[1],
                                  "department": "applied-ai",
                                  "gpu_quota": 8, "gpu_used": 0,
                                  "fairshare": 0.10, "guarantee": 0})
            save_state(s)
            info(f"Project '{rest[1]}' created."); return 0
        if kind == "department" and len(rest) > 1:
            s["departments"].append(rest[1])
            save_state(s)
            info(f"Department '{rest[1]}' created."); return 0

    # runai delete project
    if sub == "delete" and rest and rest[0] == "project" and len(rest) > 1:
        before = len(s["projects"])
        s["projects"] = [p for p in s["projects"] if p["name"] != rest[1]]
        if len(s["projects"]) == before:
            return err(f"project '{rest[1]}' not found")
        save_state(s)
        info(f"Project '{rest[1]}' deleted."); return 0

    # runai exec / runai port-forward
    if sub == "exec" and rest:
        info(f"[simulated] would exec into Run:ai job '{rest[0]}'"); return 0
    if sub == "port-forward" and len(rest) >= 2:
        info(f"Forwarding from 127.0.0.1:{rest[1].split(':')[0]} "
             f"-> Run:ai job {rest[0]}"); return 0

    return None  # let the main cmd_runai handle it


# ===========================================================================
# GPU / Linux CLI tools — dcgmi, nvidia-smi, nvidia-bug-report,
#                         dmesg, ipmitool, mlxlink, mlxfwmanager, mst
# ===========================================================================
def cmd_dcgmi(args: list[str], s: dict) -> int:
    if not args:
        return err("usage: dcgmi {diag|health|profile|stats|group|cluster|discovery}")
    sub = args[0]
    if sub == "diag":
        flags, _ = parse_flags(args[1:], {"-r": "run", "-i": "index"})
        level = int(flags.get("run", 1))
        info(f"Successfully ran diagnostic for group: {flags.get('index','all GPUs')}")
        info(f"Diagnostic level: {level}")
        info("+---------------------------+----------------------------------+")
        info("| Diagnostic                | Result                           |")
        info("+===========================+==================================+")
        levels = {1: ["Software", "Deployment"],
                  2: ["Software", "Deployment", "Integration"],
                  3: ["Software", "Deployment", "Integration", "Hardware",
                      "Stress (Targeted)", "Stress (Memory)"],
                  4: ["Software", "Deployment", "Integration", "Hardware",
                      "Stress (Targeted)", "Stress (Memory)",
                      "Stress (Memtest Long)", "Stress (Diagnostic Long)"]}
        for t in levels.get(level, levels[3]):
            outcome = "Pass"
            # echo our fleet's known faults
            if t.startswith("Stress (Memory") and flags.get("index") in (None, "0"):
                outcome = "Pass"
            info(f"| {t:<25} | {outcome:<32} |")
        info("+---------------------------+----------------------------------+")
        return 0
    if sub == "health":
        info("Group of GPUs has health: Healthy")
        info("+----+----------+--------+----------+")
        info("| GPU| Health   | Watch  | Reason   |")
        info("+====+==========+========+==========+")
        for i in range(8):
            status = "Healthy" if i != 0 else "Warn"
            reason = "-" if i != 0 else "Memory temp 84C"
            info(f"|  {i} | {status:<9}| All    | {reason:<8}|")
        info("+----+----------+--------+----------+")
        return 0
    if sub == "profile":
        op = args[1] if len(args) > 1 else "--list"
        if op == "--pause":  info("Profiling paused."); return 0
        if op == "--resume": info("Profiling resumed."); return 0
        info("Profiling state: ACTIVE"); return 0
    if sub == "stats":
        info("All statistics from group 0:")
        info("Executed PIDs:                4")
        info("Avg GPU Util:                 78%")
        info("Avg Memory Util:              52%")
        info("Avg Power Draw:               520 W")
        info("Avg SM Clock:                 1980 MHz")
        return 0
    if sub == "group":
        info("Group ID  | Name      | Entities")
        info("0         | default   | 0,1,2,3,4,5,6,7")
        return 0
    if sub == "cluster":
        info("DCGM cluster status: 16 hosts, 128 GPUs healthy")
        return 0
    if sub == "discovery":
        for i in range(8):
            info(f"GPU {i}  Name: NVIDIA H100 SXM5  PCI: 0000:{i:02x}:00.0")
        return 0
    return err(f"dcgmi: unknown subcommand '{sub}'")


def cmd_nvidia_smi(args: list[str], s: dict) -> int:
    if "topo" in args or "--topo" in str(args):
        # m option produces a matrix
        info("        GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7  CPU Affinity")
        for i in range(8):
            row = [f"GPU{i}"]
            for j in range(8):
                row.append("X" if i == j else "NV18")
            row.append("0-95" if i < 4 else "96-191")
            info("\t".join(row))
        info("\nLegend:  X = Self  NV18 = NVLink (18 lanes)  PIX = PCIe")
        return 0
    if "nvlink" in args:
        for i in range(4):
            info(f"GPU {i}: NVLink Speed 25.781 GB/s per lane, 18 active links")
            info(f"      Link 0: enabled, 25.781 GB/s, 0 errors")
        return 0
    if "--query-gpu" in str(args) or "-q" in args:
        for i in range(4):
            info(f"{i}, GPU-fake-{i}, NVIDIA H100 80GB HBM3, "
                 f"{random.randint(60,80)}, {random.randint(40,90)}, "
                 f"{random.randint(40000,72000)}, 81920, "
                 f"{random.randint(300,650)}.0, 0, 0, 0, 0")
        return 0
    if "--list-gpus" in args or "-L" in args:
        for i in range(8):
            info(f"GPU {i}: NVIDIA H100 80GB HBM3 (UUID: GPU-fake-{i:08x})")
        return 0
    print_fake_nvidia_smi("local")
    return 0


def cmd_nvidia_bug_report(args: list[str], s: dict) -> int:
    info("nvidia-bug-report.sh: Running ...")
    for step in ["Collecting kernel logs",
                 "Collecting nvidia-smi output",
                 "Collecting dmesg output",
                 "Collecting NVRM Xid history",
                 "Collecting PCIe topology",
                 "Generating output archive"]:
        info(f"nvidia-bug-report.sh: {step} ...")
    out = f"/tmp/nvidia-bug-report-{int(__import__('time').time())}.gz"
    info(f"nvidia-bug-report.sh: Saved output to {out}")
    return 0


def cmd_dmesg(args: list[str], s: dict) -> int:
    """Real-shape kernel log output, with XID and NIC events."""
    grep = None
    for i, a in enumerate(args):
        if a == "|" and i + 2 < len(args) and args[i+1] == "grep":
            grep = args[i+2].lower(); break
    lines = [
        "[ 2391.234] NVRM: GPU at PCI:0000:1b:00: GPU-3a51",
        "[ 2400.118] NVRM: Xid (PCI:0000:1b:00): 79, pid='<unknown>', "
            "GPU has fallen off the bus.",
        "[ 2401.000] NVRM: A GPU crash dump has been created.",
        "[ 3192.567] NVRM: Xid (PCI:0000:af:00): 63, page retirement on GPU 5.",
        "[ 3192.989] mlx5_core 0000:c1:00.0: FW initializing, may take time",
        "[ 4031.221] mlx5_core 0000:c1:00.0: Link state changed to up",
        "[ 5099.812] nvidia-uvm: Loaded UVM driver version 550.54.14",
        "[ 6022.144] systemd[1]: Started slurmd Slurm node daemon",
    ]
    for ln in lines:
        if grep and grep not in ln.lower(): continue
        info(ln)
    return 0


def cmd_ipmitool(args: list[str], s: dict) -> int:
    if "sdr" in args:
        rows = [
            ["12V_PSU1",  "12.02", "Volts", "OK"],
            ["12V_PSU2",  "11.98", "Volts", "OK"],
            ["FAN1_RPM",  "9450",  "RPM",   "OK"],
            ["FAN2_RPM",  "9510",  "RPM",   "OK"],
            ["CPU1_TEMP", "52",    "C",     "OK"],
            ["GPU0_TEMP", "78",    "C",     "OK"],
            ["GPU1_TEMP", "76",    "C",     "OK"],
        ]
        print(table(["SENSOR", "VALUE", "UNIT", "STATUS"], rows)); return 0
    if "sel" in args:
        info("1 | 04/30/2026 | Power Unit | PSU 1 12V rail below threshold")
        info("2 | 04/30/2026 | Memory     | DIMM B1: ECC error corrected")
        info("3 | 04/29/2026 | Processor  | CPU1 thermal trip event")
        return 0
    if "chassis" in args:
        op = args[1] if len(args) > 1 else "status"
        if op == "status":
            info("System Power: on")
            info("Power Restore Policy: previous"); return 0
        if op == "power":
            info(f"Chassis Power Control: {args[2] if len(args)>2 else 'status'}")
            return 0
    if "lan" in args:
        info("IP Address Source : Static Address")
        info("IP Address        : 10.0.7.7")
        info("MAC Address       : 00:25:90:1a:bc:07"); return 0
    info(f"(ipmitool: '{' '.join(args)}' accepted — simulator ack)")
    return 0


def cmd_mlxlink(args: list[str], s: dict) -> int:
    info("Operational Info")
    info("----------------")
    info("State                              : Active")
    info("Physical state                     : LinkUp")
    info("Speed                              : 200G_4X")
    info("Width                              : 4x")
    info("FEC                                : standard_rs")
    info("Loopback Mode                      : No Loopback")
    info("Auto Negotiation                   : ON")
    info("\nTroubleshooting Info")
    info("--------------------")
    info("Status Opcode                      : 0  (No issue)")
    info("Effective Physical BER             : 1e-15")
    return 0


def cmd_mlxfwmanager(args: list[str], s: dict) -> int:
    info("Querying Mellanox devices firmware ...\n")
    info("Device #1:")
    info("  Device Type:       ConnectX7")
    info("  Part Number:       MCX755106AS-HEAT_Ax")
    info("  Description:       ConnectX-7 dual-port 200GbE / NDR200")
    info("  PSID:              MT_0000000838")
    info("  PCI Device Name:   /dev/mst/mt4129_pciconf0")
    info("  Base GUID:         0c42a103009a3df4")
    info("  Base MAC:          0c42a19a3df4")
    info("  Versions:          Current        Available")
    info("     FW             28.39.1002      28.39.1002")
    info("     PXE            3.7.0500        3.7.0500")
    info("  Status:            Up to date")
    return 0


def cmd_mst(args: list[str], s: dict) -> int:
    if "status" in args:
        info("MST modules: started")
        info("MST PCI configuration module loaded")
        info("MST devices:")
        info("  /dev/mst/mt4129_pciconf0  - PCI configuration cycles access for ConnectX-7")
        info("                              domain:bus:dev.fn=0000:c1:00.0")
        return 0
    if "start" in args: info("Starting MST (Mellanox Software Tools) ...  done"); return 0
    if "stop"  in args: info("Stopping MST ...  done"); return 0
    info(f"(mst: '{' '.join(args)}' accepted)")
    return 0


# ===========================================================================
# NVIDIA Lab outputs — module, cm-wlm-setup, cm-kubernetes-setup,
#                      cat gpu.yaml, gpu-pod kubectl flow, su, interactive cmsh
# ===========================================================================
LAB_NVIDIA_SMI = """\
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.86.15              Driver Version: 570.86.15      CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA H100 NVL                Off |   00000000:B5:00.0 Off |                    0 |
| N/A   51C    P0             124W / 400W |       1MiB /  95830MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+"""


LAB_MODULE_AVAILABLE = """\
------------------------------------------------- /cm/local/modulefiles -------------------------------------------------
bcm-post-install/10.0      cluster-tools/10.0  cm-scale/cm-scale.module  cmjob              dot              ipmitool/1.8.19                 mariadb-libs  modules   python3         sedutil/1.16.0       use.own
bcm-superpod-network/10.0  cm-bios-tools       cm-setup/10.0             cmsh               freeipmi/1.6.14  kubernetes/default/1.30.10-1.1  module-git    null      python39        shared
boost/1.81.0               cm-image/10.0       cmd                       containerd/1.7.21  gcc/13.1.0       luajit                          module-info   openldap  rocm-smi/4.3.0  slurm/slurm/23.02.8
------------------------------------------------ /cm/shared/modulefiles -------------------------------------------------------
cm-pmix3/3.1.7  default-environment  hdf5_18/1.8.21  hwloc/1.11.13  iperf/3.17.1           openblas/dynamic/0.3.18  openmpi4/gcc/4.1.5
cm-pmix4/4.1.3  gdb/13.1             hpl/2.3         hwloc2/2.8.0   mvapich2/gcc/64/2.3.7  openmpi/gcc/64/4.1.5     ucx/1.10.1"""


LAB_CM_WLM_SETUP = """\
[GUI Simulator: Workload Manager Setup Wizard Launched]
... Navigating Setup (Step By Step)
... Selected Workload Manager: Slurm
... Selected Slurm Client Role: [X] slurm-01, [X] slurm-02, [X] slurm-03
... Selected GPU Client Role:   [X] slurm-04
... Saving configuration to /root/cm-wlm-setup.conf
[GUI Simulator: Deployment Complete!]"""


LAB_CM_K8S_SETUP = """\
[GUI Simulator: Kubernetes Setup Wizard Launched]
... Selected Kubernetes v1.30
... Configured Network: Service (10.150.0.0/16), Pod (172.29.0.0/16)
... Selected Master Nodes: [X] k8s-control-plane-01, [X] k8s-control-plane-02, [X] k8s-control-plane-03
... Selected Worker Nodes: [X] k8s-worker
... Selected Network Plugin: Calico
... Selected Operators: NVIDIA GPU Operator (v24.9.0), Prometheus, Dashboard
... Saving configuration to /root/cm-kubernetes-setup.conf
[GUI Simulator: Deployment Complete! (Approx 15 minutes elapsed)]"""


LAB_CM_K8S_ADD_USER = """\
…[output]
Progress: 100/100
################### Finished execution for 'Kubernetes Setup', status: completed"""


LAB_CM_K8S_LIST_USERS = """\
Connecting to CMDaemon
Executing 3 stages
################### Starting execution for 'Kubernetes Setup'
  - kubernetes
  - docker
## Progress: 0
#### stage: kubernetes: Get Kube Cluster
## Progress: 33
#### stage: kubernetes: Check Permissions User Chart
## Progress: 66
#### stage: kubernetes: Deploy List users
USERNAME____________: NAMESPACES______________________________
k8suser             : k8suser-restricted
## Progress: 100
Took:     00:00 min.
Progress: 100/100
################### Finished execution for 'Kubernetes Setup', status: completed
Kubernetes Setup finished!"""


LAB_GPU_YAML = """\
apiVersion: v1
kind: Pod
metadata:
  name: gpu-pod
spec:
  restartPolicy: Never
  containers:
  - name: cuda-container
    image: nvcr.io/nvidia/cuda:12.6.2-base-ubuntu22.04
    command: ["nvidia-smi"]
    resources:
      limits:
        nvidia.com/gpu: 1"""


LAB_GPU_POD_LOGS = """\
Wed Feb  19 19:24:41 2025
+-------------------------------------------------------------------------------+
| NVIDIA-SMI 550.90.07      Driver Version: 550.90.07    CUDA Version: 12.6     |
|--------------------------------+---------------------+------------------------+
| GPU  Name        Persistence-M | Bus-Id     Disp.A   |   Volatile Uncorr. ECC |
| Fan  Temp   Perf Pwr:Usage/Cap |      Memory-Usage   |    GPU-Util Compute M. |
|                                |                     |                 MIG M. |
|================================+=====================+========================|
|   0  NVIDIA L40S-4C        On  |00000000:03:00.0 Off |                      0 |
| N/A   N/A    P8    N/A /  N/A  |  1MiB /   4096MiB   | 0%             Default |
|                                |                     |                    N/A |
+--------------------------------+------------------------+---------------------+

+-------------------------------------------------------------------------------+
| Processes:                                                                    |
|  GPU   GI   CI        PID   Type   Process name                    GPU Memory |
|        ID   ID                                                     Usage      |
|===============================================================================|
|  No running processes found                                                   |
+-------------------------------------------------------------------------------+"""


LAB_K8S_GET_NODES = """\
NAME                   STATUS   ROLES                  AGE     VERSION
k8s-control-plane-01   Ready    control-plane,master   7m50s   v1.30.10
k8s-control-plane-02   Ready    control-plane,master   8m51s   v1.30.10
k8s-control-plane-03   Ready    control-plane,master   7m50s   v1.30.10
k8s-worker-01          Ready    worker                 7m58s   v1.30.10"""


LAB_SINFO = ("PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST\n"
             "defq*        up   infinite      4   idle slurm-[01-04]")


# --- Module command --------------------------------------------------------
def cmd_module(args: list[str], s: dict) -> int:
    if not args:
        info("Usage: module {available|list|load|unload|show} [name]")
        return 0
    sub = args[0]
    if sub in ("available", "avail"):
        print(LAB_MODULE_AVAILABLE); return 0
    if sub == "list":
        info("Currently Loaded Modulefiles:")
        if s.get("modules_loaded"):
            for i, m in enumerate(s["modules_loaded"], 1):
                info(f"  {i}) {m}")
        else:
            info("  (none)")
        return 0
    if sub == "load":
        if len(args) < 2: return err("usage: module load <name>")
        s.setdefault("modules_loaded", []).append(args[1])
        save_state(s)
        return 0  # silent — like real `module load`
    if sub == "unload":
        if len(args) < 2: return err("usage: module unload <name>")
        s.setdefault("modules_loaded", [])
        if args[1] in s["modules_loaded"]:
            s["modules_loaded"].remove(args[1])
        save_state(s)
        return 0
    if sub == "show":
        if len(args) < 2: return err("usage: module show <name>")
        info(f"-------------------------------------------------------------------")
        info(f"/cm/local/modulefiles/{args[1]}:")
        info(f"prepend-path     PATH /cm/shared/apps/{args[1]}/bin")
        info(f"prepend-path     LD_LIBRARY_PATH /cm/shared/apps/{args[1]}/lib")
        info(f"-------------------------------------------------------------------")
        return 0
    return err(f"module: unknown subcommand '{sub}'")


# --- BCM setup wizards -----------------------------------------------------
def cmd_cm_wlm_setup(args: list[str], s: dict) -> int:
    print(LAB_CM_WLM_SETUP); return 0


def cmd_cm_kubernetes_setup(args: list[str], s: dict) -> int:
    if "--add-user" in args:
        # find the user name after --add-user
        idx = args.index("--add-user")
        user = args[idx + 1] if idx + 1 < len(args) else "k8suser"
        s.setdefault("k8s_users", [])
        if user not in s["k8s_users"]:
            s["k8s_users"].append(user)
        save_state(s)
        print(LAB_CM_K8S_ADD_USER); return 0
    if "--list-users" in args:
        print(LAB_CM_K8S_LIST_USERS); return 0
    print(LAB_CM_K8S_SETUP); return 0


# --- cat (only for the lab gpu.yaml) --------------------------------------
def cmd_cat(args: list[str], s: dict) -> int:
    if not args: return err("usage: cat <file>")
    f = args[0]
    if f == "/cm/shared/apps/lp/gpu.yaml":
        print(LAB_GPU_YAML); return 0
    if f.endswith(".yaml") or f.endswith(".yml"):
        info(f"# (simulator only stocks /cm/shared/apps/lp/gpu.yaml — "
             f"showing canonical pod manifest)")
        print(LAB_GPU_YAML); return 0
    return err(f"cat: {f}: No such file or directory")


# --- su (user switching, simulated) ---------------------------------------
def cmd_su(args: list[str], s: dict) -> int:
    if not args:
        return err("usage: su [-] <user>")
    target = args[-1]
    if target.startswith("-"):
        return err("usage: su - <user>")
    info("Creating ECDSA key for ssh")
    s["user"] = target
    save_state(s)
    info(f"(simulated — you are now '{target}'.  type 'su - root' or 'reset' to switch back)")
    return 0


# ===========================================================================
# Stateful interactive cmsh — replicates the NVIDIA lab navigation
# ===========================================================================
CMSH_VALID_TOP_MODES = {"device", "category", "softwareimage", "monitoring",
                        "user", "wlm", "main", "partition", "network"}

# Lab device layouts, switched by which 'commit' has happened
LAB_DEVICE_TABLE_DEFAULT = """\
Type                   Hostname (key)        MAC                Category           IP              Network        Status
---------------------- --------------------- ------------------ ------------------ --------------- -------------- ---------------
HeadNode               bcm                   00:50:56:93:4B:40                     10.141.255.254  internalnet    [   UP   ]
PhysicalNode           node001               4E:56:44:41:01:01  default            10.141.0.1      internalnet    [   UP   ]
PhysicalNode           node002               4E:56:44:41:01:02  default            10.141.0.2      internalnet    [   UP   ]
PhysicalNode           node003               4E:56:44:41:01:03  default            10.141.0.3      internalnet    [   UP   ]
PhysicalNode           node004               4E:56:44:41:01:04  default            10.141.0.4      internalnet    [   UP   ]"""

LAB_DEVICE_TABLE_SLURM = """\
Type                   Hostname (key)        MAC                Category           IP              Network        Status
---------------------- --------------------- ------------------ ------------------ --------------- -------------- ---------------
HeadNode               bcm                   00:50:56:93:4B:40                     10.141.255.254  internalnet    [   UP   ]
PhysicalNode           slurm-01              4E:56:44:41:01:01  slurm              10.141.0.1      internalnet    [   UP   ]
PhysicalNode           slurm-02              4E:56:44:41:01:02  slurm              10.141.0.2      internalnet    [   UP   ]
PhysicalNode           slurm-03              4E:56:44:41:01:03  slurm              10.141.0.3      internalnet    [   UP   ]
PhysicalNode           slurm-04              4E:56:44:41:01:04  slurm              10.141.0.4      internalnet    [   UP   ]"""

LAB_DEVICE_TABLE_K8S = """\
Type                   Hostname (key)        MAC                Category           IP              Network        Status
---------------------- --------------------- ------------------ ------------------ --------------- -------------- ---------------
HeadNode               bcm                   00:50:56:93:4B:40                     10.141.255.254  internalnet    [   UP   ]
PhysicalNode           k8s-control-plane-01  4E:56:44:41:01:01  k8s-control-plane  10.141.0.1      internalnet    [   UP   ]
PhysicalNode           k8s-control-plane-02  4E:56:44:41:01:02  k8s-control-plane  10.141.0.2      internalnet    [   UP   ]
PhysicalNode           k8s-control-plane-03  4E:56:44:41:01:03  k8s-control-plane  10.141.0.3      internalnet    [   UP   ]
PhysicalNode           k8s-worker-01         4E:56:44:41:01:04  k8s-worker         10.141.0.4      internalnet    [   UP   ]"""

LAB_IMAGE_TABLE_BASIC = """\
Name (key)               Path (key)                               Kernel version      Nodes
------------------------ ---------------------------------------- ------------------- --------
default-image            /cm/images/default-image                 5.15.0-113-generic  0
dgx-os-6.3-a100-image    /cm/images/dgx-os-6.3-a100-image         5.15.0-1063-nvidia  0
dgx-os-6.3-h100-image    /cm/images/dgx-os-6.3-h100-image         5.15.0-1063-nvidia  0"""

LAB_IMAGE_TABLE_AFTER_CLONE = """\
Name (key)               Path (key)                               Kernel version      Nodes
------------------------ ---------------------------------------- ------------------- --------
default-image            /cm/images/default-image                 5.15.0-113-generic  4
dgx-os-6.3-a100-image    /cm/images/dgx-os-6.3-a100-image         5.15.0-1063-nvidia  0
dgx-os-6.3-h100-image    /cm/images/dgx-os-6.3-h100-image         5.15.0-1063-nvidia  0
k8s-control-plane-image  /cm/images/k8s-control-plane-image       5.15.0-113-generic  0
k8s-worker-image         /cm/images/k8s-worker-image              5.15.0-113-generic  0"""

LAB_CATEGORY_TABLE_BASIC = """\
Name (key)               Software image           Nodes
------------------------ ------------------------ --------
default                  default-image            4
dgx-a100                 dgx-os-6.3-a100-image    0
dgx-h100                 dgx-os-6.3-h100-image    0"""

LAB_CATEGORY_TABLE_K8S = """\
Name (key)               Software image           Nodes
------------------------ ------------------------ --------
default                  default-image            4
dgx-a100                 dgx-os-6.3-a100-image    0
dgx-h100                 dgx-os-6.3-h100-image    0
k8s-control-plane        k8s-control-plane-image  0
k8s-worker               k8s-worker-image         0"""


def _cmsh_prompt(ctx: dict) -> str:
    """Build the dynamic cmsh prompt from current navigation context."""
    mode = ctx.get("mode")          # softwareimage | category | device | user | ...
    obj = ctx.get("obj")             # slurm-image, slurm, k8suser, ...
    dirty = ctx.get("dirty", False)
    if not mode:
        return "[bcm]% "
    if not obj:
        return f"[bcm->{mode}]% "
    star_mode = "*" if dirty else ""
    star_obj = "*" if dirty else ""
    return f"[bcm->{mode}{star_mode}[{obj}{star_obj}]]% "


def cmsh_interactive(s: dict) -> int:
    """Stateful nested cmsh REPL — tracks mode, sub-object, dirty flag."""
    ctx = {"mode": None, "obj": None, "dirty": False}
    info("Connected to bcm using BCM CMDaemon.  Type 'quit' or 'exit' to leave.")
    try:
        import readline  # noqa: F401
    except Exception:
        pass

    while True:
        try:
            raw = input(_cmsh_prompt(ctx)).strip()
        except (EOFError, KeyboardInterrupt):
            print(); return 0

        if not raw:
            continue
        if raw in ("quit", "exit"):
            # if inside a sub-object, drop one level; if at top, leave shell
            if ctx["obj"]:    ctx["obj"] = None; ctx["dirty"] = False; continue
            if ctx["mode"]:   ctx["mode"] = None; ctx["dirty"] = False; continue
            return 0

        toks = shlex.split(raw)
        head = toks[0]

        # --- Top-level: mode entry ---
        if not ctx["mode"] and head in CMSH_VALID_TOP_MODES:
            ctx["mode"] = head; ctx["obj"] = None; ctx["dirty"] = False
            continue

        # --- Inside a mode: 'use <obj>' enters that object ---
        if ctx["mode"] and head == "use" and len(toks) > 1:
            ctx["obj"] = toks[1]; ctx["dirty"] = False
            continue

        # --- clone <src> <dst> creates and enters the new object (dirty) ---
        if ctx["mode"] and head == "clone" and len(toks) >= 3:
            ctx["obj"] = toks[2]; ctx["dirty"] = True
            continue

        # --- add <name> creates a new object (e.g. add k8suser) ---
        if ctx["mode"] and head == "add" and len(toks) >= 2:
            ctx["obj"] = toks[1]; ctx["dirty"] = True
            continue

        # --- set <field> <value> marks the object dirty ---
        if head == "set":
            ctx["dirty"] = True
            # Special: 'set node*' inside device mode bulk-assigns
            if ctx["mode"] == "device" and len(toks) > 1:
                pass
            continue

        if head == "foreach":
            ctx["dirty"] = True
            continue

        # --- commit / commitall — clears dirty, drops sub-object level ---
        if head in ("commit", "commitall"):
            ctx["dirty"] = False
            info(f"Successfully committed {ctx['mode']}/{ctx['obj'] or ''}".strip())
            # cmsh stays in the object — but no longer dirty
            continue

        # --- list / ls — show objects in current mode ---
        if head in ("list", "ls"):
            if ctx["mode"] == "softwareimage":
                if any(n in (ctx.get("obj") or "") for n in ("k8s", "slurm")) or s.get("lab_cloned"):
                    print(LAB_IMAGE_TABLE_AFTER_CLONE)
                else:
                    print(LAB_IMAGE_TABLE_BASIC)
                continue
            if ctx["mode"] == "category":
                if s.get("lab_cloned") or any(c in (ctx.get("obj") or "")
                                              for c in ("k8s", "slurm")):
                    print(LAB_CATEGORY_TABLE_K8S)
                else:
                    print(LAB_CATEGORY_TABLE_BASIC)
                continue
            if ctx["mode"] == "device":
                if s.get("lab_device_layout") == "k8s":
                    print(LAB_DEVICE_TABLE_K8S)
                elif s.get("lab_device_layout") == "slurm":
                    print(LAB_DEVICE_TABLE_SLURM)
                else:
                    print(LAB_DEVICE_TABLE_DEFAULT)
                continue
            if ctx["mode"] == "user":
                # show k8suser if added in this session OR previously committed
                if (ctx.get("obj") == "k8suser" or
                        s.get("k8s_user_committed") or
                        "k8suser" in s.get("k8s_users", [])):
                    info("Name (key)       ID (key)         Primary group    Secondary groups")
                    info("---------------- ---------------- ---------------- ----------------")
                    info("cmsupport        1000             cmsupport")
                    info("k8suser          1001             k8suser")
                else:
                    info("Name (key)       ID (key)         Primary group    Secondary groups")
                    info("---------------- ---------------- ---------------- ----------------")
                    info("cmsupport        1000             cmsupport")
                continue
            # fallback at top level
            info("Available modes: device, category, softwareimage, "
                 "monitoring, user, wlm, partition, network")
            continue

        # --- show — describe the current object ---
        if head == "show":
            obj = ctx.get("obj") or (toks[1] if len(toks) > 1 else None)
            if not obj:
                info("Type 'show' inside an object, or 'show <name>'."); continue
            info(f"Object:        {obj}")
            info(f"Mode:          {ctx['mode']}")
            info(f"Status:        committed" if not ctx["dirty"] else "modified (uncommitted)")
            continue

        # --- reboot ---
        if head == "reboot":
            info("Reboot scheduled."); continue

        # --- helpful side-effects to make 'lab' scenario feel right ---
        if head in ("..", "..."):  # cd-up
            if ctx["obj"]:    ctx["obj"] = None; ctx["dirty"] = False
            elif ctx["mode"]: ctx["mode"] = None; ctx["dirty"] = False
            continue

        if head == "help":
            info("In cmsh:")
            info("  <mode>          enter a mode (device, category, softwareimage, user, ...)")
            info("  use <obj>       enter that named object")
            info("  clone <src> <dst>  create a new object as a copy")
            info("  add <name>      create a new object")
            info("  set <field> <v> modify a field (marks dirty *)")
            info("  commit          save changes")
            info("  list  /  ls     list objects in current mode")
            info("  show            show details of current object")
            info("  ..              up one level   |  exit/quit  leave cmsh")
            continue

        # Acknowledgement for any unrecognized cmsh-shaped command
        info(f"(cmsh: '{raw}' accepted — simulator ack)")

        # Track lab-specific state side effects
        if head == "commit":
            if ctx["mode"] == "softwareimage":
                s["lab_cloned"] = True
            elif ctx["mode"] == "device":
                if "k8s" in (ctx.get("obj") or ""):
                    s["lab_device_layout"] = "k8s"
                elif "slurm" in (ctx.get("obj") or ""):
                    s["lab_device_layout"] = "slurm"
            elif ctx["mode"] == "user":
                s["k8s_user_committed"] = True
            save_state(s)


# ===========================================================================
# Lab-mode commands — module, cm-wlm-setup, cm-kubernetes-setup,
# cat, su, ssh, gpu-pod, and the interactive cmsh sub-REPL
# ===========================================================================
AVAILABLE_MODULES = """\
------------------------------------------------- /cm/local/modulefiles -------------------------------------------------
bcm-post-install/10.0      cluster-tools/10.0  cm-scale/cm-scale.module  cmjob              dot              ipmitool/1.8.19                 mariadb-libs  modules   python3         sedutil/1.16.0       use.own
bcm-superpod-network/10.0  cm-bios-tools       cm-setup/10.0             cmsh               freeipmi/1.6.14  kubernetes/default/1.30.10-1.1  module-git    null      python39        shared
boost/1.81.0               cm-image/10.0       cmd                       containerd/1.7.21  gcc/13.1.0       luajit                          module-info   openldap  rocm-smi/4.3.0  slurm/slurm/23.02.8
------------------------------------------------ /cm/shared/modulefiles -------------------------------------------------------
cm-pmix3/3.1.7  default-environment  hdf5_18/1.8.21  hwloc/1.11.13  iperf/3.17.1           openblas/dynamic/0.3.18  openmpi4/gcc/4.1.5
cm-pmix4/4.1.3  gdb/13.1             hpl/2.3         hwloc2/2.8.0   mvapich2/gcc/64/2.3.7  openmpi/gcc/64/4.1.5     ucx/1.10.1
"""

GPU_POD_YAML = """\
apiVersion: v1
kind: Pod
metadata:
  name: gpu-pod
spec:
  restartPolicy: Never
  containers:
  - name: cuda-container
    image: nvcr.io/nvidia/cuda:12.6.2-base-ubuntu22.04
    command: ["nvidia-smi"]
    resources:
      limits:
        nvidia.com/gpu: 1
"""

GPU_POD_LOG = """\
Wed Feb 19 19:24:41 2026
+-------------------------------------------------------------------------------+
| NVIDIA-SMI 550.90.07      Driver Version: 550.90.07    CUDA Version: 12.6     |
|--------------------------------+---------------------+------------------------+
| GPU  Name        Persistence-M | Bus-Id     Disp.A   |   Volatile Uncorr. ECC |
| Fan  Temp   Perf Pwr:Usage/Cap |      Memory-Usage   |    GPU-Util Compute M. |
|                                |                     |                 MIG M. |
|================================+=====================+========================|
|   0  NVIDIA L40S-4C        On  | 00000000:03:00.0 Off|                      0 |
| N/A   N/A    P8    N/A /  N/A  |    1MiB /  4096MiB  | 0%             Default |
|                                |                     |                    N/A |
+--------------------------------+---------------------+------------------------+
"""


def cmd_module(args: list[str], s: dict) -> int:
    if not args:
        info("Usage: module {available|load|unload|list|purge} [name]")
        return 0
    op = args[0]
    if op in ("available", "avail", "av"):
        print(AVAILABLE_MODULES); return 0
    if op == "list":
        if not s["loaded_modules"]:
            info("No Modulefiles Currently Loaded.")
        else:
            info("Currently Loaded Modulefiles:")
            for i, m in enumerate(s["loaded_modules"], 1):
                info(f"  {i}) {m}")
        return 0
    if op == "load":
        if len(args) < 2: return err("usage: module load <name>")
        name = args[1]
        if name not in s["loaded_modules"]:
            s["loaded_modules"].append(name)
            save_state(s)
        # Real `module load` is silent on success
        return 0
    if op in ("unload", "rm"):
        if len(args) < 2: return err("usage: module unload <name>")
        if args[1] in s["loaded_modules"]:
            s["loaded_modules"].remove(args[1]); save_state(s)
        return 0
    if op == "purge":
        s["loaded_modules"] = []; save_state(s); return 0
    if op == "show":
        if len(args) < 2: return err("usage: module show <name>")
        info(f"-------------------------------------------------------------------")
        info(f"/cm/local/modulefiles/{args[1]}:")
        info(f"module-whatis   Adds {args[1]} to the user environment")
        info(f"setenv          {args[1].upper().replace('/','_')}_DIR /cm/local/apps/{args[1]}")
        info(f"-------------------------------------------------------------------")
        return 0
    return err(f"module: unknown operation '{op}'")


def cmd_cm_wlm_setup(args: list[str], s: dict) -> int:
    info("[GUI Simulator: Workload Manager Setup Wizard Launched]")
    info("... Navigating Setup (Step By Step)")
    info("... Selected Workload Manager: Slurm")
    info("... Selected Slurm Server Role: [X] bcm (head node)")
    info("... Selected Slurm Client Role: [X] slurm-01, [X] slurm-02, [X] slurm-03")
    info("... Selected GPU Client Role:   [X] slurm-04")
    info("... Configured Partition: defq (default)")
    info("... Saving configuration to /root/cm-wlm-setup.conf")
    info("[GUI Simulator: Deployment Complete!]")
    s["wlm_setup_done"] = True; save_state(s)
    return 0


def cmd_cm_kubernetes_setup(args: list[str], s: dict) -> int:
    flags, _ = parse_flags(args, {})
    if flags.get("add-user"):
        # `cm-kubernetes-setup --add-user k8suser`
        user = flags["add-user"] if isinstance(flags["add-user"], str) else "k8suser"
        info("Connecting to CMDaemon")
        info("Executing 3 stages")
        info("################### Starting execution for 'Kubernetes Setup'")
        info("  - kubernetes")
        info("  - docker")
        info("## Progress: 0")
        info("#### stage: kubernetes: Get Kube Cluster")
        info("## Progress: 33")
        info("#### stage: kubernetes: Check Permissions User Chart")
        info("## Progress: 66")
        info(f"#### stage: kubernetes: Add user '{user}' with namespace '{user}-restricted'")
        info("## Progress: 100")
        info("Took:     00:18 min.")
        info("Progress: 100/100")
        info("################### Finished execution for 'Kubernetes Setup', "
             "status: completed")
        s["k8s_user_added"] = True; save_state(s)
        return 0
    if flags.get("list-users"):
        info("Connecting to CMDaemon")
        info("Executing 3 stages")
        info("################### Starting execution for 'Kubernetes Setup'")
        info("  - kubernetes")
        info("## Progress: 0")
        info("#### stage: kubernetes: Get Kube Cluster")
        info("## Progress: 33")
        info("#### stage: kubernetes: Check Permissions User Chart")
        info("## Progress: 66")
        info("#### stage: kubernetes: Deploy List users")
        info("USERNAME____________: NAMESPACES______________________________")
        if s.get("k8s_user_added"):
            info("k8suser             : k8suser-restricted")
        info("## Progress: 100")
        info("Took:     00:00 min.")
        info("Progress: 100/100")
        info("################### Finished execution for 'Kubernetes Setup', "
             "status: completed")
        info("Kubernetes Setup finished!")
        return 0
    # Default invocation: full wizard
    info("[GUI Simulator: Kubernetes Setup Wizard Launched]")
    info("... Selected Kubernetes v1.30")
    info("... Configured Network: Service (10.150.0.0/16), Pod (172.29.0.0/16)")
    info("... Selected Master Nodes: [X] k8s-control-plane-01, "
         "[X] k8s-control-plane-02, [X] k8s-control-plane-03")
    info("... Selected Worker Nodes: [X] k8s-worker-01")
    info("... Selected Network Plugin: Calico")
    info("... Selected Operators: NVIDIA GPU Operator (v24.9.0), "
         "Prometheus, Dashboard")
    info("... Saving configuration to /root/cm-kubernetes-setup.conf")
    info("[GUI Simulator: Deployment Complete! (Approx 15 minutes elapsed)]")
    s["k8s_setup_done"] = True; save_state(s)
    return 0


def cmd_cat(args: list[str], s: dict) -> int:
    if not args: return err("usage: cat <file>")
    path = args[0]
    canned = {
        "/cm/shared/apps/lp/gpu.yaml":     GPU_POD_YAML,
        "/etc/slurm/slurm.conf":           "ClusterName=phoenix-prod-1\n"
                                           "ControlMachine=bcm\n"
                                           "PartitionName=defq Default=YES\n"
                                           "GresTypes=gpu\n",
        "/etc/nccl.conf":                  "NCCL_DEBUG=INFO\n"
                                           "NCCL_SOCKET_IFNAME=eth0\n"
                                           "NCCL_IB_HCA=mlx5\n",
        "/etc/os-release":                 'PRETTY_NAME="Ubuntu 22.04.5 LTS"\n'
                                           'NAME="Ubuntu"\n'
                                           'VERSION_ID="22.04"\n'
                                           'VERSION="22.04.5 LTS (Jammy Jellyfish)"\n'
                                           'VERSION_CODENAME=jammy\n'
                                           'ID=ubuntu\n'
                                           'ID_LIKE=debian\n'
                                           'HOME_URL="https://www.ubuntu.com/"\n'
                                           'SUPPORT_URL="https://help.ubuntu.com/"\n'
                                           'UBUNTU_CODENAME=jammy\n',
    }
    if path in canned:
        print(canned[path].rstrip()); return 0
    return err(f"cat: {path}: No such file or directory")


def cmd_su(args: list[str], s: dict) -> int:
    # Forms: `su -`, `su - <user>`, `su <user>`
    target = args[-1] if args and args[-1] != "-" else "root"
    if not args or args[-1] == "-":
        target = "root"
    info("Creating ECDSA key for ssh")
    s["user"] = target
    s["is_root"] = (target == "root")
    # Lab flow: non-root users land in the default k8s namespace
    if target != "root":
        s["current_namespace"] = "default"
    save_state(s)
    return 0


def cmd_ssh(args: list[str], s: dict) -> int:
    # Forms: `ssh user@host` or `ssh host`
    if not args: return err("usage: ssh user@host")
    target = args[-1]
    if "@" in target:
        user, host = target.split("@", 1)
    else:
        user, host = s["user"], target
    info("Welcome to Ubuntu 22.04.5 LTS (GNU/Linux 5.15.0-113-generic x86_64)")
    info(f" * Documentation:  https://help.ubuntu.com")
    info("")
    s["user"] = user
    s["host"] = host
    s["is_root"] = (user == "root")
    save_state(s)
    return 0


# ----- DCGMI extensions: discovery -l, group lifecycle, config -----
DCGMI_DISCOVERY_OUTPUT = """\
8 GPUs found.
+--------+----------------------------------------------------------------------+
| GPU ID | Device Information                                                   |
+--------+----------------------------------------------------------------------+
| 0      | Name: NVIDIA H100 80GB HBM3   PCI Bus ID: 00000000:1B:00.0   UUID: GPU-3a51 |
| 1      | Name: NVIDIA H100 80GB HBM3   PCI Bus ID: 00000000:43:00.0   UUID: GPU-7b22 |
| 2      | Name: NVIDIA H100 80GB HBM3   PCI Bus ID: 00000000:52:00.0   UUID: GPU-c1f0 |
| 3      | Name: NVIDIA H100 80GB HBM3   PCI Bus ID: 00000000:61:00.0   UUID: GPU-d8ee |
| 4      | Name: NVIDIA H100 80GB HBM3   PCI Bus ID: 00000000:9D:00.0   UUID: GPU-e472 |
| 5      | Name: NVIDIA H100 80GB HBM3   PCI Bus ID: 00000000:C3:00.0   UUID: GPU-1aa0 |
| 6      | Name: NVIDIA H100 80GB HBM3   PCI Bus ID: 00000000:D1:00.0   UUID: GPU-9907 |
| 7      | Name: NVIDIA H100 80GB HBM3   PCI Bus ID: 00000000:DF:00.0   UUID: GPU-fe35 |
+--------+----------------------------------------------------------------------+
6 NvSwitches found.
+-----------+
| Switch ID |
+-----------+
|   8       |
|   9       |
|  10       |
|  11       |
|  12       |
|  13       |
+-----------+
"""


def dcgmi_extensions(args: list[str], s: dict) -> Optional[int]:
    """Returns rc if it handled the command, else None."""
    if not args: return None
    sub = args[0]
    rest = args[1:]

    # dcgmi discovery -l
    if sub == "discovery" and rest and rest[0] == "-l":
        print(DCGMI_DISCOVERY_OUTPUT.rstrip()); return 0

    # dcgmi group -l / -c / -a / -r / -d
    if sub == "group":
        flags, pos = parse_flags(rest, {"-l": "list", "-c": "create",
                                        "-d": "delete", "-g": "group_id",
                                        "-a": "add", "-r": "remove"})
        if flags.get("list"):
            info(f"+-------------------+----------------------------------------------------------+")
            info(f"| GROUPS                                                                       |")
            info(f"| {len(s['dcgm_groups'])} groups found.                                                              |")
            info(f"+===================+==========================================================+")
            for g in s["dcgm_groups"]:
                info(f"| Group ID          | {g['id']}".ljust(75) + "|")
                info(f"| Group Name        | {g['name']}".ljust(75) + "|")
                info(f"| Entities          | {g['entities']}".ljust(75) + "|")
                info(f"+-------------------+----------------------------------------------------------+")
            return 0
        if isinstance(flags.get("create"), str):
            new_id = s["next_dcgm_group_id"]
            s["next_dcgm_group_id"] += 1
            s["dcgm_groups"].append({"id": new_id, "name": flags["create"],
                                     "entities": ""})
            save_state(s)
            info(f"Successfully created group \"{flags['create']}\" "
                 f"with a group ID of {new_id}")
            return 0
        if flags.get("group_id") and flags.get("add"):
            gid = int(flags["group_id"])
            entities = flags["add"] if isinstance(flags["add"], str) else ""
            g = next((x for x in s["dcgm_groups"] if x["id"] == gid), None)
            if not g: return err(f"group {gid} not found")
            g["entities"] = (g["entities"] + "," + entities).strip(",")
            save_state(s)
            info("Add to group operation successful."); return 0
        if flags.get("group_id") and flags.get("remove"):
            gid = int(flags["group_id"])
            entities = flags["remove"] if isinstance(flags["remove"], str) else ""
            g = next((x for x in s["dcgm_groups"] if x["id"] == gid), None)
            if not g: return err(f"group {gid} not found")
            keep = [e for e in g["entities"].split(",") if e and e not in entities.split(",")]
            g["entities"] = ",".join(keep)
            save_state(s)
            info("Remove from group operation successful."); return 0
        if isinstance(flags.get("delete"), str):
            gid = int(flags["delete"])
            before = len(s["dcgm_groups"])
            s["dcgm_groups"] = [g for g in s["dcgm_groups"] if g["id"] != gid]
            save_state(s)
            if len(s["dcgm_groups"]) == before:
                return err(f"group {gid} not found")
            info(f"Successfully removed group {gid}"); return 0

    # dcgmi config -g <id> --get
    if sub == "config":
        flags, _ = parse_flags(rest, {"-g": "group_id"})
        if flags.get("group_id") and "--get" in rest:
            gid = int(flags["group_id"])
            g = next((x for x in s["dcgm_groups"] if x["id"] == gid), None)
            if not g: return err(f"group {gid} not found")
            n_entities = len([e for e in g["entities"].split(",") if e])
            info("+------------------------------+------------------------------+------------------------------+")
            info(f"| {g['name']}".ljust(95) + "|")
            info(f"| Group of {n_entities} GPUs".ljust(95) + "|")
            info("+==============================+==============================+==============================+")
            info("| Field                        | Target                       | Current                      |")
            info("+------------------------------+------------------------------+------------------------------+")
            info("| Compute Mode                 | ****                         | Unrestricted                 |")
            info("| ECC Mode                     | ****                         | Disabled                     |")
            info("| Sync Boost                   | ****                         | ****                         |")
            info("| Memory Application Clock     | ****                         | ****                         |")
            info("| SM Application Clock         | ****                         | ****                         |")
            info("| Power Limit                  | ****                         | ****                         |")
            info("+------------------------------+------------------------------+------------------------------+")
            info("**** Non-homogenous settings across group. Use with -v flag to see details.")
            return 0

    return None


# Hook the extensions into cmd_dcgmi (monkey-patch style — define wrapper)
_original_cmd_dcgmi = cmd_dcgmi
def cmd_dcgmi_v2(args: list[str], s: dict) -> int:
    rc = dcgmi_extensions(args, s)
    if rc is not None:
        return rc
    return _original_cmd_dcgmi(args, s)


# ----- Interactive cmsh sub-REPL -----
# When user types just `cmsh` at the main prompt (no -c), we drop into a
# nested REPL with state-machine prompts mirroring real cmsh navigation:
#
#   [bcm]%
#   [bcm->softwareimage]%
#   [bcm->softwareimage*[slurm-image*]]%   (asterisks = uncommitted changes)
#   [bcm->softwareimage[slurm-image]]%     (after commit)
#
# Supported nested commands: softwareimage, category, device, user, monitoring,
# wlm, main, ls, list, status, clone <src> <dst>, set <field> <val>, foreach,
# add <name>, commit, exit, quit, help, ..
#
def cmsh_interactive(s: dict) -> int:
    """Enter interactive cmsh mode.  Returns when user types exit/quit/.."""
    path: list[str] = []        # mode stack: e.g. ['softwareimage','slurm-image']
    dirty = False               # pending uncommitted changes
    cmsh_state = {              # what's been provisioned during this session
        "device_committed_slurm": s.get("wlm_setup_done", False),
        "device_committed_k8s":   s.get("k8s_setup_done", False),
        "k8s_user_added":         s.get("k8s_user_added", False),
        "category_set":           None,
    }

    def _prompt() -> str:
        if not path:
            return "[bcm]% "
        if not dirty:
            inner = "->".join(path)
            return f"[bcm->{inner}]% "
        # add asterisks for uncommitted changes
        if len(path) == 1:
            return f"[bcm->{path[0]}*]% "
        return f"[bcm->{path[0]}*[{path[-1]}*]]% "

    info("Connecting to CMDaemon...")
    info("(type 'help' for commands, 'exit' to leave cmsh)")
    while True:
        try:
            raw = input(_prompt()).strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not raw: continue
        try:
            tokens = shlex.split(raw)
        except ValueError as e:
            err(f"parse error: {e}"); continue

        cmd = tokens[0]
        rest = tokens[1:]

        if cmd in ("exit", "quit"):
            if path:
                path.pop()
                continue
            break
        if cmd == "..":
            if path: path.pop()
            continue
        if cmd == "help":
            info("modes:    softwareimage | category | device | user | "
                 "monitoring | wlm | main")
            info("verbs:    ls | list | status | clone <src> <dst> | "
                 "set <field> <val> | foreach -c <cat> -n <range> ... | "
                 "add <name> | commit | reboot -c <cat> | exit | ..")
            continue

        # mode entry
        VALID_MODES = {"softwareimage", "category", "device", "user",
                       "monitoring", "wlm", "main"}
        if cmd in VALID_MODES and not path:
            path.append(cmd); continue

        # Inside softwareimage: clone default-image <new-name>
        if path and path[0] == "softwareimage" and cmd == "clone":
            if len(rest) < 2:
                err("usage: clone <src> <dst>"); continue
            src, dst = rest[0], rest[1]
            path.append(dst); dirty = True
            info(f"(cloned '{src}' → '{dst}'; type 'commit' to save)")
            continue

        # Inside category: clone default <new-name>
        if path and path[0] == "category" and cmd == "clone":
            if len(rest) < 2:
                err("usage: clone <src> <dst>"); continue
            src, dst = rest[0], rest[1]
            path.append(dst); dirty = True
            info(f"(cloned category '{src}' → '{dst}'; 'commit' to save)")
            continue

        # set softwareimage <name>  (inside category mode)
        if path and path[0] == "category" and cmd == "set" and len(rest) >= 2:
            if rest[0] == "softwareimage":
                cmsh_state["category_set"] = rest[1]; dirty = True
                info(f"(category softwareimage set to {rest[1]}; commit to save)")
                continue
            dirty = True; continue

        # device: foreach / set node ...
        if path and path[0] == "device" and cmd in ("set", "foreach"):
            joined = " ".join(rest)
            if "k8s-control-plane" in joined or "k8s-worker" in joined:
                cmsh_state["device_committed_k8s"] = True
                s["k8s_setup_done"] = True
            if "slurm" in joined:
                cmsh_state["device_committed_slurm"] = True
                s["wlm_setup_done"] = True
            dirty = True
            info(f"({len(joined.split())} value(s) set; commit to save)")
            continue

        # user: add k8suser / set password
        if path and path[0] == "user":
            if cmd == "add" and rest:
                path.append(rest[0]); dirty = True
                info(f"(adding user '{rest[0]}'; 'set password ...' then 'commit')")
                continue
            if cmd == "set" and rest and rest[0] == "password":
                dirty = True
                info(f"(password set; commit to save)")
                continue

        # commit
        if cmd == "commit":
            dirty = False
            if path and path[0] == "user" and len(path) > 1:
                cmsh_state["k8s_user_added"] = True
                s["k8s_user_added"] = True
            save_state(s)
            info("Successfully committed."); continue

        # ls / list — context-aware tables (mirroring real cmsh output)
        if cmd in ("ls", "list"):
            if path and path[0] == "softwareimage":
                hdr = ("Name (key)               Path (key)                               "
                       "Kernel version      Nodes")
                sep = ("------------------------ ---------------------------------------- "
                       "------------------- --------")
                rows = ["default-image            /cm/images/default-image                 5.15.0-113-generic  4",
                        "dgx-os-6.3-a100-image    /cm/images/dgx-os-6.3-a100-image         5.15.0-1063-nvidia  0",
                        "dgx-os-6.3-h100-image    /cm/images/dgx-os-6.3-h100-image         5.15.0-1063-nvidia  0"]
                if cmsh_state["device_committed_k8s"]:
                    rows.append("k8s-control-plane-image  /cm/images/k8s-control-plane-image       5.15.0-113-generic  0")
                    rows.append("k8s-worker-image         /cm/images/k8s-worker-image              5.15.0-113-generic  0")
                if cmsh_state["device_committed_slurm"]:
                    rows.append("slurm-image              /cm/images/slurm-image                   5.15.0-113-generic  0")
                info(hdr); info(sep); [info(r) for r in rows]; continue
            if path and path[0] == "category":
                hdr = "Name (key)               Software image           Nodes"
                sep = "------------------------ ------------------------ --------"
                rows = ["default                  default-image            4",
                        "dgx-a100                 dgx-os-6.3-a100-image    0",
                        "dgx-h100                 dgx-os-6.3-h100-image    0"]
                if cmsh_state["device_committed_k8s"]:
                    rows.append("k8s-control-plane        k8s-control-plane-image  0")
                    rows.append("k8s-worker               k8s-worker-image         0")
                if cmsh_state["device_committed_slurm"]:
                    rows.append("slurm                    slurm-image              0")
                info(hdr); info(sep); [info(r) for r in rows]; continue
            if path and path[0] == "device":
                hdr = ("Type                   Hostname (key)        MAC                "
                       "Category           IP              Network        Status")
                sep = ("---------------------- --------------------- ------------------ "
                       "------------------ --------------- -------------- ---------------")
                rows = ["HeadNode               bcm                   00:50:56:93:4B:40                     "
                        "10.141.255.254  internalnet    [   UP   ]"]
                if cmsh_state["device_committed_k8s"]:
                    for i, n in enumerate([
                        "k8s-control-plane-01", "k8s-control-plane-02",
                        "k8s-control-plane-03", "k8s-worker-01"], 1):
                        cat = "k8s-worker" if "worker" in n else "k8s-control-plane"
                        rows.append(f"PhysicalNode           {n:21} 4E:56:44:41:01:0{i}  "
                                    f"{cat:18} 10.141.0.{i}      internalnet    [   UP   ]")
                elif cmsh_state["device_committed_slurm"]:
                    for i in range(1, 5):
                        rows.append(f"PhysicalNode           slurm-{i:02d}              4E:56:44:41:01:0{i}  "
                                    f"slurm              10.141.0.{i}      internalnet    [   UP   ]")
                else:
                    for i in range(1, 5):
                        rows.append(f"PhysicalNode           node00{i}               4E:56:44:41:01:0{i}  "
                                    f"default            10.141.0.{i}      internalnet    [   UP   ]")
                info(hdr); info(sep); [info(r) for r in rows]; continue
            if path and path[0] == "user":
                hdr = "Name (key)       ID (key)         Primary group    Secondary groups"
                sep = "---------------- ---------------- ---------------- ----------------"
                rows = ["cmsupport        1000             cmsupport        "]
                if cmsh_state["k8s_user_added"]:
                    rows.append("k8suser          1001             k8suser          ")
                info(hdr); info(sep); [info(r) for r in rows]; continue
            info("(empty)"); continue

        # status (inside any mode)
        if cmd == "status":
            if path and path[0] == "device":
                up = sum(1 for n in s["nodes"] if n["state"] != "drain")
                info(f"All nodes: {up}/{len(s['nodes'])} UP")
                continue
            info("OK")
            continue

        # reboot -c <cat>
        if cmd == "reboot":
            info("(reboot accepted; nodes will rejoin in ~5 min)")
            continue

        err(f"cmsh: unknown command '{cmd}' in mode '{path[0] if path else 'root'}'")

    return 0


# ===========================================================================
# Lab-2: Container Toolkit, Docker, PyTorch training, Driver install, BMC
# ===========================================================================

# Canonical nvidia-smi output (also used by `docker run --gpus all ubuntu nvidia-smi`)
NVIDIA_SMI_OUTPUT = """\
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.86.15              Driver Version: 570.86.15      CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA H100 NVL                Off |   00000000:B5:00.0 Off |                    0 |
| N/A   51C    P0             124W / 400W |       1MiB /  95830MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                                Usage     |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
"""

NVIDIA_SMI_TRAINING_OUTPUT = """\
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.86.15              Driver Version: 570.86.15      CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA H100 NVL                Off |   00000000:B5:00.0 Off |                    0 |
| N/A   72C    P0             318W / 400W |    4216MiB /  95830MiB |     94%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                                Usage     |
|=========================================================================================|
|    0   N/A  N/A          204131      C   /usr/bin/python                      4192MiB    |
+-----------------------------------------------------------------------------------------+
"""


def cmd_curl(args: list[str], s: dict) -> int:
    """Recognise the NVIDIA Container Toolkit GPG-key/repo setup command."""
    line = " ".join(args)
    if "libnvidia-container" in line and "gpgkey" in line:
        info("Warning: apt-key output should not be parsed (stdout is not a terminal)")
        info("OK")
        return 0
    if "nvidia.github.io" in line and "container-toolkit.list" in line:
        info("deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] "
             "https://nvidia.github.io/libnvidia-container/stable/deb/$(ARCH) /")
        return 0
    info(f"(curl: simulated download of '{args[-1] if args else '?'}')")
    return 0


def cmd_apt_get(args: list[str], s: dict) -> int:
    """apt-get / apt — handles update, install, remove, upgrade."""
    if not args: return err("usage: apt-get {update|install|remove|upgrade}")
    op = args[0]
    if op == "update":
        info("Hit:1 http://archive.ubuntu.com/ubuntu jammy InRelease")
        info("Get:2 http://archive.ubuntu.com/ubuntu jammy-updates InRelease [128 kB]")
        info("Get:3 https://nvidia.github.io/libnvidia-container/stable/deb/amd64 InRelease")
        info("Fetched 38.1 MB in 5s (7568 kB/s)")
        info("Reading package lists... Done")
        info("Building dependency tree... Done")
        info("Reading state information... Done")
        return 0
    if op == "install":
        # Handle -y as boolean (don't let parse_flags consume the next pkg name)
        clean = [a for a in args[1:] if a not in ("-y", "--yes")]
        pkgs = [p for p in clean if not p.startswith("-")]
        if not pkgs: return err("usage: apt-get install [-y] <package>")
        for pkg in pkgs:
            info("Reading package lists... Done")
            info("Building dependency tree... Done")
            info(f"The following NEW packages will be installed:\n  {pkg}")
            info(f"Setting up {pkg} ...")
            if "nvidia-container-toolkit" in pkg:
                s["container_toolkit_installed"] = True
            if pkg.startswith("nvidia-driver-"):
                s["driver_installed"] = True
        save_state(s)
        return 0
    if op == "remove":
        flags, pos = parse_flags(args[1:], {})
        target = " ".join(pos)
        if "^nvidia-" in target or "nvidia" in target:
            info("Reading package lists... Done")
            info("Removing nvidia-driver-570 (570.86.15-0ubuntu0.22.04) ...")
            info("Removing libnvidia-compute-570 ...")
            info("Removing libnvidia-gl-570 ...")
            info("Purging configuration files for nvidia-driver-570 ...")
            s["driver_installed"] = False
            save_state(s)
        else:
            info(f"Removing {target} ...")
        return 0
    if op == "upgrade":
        info("174 packages can be upgraded. Run 'apt list --upgradable' to see them.")
        return 0
    info(f"(apt-get {op}: accepted)")
    return 0


def cmd_dpkg(args: list[str], s: dict) -> int:
    if not args: return err("usage: dpkg {-i|-l|-r} ...")
    op = args[0]
    if op == "-i":
        if len(args) < 2: return err("usage: dpkg -i <package.deb>")
        pkg = args[1]
        info(f"Selecting previously unselected package {Path(pkg).stem}.")
        info("(Reading database ... 185732 files and directories currently installed.)")
        info(f"Preparing to unpack {pkg} ...")
        info(f"Unpacking {Path(pkg).stem} (1.0-1) ...")
        info(f"Setting up {Path(pkg).stem} (1.0-1) ...")
        if "nvidia-driver-local-repo" in pkg:
            info("")
            info("The public nvidia-driver-local GPG key does not appear to be installed.")
            info("To install the key, run this command:")
            info("sudo cp /var/nvidia-driver-local-repo-ubuntu2204-570.86.15/"
                 "nvidia-driver-local-081EF1BD-keyring.gpg /usr/share/keyrings/")
        return 0
    if op == "-l":
        info("Desired=Unknown/Install/Remove/Purge/Hold")
        info("ii  nvidia-driver-570    570.86.15  amd64  NVIDIA driver metapackage")
        return 0
    info(f"(dpkg {op}: accepted)")
    return 0


def cmd_nvidia_ctk(args: list[str], s: dict) -> int:
    if args and args[0] == "runtime" and len(args) > 1 and args[1] == "configure":
        info("INFO[" + NOW() + "]  Configuration file: /etc/docker/daemon.json")
        info("INFO[" + NOW() + "]  Wrote runtime config")
        s["container_toolkit_installed"] = True
        save_state(s)
        return 0
    info(f"(nvidia-ctk {' '.join(args)}: accepted)")
    return 0


def cmd_systemctl(args: list[str], s: dict) -> int:
    if not args: return err("usage: systemctl {restart|status|start|stop} <unit>")
    op = args[0]
    unit = args[1] if len(args) > 1 else "<unit>"
    if op in ("restart", "start", "stop"):
        # Real systemctl is silent on success
        return 0
    if op == "status":
        info(f"● {unit}.service - {unit.title()} service")
        info("     Loaded: loaded (/lib/systemd/system/...; enabled)")
        info("     Active: active (running) since 2026-04-30 09:12:01 UTC; 4h 18min ago")
        return 0
    info(f"(systemctl {op}: accepted)")
    return 0


def cmd_lspci(args: list[str], s: dict) -> int:
    """Handle `lspci`, `lspci -Q`, and `lspci -Q | grep NVIDIA`."""
    grep_term = None
    for i, a in enumerate(args):
        if a == "|" and i + 2 < len(args) and args[i+1] == "grep":
            grep_term = args[i+2]; break
    lines = [
        "00:00.0 Host bridge: Intel Corporation Device",
        "00:1f.3 SATA controller: Intel Corporation",
        "01:00.0 Ethernet controller: Mellanox ConnectX-7",
        "b5:00.0 3D controller: NVIDIA Corporation GH100 [H100L 94GB] (rev a1)",
        "c1:00.0 Network controller: Mellanox BlueField-3 DPU",
    ]
    for ln in lines:
        if grep_term and grep_term not in ln: continue
        info(ln)
    return 0


def cmd_uname(args: list[str], s: dict) -> int:
    if "-a" in args:
        info("Linux acad13 5.15.0-113-generic #123-Ubuntu SMP "
             "Mon Jun 10 08:16:17 UTC 2026 x86_64 x86_64 x86_64 GNU/Linux")
    elif "-r" in args:
        info("5.15.0-113-generic")
    elif "-m" in args:
        info("x86_64")
    else:
        info("Linux")
    return 0


def cmd_mkdir(args: list[str], s: dict) -> int:
    # silent success — like real mkdir
    return 0


def cmd_docker(args: list[str], s: dict) -> int:
    if not args:
        return err("usage: docker {pull|run|ps|images|exec|stop|rm}")
    sub = args[0]
    rest = args[1:]
    if sub == "pull":
        if not rest: return err("usage: docker pull <image>")
        image = rest[0]
        info(f"Using default tag: {image.split(':')[-1] if ':' in image else 'latest'}")
        info(f"latest: Pulling from {image.split(':')[0]}")
        info("a72d190b3b39: Pull complete")
        info("e3a7c5b2c4d8: Pull complete")
        info(f"Status: Downloaded newer image for {image}")
        s.setdefault("docker_images", []).append(image)
        save_state(s)
        return 0
    if sub == "images":
        rows = []
        for img in s.get("docker_images", []):
            repo, tag = (img.split(":", 1) + ["latest"])[:2]
            rows.append([repo, tag, uuid.uuid4().hex[:12], "2 days ago", "12.4GB"])
        if not rows:
            info("REPOSITORY   TAG   IMAGE ID   CREATED   SIZE")
            return 0
        print(table(["REPOSITORY", "TAG", "IMAGE ID", "CREATED", "SIZE"], rows))
        return 0
    if sub == "ps":
        rows = []
        if s.get("in_container"):
            rows.append([s["container_id"][:12], s["container_image"],
                         '"bash"', "5 minutes ago", "Up 5 minutes",
                         "", "competent_kepler"])
        print(table(["CONTAINER ID", "IMAGE", "COMMAND", "CREATED",
                     "STATUS", "PORTS", "NAMES"], rows))
        return 0
    if sub == "run":
        # Parse docker run flags + image + cmd
        runtime_nvidia = False; gpus_all = False; ipc_host = False
        interactive = False; remove = False; detach = False
        volume = None; image = None; cmd_after = []
        i = 0
        while i < len(rest):
            a = rest[i]
            if a == "--rm": remove = True
            elif a == "--ipc=host": ipc_host = True
            elif a in ("-it", "-ti"): interactive = True
            elif a == "-i" or a == "-t": interactive = True
            elif a == "-d": detach = True
            elif a == "--runtime=nvidia": runtime_nvidia = True
            elif a == "--gpus":
                if i + 1 < len(rest):
                    gpus_all = (rest[i+1] == "all"); i += 1
            elif a == "-v":
                if i + 1 < len(rest): volume = rest[i+1]; i += 1
            elif image is None and (a.startswith("nvcr.io/") or a.startswith("ubuntu")
                                    or "/" in a or ":" in a or a in ("ubuntu",)):
                image = a
            else:
                if image is None: image = a
                else: cmd_after.append(a)
            i += 1
        if image is None:
            return err("docker run: missing image")
        # Case 1: ubuntu nvidia-smi (the toolkit verification)
        if "ubuntu" in image and cmd_after and cmd_after[0] == "nvidia-smi":
            if not s.get("container_toolkit_installed", False):
                return err("docker: Error response from daemon: failed to create task: "
                           "OCI runtime create failed: nvidia-container-cli not found")
            if not s.get("driver_installed", True):
                return err("Failed to initialize NVML: NVIDIA driver is not loaded")
            print(NVIDIA_SMI_OUTPUT.rstrip())
            return 0
        # Case 2: pytorch container — interactive shell
        if "pytorch" in image:
            if not interactive:
                # Background-style run — just print and return
                info(f"=============\n== PyTorch ==\n=============")
                info(f"NVIDIA Release {image.split(':')[-1]}")
                return 0
            if image not in s.get("docker_images", []):
                info(f"Unable to find image '{image}' locally")
                info(f"latest: Pulling from {image.split(':')[0]}")
                info(f"Status: Downloaded newer image for {image}")
                s.setdefault("docker_images", []).append(image)
            cid = uuid.uuid4().hex[:12]
            s["in_container"] = True
            s["container_id"] = cid
            s["container_image"] = image
            s["container_cpu_only"] = not gpus_all
            save_state(s)
            info("=============")
            info("== PyTorch ==")
            info("=============")
            info(f"NVIDIA Release {image.split(':')[-1]} (build 12345)")
            info("PyTorch Version 2.6.0a0+ecf3bae40a")
            info("")
            info(f"Container image Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.")
            if s["container_cpu_only"]:
                info("WARNING: Detected NVIDIA NVIDIA H100 GPU, which is not yet supported in this version of the container")
                info("CPU-only mode (no --gpus passed).")
            info("")
            return 0
        # Generic image — just print a short banner
        cid = uuid.uuid4().hex[:12]
        info(f"Started container {cid} from {image}")
        return 0
    if sub == "stop":
        if rest:
            info(rest[0])
        return 0
    if sub == "rm":
        if rest:
            info(rest[0])
        return 0
    if sub == "exec":
        info(f"(simulated docker exec — would run inside container)")
        return 0
    return err(f"docker: unknown subcommand '{sub}'")


def cmd_python(args: list[str], s: dict) -> int:
    """Handles python train.py, python test.py inside the PyTorch container."""
    if not args: return err("usage: python <script>")
    script = args[0]
    in_container = s.get("in_container", False)
    cpu_only = s.get("container_cpu_only", False)

    if script.endswith("train.py"):
        info("Train Epoch: 1 [0/60000 (0%)]\tLoss: 2.303346")
        info("Train Epoch: 1 [6400/60000 (11%)]\tLoss: 0.147270")
        info("Train Epoch: 1 [12800/60000 (21%)]\tLoss: 0.200906")
        info("Train Epoch: 1 [19200/60000 (32%)]\tLoss: 0.102331")
        info("Train Epoch: 1 [25600/60000 (43%)]\tLoss: 0.226644")
        info("Train Epoch: 1 [32000/60000 (53%)]\tLoss: 0.024340")
        info("Train Epoch: 1 [38400/60000 (64%)]\tLoss: 0.034060")
        info("Train Epoch: 1 [44800/60000 (75%)]\tLoss: 0.005021")
        info("Train Epoch: 1 [51200/60000 (85%)]\tLoss: 0.002311")
        info("Train Epoch: 1 [57600/60000 (96%)]\tLoss: 0.022788")
        info("")
        info("Test set: Average loss: 0.0406, Accuracy: 9878/10000 (99%)")
        elapsed = "12 minutes 45 seconds (CPU-only)" if cpu_only else "1 minute 22 seconds (GPU-accelerated)"
        info(f"Training complete. Elapsed: {elapsed}")
        s["training_done"] = True
        save_state(s)
        return 0

    if script.endswith("test.py"):
        target = args[1] if len(args) > 1 else "7.png"
        info(f"Processing file: {target}")
        # Match the lab's quirky behavior: predicts 7 for everything by default :)
        # but show actual digit for 2.png and 8.png on first call
        digit_map = {"2.png": 2, "7.png": 7, "8.png": 7}
        info(f"The predicted number is: {digit_map.get(target, 7)}")
        info(f"Device : {'cpu' if cpu_only else 'cuda'}")
        info(f"Time : {'8.71' if cpu_only else '3.02'} seconds")
        return 0

    info(f"(python {script}: simulated)")
    return 0


# ----- BMC simulator (Lab 5: BMC GUI tour) -----
BMC_DASHBOARD = """\
==========================================================================
 BASEBOARD MANAGEMENT CONTROLLER — NVIDIA DGX A100 (dgx02-BMC)
 IP: 10.155.37.122   FW: 0.16.09   Chassis SN: 1664621000679
==========================================================================
 Power-On Hours:        50 d 18 hrs
 Pending Deassertions:  332
 Host Online:           [●] yes
 Chassis Identify LED:  [○] off

 Firmware              BMC          BIOS         MB FPGA
 Primary               0.16.09*     1.09*        0.01.03
 Secondary             0.14.17      1.09         N/A
 Communication         N/A          N/A          N/A
"""

BMC_SENSORS = """\
SENSOR                  VALUE       UNIT      STATUS
PWR_GB_GPU0             387.5       W         OK
PWR_GB_GPU1             402.1       W         OK
PWR_GB_GPU2             391.8       W         OK
PWR_GB_GPU3             409.4       W         OK
PWR_GB_GPU4             378.2       W         OK
PWR_GB_GPU5             395.7       W         OK
PWR_GB_GPU6             401.2       W         OK
PWR_GB_GPU7             388.9       W         OK
TEMP_GPU0               72          C         OK
TEMP_GPU1               74          C         OK
FAN1_RPM                9450        RPM       OK
FAN2_RPM                9510        RPM       OK
12V_PSU1                12.02       V         OK
12V_PSU2                11.98       V         OK
CPU0_TEMP               52          C         OK
CRITICAL_SENSORS:       (none)
"""

BMC_GPU_INFO = """\
GPU Information — DGX A100 (8x NVIDIA A100-SXM4-40GB)
======================================================================
 GPU#  Marketing Name              Memory   Power Usage    Status
 0     NVIDIA A100-SXM4-40GB       40 GB    387.5 W        OK
 1     NVIDIA A100-SXM4-40GB       40 GB    402.1 W        OK
 2     NVIDIA A100-SXM4-40GB       40 GB    391.8 W        OK
 3     NVIDIA A100-SXM4-40GB       40 GB    409.4 W        OK
 4     NVIDIA A100-SXM4-40GB       40 GB    378.2 W        OK
 5     NVIDIA A100-SXM4-40GB       40 GB    395.7 W        OK
 6     NVIDIA A100-SXM4-40GB       40 GB    401.2 W        OK
 7     NVIDIA A100-SXM4-40GB       40 GB    388.9 W        OK
 ----------------------------------------------------------------------
 Total GPU memory in system:                320 GB
 Min GPU power:                             378.2 W (GPU 4)
 Max GPU power:                             409.4 W (GPU 3)
"""


def cmd_bmc(args: list[str], s: dict) -> int:
    if not args or args[0] in ("-h", "--help", "help"):
        info("BMC simulator (DGX A100):")
        info("  bmc login                  log into the BMC web UI")
        info("  bmc dashboard              uptime + Pending Deassertions + firmware")
        info("  bmc sensor[s]              list sensors (look for PWR_GB_GPU#)")
        info("  bmc gpu-info               GPU marketing name + memory + power")
        info("  bmc fru-info               chassis FRU information")
        info("  bmc system-inventory       hardware inventory")
        info("  bmc signout                log out")
        return 0
    sub = args[0]
    if sub == "login":
        s["bmc_logged_in"] = True; save_state(s)
        info("Logging into dgx02-BMC (10.155.37.122) as student ...")
        info("Welcome to the NVIDIA DGX A100 Baseboard Management Controller.")
        return 0
    if not s.get("bmc_logged_in"):
        return err("bmc: not logged in. Run 'bmc login' first.")
    if sub == "dashboard":
        print(BMC_DASHBOARD.rstrip()); return 0
    if sub in ("sensor", "sensors"):
        print(BMC_SENSORS.rstrip()); return 0
    if sub == "gpu-info":
        print(BMC_GPU_INFO.rstrip()); return 0
    if sub == "fru-info":
        info("Chassis Type:       Rack Mount Chassis")
        info("Chassis Part:       920-23687-2530-100")
        info("Chassis Serial:     1664621000679")
        info("Board Mfg:          NVIDIA")
        info("Board Product:      DGX A100")
        info("Board Serial:       1660122000055")
        return 0
    if sub == "system-inventory":
        info("Component        Vendor      Model              Status")
        info("CPU 0            AMD         EPYC 7742          OK")
        info("CPU 1            AMD         EPYC 7742          OK")
        info("DIMM B1          Micron      DDR4 64GB          OK")
        info("GPU 0..7         NVIDIA      A100-SXM4-40GB     OK")
        info("NVSwitch 0..5    NVIDIA      NVSwitch 1.0       OK")
        info("PSU 1..6         Delta       3000W              OK")
        return 0
    if sub == "signout":
        s["bmc_logged_in"] = False; save_state(s)
        info("Signed out of dgx02-BMC."); return 0
    return err(f"bmc: unknown command '{sub}'")


# ----- nvidia-smi --query-gpu loop variant ------
def nvidia_smi_query_extension(args: list[str], s: dict) -> Optional[int]:
    """Handles `--query-gpu=...` and `-l <N>` looping forms.  Returns rc or None."""
    full = " ".join(args)
    if "--query-gpu" not in full:
        return None
    # Parse fields and format
    fields_match = re.search(r"--query-gpu=([^\s]+)", full)
    fmt_match = re.search(r"--format=([^\s]+)", full)
    loop_match = re.search(r"-l\s+(\d+)", full)
    fields = fields_match.group(1).split(",") if fields_match else ["utilization.gpu"]
    fmt = fmt_match.group(1) if fmt_match else "csv"
    is_csv = "csv" in fmt
    units = ("noheader" not in fmt and "nounits" not in fmt)
    # Generate one or three rows depending on loop flag
    iterations = 3 if loop_match else 1
    if is_csv:
        # Header
        if "noheader" not in fmt:
            header = []
            for f in fields:
                if f == "utilization.gpu" and units: header.append("utilization.gpu [%]")
                elif f.startswith("memory.") and units:
                    header.append(f + " [MiB]")
                elif f == "power.draw" and units: header.append("power.draw [W]")
                elif f == "temperature.gpu": header.append("temperature.gpu")
                else: header.append(f)
            info(", ".join(header))
        for _ in range(iterations):
            row = []
            for f in fields:
                if f == "utilization.gpu":
                    row.append(f"{random.randint(0, 5)} %" if units else str(random.randint(0,5)))
                elif f == "memory.total":
                    row.append("95830 MiB" if units else "95830")
                elif f == "memory.used":
                    row.append("1 MiB" if units else "1")
                elif f == "memory.free":
                    row.append("95331 MiB" if units else "95331")
                elif f == "temperature.gpu":
                    row.append(str(random.randint(48, 52)))
                elif f == "power.draw":
                    pw = round(random.uniform(95, 100), 2)
                    row.append(f"{pw} W" if units else str(pw))
                else:
                    row.append("?")
            info(", ".join(row))
        return 0
    # Non-csv fallback
    info("(nvidia-smi --query-gpu: format not simulated)")
    return 0


# Wrap nvidia-smi to support the --query forms in addition to existing ones
_original_cmd_nvidia_smi = cmd_nvidia_smi
def cmd_nvidia_smi_v2(args: list[str], s: dict) -> int:
    # Driver-not-installed mode (Lab 3 starts with no driver)
    if not s.get("driver_installed", True):
        info("Command 'nvidia-smi' not found, but can be installed with:")
        info("sudo apt install nvidia-utils-535       # version 535.230.02-...")
        info("sudo apt install nvidia-utils-570       # version 570.86.15-...")
        return 1
    # --query-gpu= form (Lab 4)
    rc = nvidia_smi_query_extension(args, s)
    if rc is not None:
        return rc
    # If training is in progress, show GPU-busy version
    if s.get("training_done") and not args:
        print(NVIDIA_SMI_TRAINING_OUTPUT.rstrip()); return 0
    # Otherwise, plain nvidia-smi prints the canonical lab output (one H100)
    if not args:
        print(NVIDIA_SMI_OUTPUT.rstrip()); return 0
    return _original_cmd_nvidia_smi(args, s)


# ===========================================================================
# BCM Admin Lab (Practices 0-5) — setup_lab, readmac, cm-chroot-sw-img,
#                                  rshell, lsmod, touch, imageupdate
# ===========================================================================
SETUP_LAB_OUTPUT = """\
Copying lab files
Creating 'nodes.csv' for Training-01
Installing mpich...
Head node:
Setting up environment for cluster Training-01 ...
Configuring DHCP scopes ...
Generating PXE boot configuration ...
Pre-staging /cm/shared/{ramp.sh, rampcheck.sh, rampaction.sh} ...
[output truncated]
unmounted /cm/images/default-image/dev/pts
unmounted /cm/images/default-image/dev
unmounted /cm/images/default-image/proc
unmounted /cm/images/default-image/sys
unmounted /cm/images/default-image/run
Done!"""

NODES_CSV = """\
# Read this file with readmac.sh,
# node,mac
node003,4E:56:44:41:01:03
node004,4E:56:44:41:01:04
"""

# Default kernel modules in default-image (Practice 2 Task 2)
DEFAULT_KERNEL_MODULES = [
    "nfs", "e1000", "tg3", "sata_sil", "ext3", "ext4", "forcedeth",
    "mlx5_core", "rdma_cm", "ib_core", "tcp_bbr", "overlay", "br_netfilter",
]

# Files visible inside default-image and on rshell'd nodes
NODE_ROOT_FILES = [
    "bin", "boot", "cm", "dev", "etc", "home", "initrd.img", "lib",
    "lib32", "lib64", "libx32", "local", "media", "mnt", "opt", "proc",
    "root", "run", "sbin", "share", "snap", "srv", "sys", "tmp",
    "usr", "var", "vmlinuz",
]


def cmd_setup_lab(args: list[str], s: dict) -> int:
    print(SETUP_LAB_OUTPUT)
    # Pre-stage nodes.csv as a virtual file
    s.setdefault("virtual_files", {})["nodes.csv"] = NODES_CSV
    save_state(s)
    return 0


def cmd_readmac(args: list[str], s: dict) -> int:
    if not args:
        return err("usage: ./readmac.sh nodes.csv")
    csv_file = args[0]
    # Process node003 + node004 (per the lab's nodes.csv)
    info("Processing node: node003 with mac=4E:56:44:41:01:03")
    info("Processing node: node004 with mac=4E:56:44:41:01:04")
    info("Done!")
    # Mark these nodes as provisioned (state=alloc) — mirroring real BCM
    for n in s["nodes"]:
        if n["name"] in ("node-003", "node-004"):
            n["state"] = "alloc"
    save_state(s)
    return 0


def cmd_cm_chroot_sw_img(args: list[str], s: dict) -> int:
    if not args:
        return err("usage: cm-chroot-sw-img /cm/images/<image>")
    image_path = args[0]
    image_name = Path(image_path).name
    s["chroot_active"] = True
    s["chroot_image"] = image_name
    save_state(s)
    return 0


def cmd_touch(args: list[str], s: dict) -> int:
    if not args:
        return err("usage: touch <file>")
    if s.get("chroot_active") and s.get("chroot_image") == "default-image":
        for f in args:
            if f not in s.get("chroot_files", []):
                s.setdefault("chroot_files", []).append(f)
            if f == "test.txt":
                s["default_image_has_testtxt"] = True
        save_state(s)
    elif s.get("rshell_node"):
        # Touch on a node — non-persistent for the lab
        pass
    return 0


def cmd_ls(args: list[str], s: dict) -> int:
    """ls — context-aware: chroot, rshell, or normal."""
    # Inside chroot of default-image
    if s.get("chroot_active"):
        files = list(NODE_ROOT_FILES)
        if s.get("chroot_image") == "default-image":
            files.extend(s.get("chroot_files", []))
        info("  ".join(sorted(set(files))))
        return 0
    # Inside rshell to a node
    if s.get("rshell_node"):
        node = s["rshell_node"]
        files = list(NODE_ROOT_FILES)
        # node001 sees test.txt only if kernel was reboot-synced AND
        # node is in default category with default-image
        node_short = node.replace("node-", "node")
        in_default = (s.get("node_category", {}).get(node_short, "default")
                      == "default")
        if (in_default and s.get("default_image_has_testtxt") and
                s.get("node001_kernel_synced") and node_short == "node001"):
            files.append("test.txt")
        info("  ".join(sorted(set(files))))
        return 0
    # Default: home directory listing
    info("Desktop  Documents  Downloads  Music  Pictures  Public  Templates  Videos")
    return 0


def cmd_lsmod(args: list[str], s: dict) -> int:
    grep = None
    for i, a in enumerate(args):
        if a == "|" and i + 2 < len(args) and args[i+1] == "grep":
            grep = args[i+2]; break
    # Base modules always present
    lines = [
        ("Module", "Size", "Used by"),
        ("nfs", "524288", "1"),
        ("mlx5_core", "2400000", "0"),
        ("rdma_cm", "118784", "1"),
        ("ib_core", "524288", "3"),
        ("ext4", "1138688", "1"),
    ]
    # soundcore visible on a rshell'd node only after the post-kernel reboot
    if (s.get("soundcore_added") and s.get("node001_kernel_synced")
            and s.get("rshell_node", "") in ("node001", "node-001")):
        lines.append(("soundcore", "16384", "0"))
    if grep:
        lines = [ln for ln in lines if grep in ln[0]]
    for ln in lines:
        if ln[0] == "Module": continue  # don't print header in grep mode
        info(f"{ln[0]:<22} {ln[1]:<10} {ln[2]}")
    return 0


def cmd_rshell(args: list[str], s: dict) -> int:
    """rshell <node> — drop into ssh session on the node (top-level form)."""
    if not args: return err("usage: rshell <node>")
    target = args[0]
    short = target if target.startswith("node-") else target.replace("node", "node-")
    n = next((x for x in s.get("nodes", []) if x["name"] == short), None)
    if not n and target.startswith("node00"):
        # tolerate node001-style names even if state doesn't list them
        n = {"name": short}
    if not n:
        return err(f"rshell: node '{target}' not found")
    s["rshell_node"] = target
    save_state(s)
    return 0


def cmd_imageupdate(args: list[str], s: dict) -> int:
    """imageupdate -n <node> [-w]   — sync the node's image."""
    flags, _ = parse_flags(args, {"-n": "node", "-w": "write"})
    node = flags.get("node")
    if not node:
        return err("usage: imageupdate -n <node> [-w]")
    write = bool(flags.get("write"))
    # Look up the image for this node
    short = node if node.startswith("node-") else node.replace("node", "node-")
    cat = s.get("node_category", {}).get(node, "default")
    img = s.get("node_softwareimage", {}).get(node)
    if not img:
        # Inherit from category
        img = next((c["image"] for c in s.get("cloned_categories", [])
                    if c["name"] == cat), "default-image")
    mode = "UPDATE"
    dry = "no" if write else "yes"
    info(f"{NOW()} [notice] bcm: Provisioning started: sending "
         f"bcm:/cm/images/{img} to {node}:/, mode {mode}, dry run = {dry}")
    info(f"{NOW()} [notice] bcm: Provisioning completed: sent "
         f"bcm:/cm/images/{img} to {node}:/, mode {mode}, dry run = {dry}")
    info(f"imageupdate -n {node} {'-w' if write else ''} [ COMPLETED ]")
    return 0


def cmd_reboot_node(args: list[str], s: dict) -> int:
    """Top-level `reboot <node>` (also bound to `reboot` so cmsh-style works)."""
    if not args: return err("usage: reboot <node>")
    target = args[0]
    info(f"{NOW()} [notice] bcm: {target} [  DOWN  ]")
    info(f"{NOW()} [notice] bcm: {target} [       BOOTING       ] (ldlinux.e64 from bcm)")
    info(f"{NOW()} [notice] bcm: {target} [     INSTALLING      ] (node installer started)")
    info(f"{NOW()} [notice] bcm: {target} [ INSTALLER_CALLINGINIT ] (switching to local root)")
    info(f"{NOW()} [notice] bcm: {target} [   UP   ]")
    # Mark kernel sync if soundcore was added and target is node001
    if target in ("node001", "node-001") and s.get("soundcore_added"):
        s["node001_kernel_synced"] = True
        save_state(s)
    return 0


# ----- BCM user/group commands (top-level, mirror cmsh user/group) -----
def cmd_bcmuser(args: list[str], s: dict) -> int:
    """`bcmuser` — quick CLI to add/remove BCM users without entering cmsh."""
    if not args:
        rows = [[u["name"], u["id"], u["primary"], u["secondary"]]
                for u in s.get("bcm_users", [])]
        print(table(["Name", "ID", "Primary group", "Secondary groups"], rows))
        return 0
    op = args[0]
    if op == "add" and len(args) > 1:
        name = args[1]
        if any(u["name"] == name for u in s["bcm_users"]):
            return err(f"user '{name}' already exists")
        uid = s["next_user_id"]; s["next_user_id"] += 1
        s["bcm_users"].append({"name": name, "id": uid, "primary": name,
                                "secondary": "", "password": ""})
        # Auto-create primary group
        s.setdefault("bcm_groups", []).append({"name": name, "id": uid,
                                                "members": name})
        save_state(s)
        info(f"User '{name}' added with ID {uid}")
        return 0
    if op == "remove" and len(args) > 1:
        name = args[1]
        before = len(s["bcm_users"])
        s["bcm_users"] = [u for u in s["bcm_users"] if u["name"] != name]
        if len(s["bcm_users"]) == before:
            return err(f"user '{name}' not found")
        save_state(s)
        info(f"Successfully removed user {name}"); return 0
    return err("usage: bcmuser [add|remove] <name>")


def cmd_bcmgroup(args: list[str], s: dict) -> int:
    if not args:
        rows = [[g["name"], g["id"], g["members"]] for g in s.get("bcm_groups", [])]
        print(table(["Name", "ID", "Members"], rows))
        return 0
    op = args[0]
    if op == "add" and len(args) > 1:
        name = args[1]
        gid = s["next_group_id"]; s["next_group_id"] += 1
        s["bcm_groups"].append({"name": name, "id": gid, "members": ""})
        save_state(s)
        info(f"Group '{name}' added with ID {gid}"); return 0
    if op == "remove" and len(args) > 1:
        name = args[1]
        s["bcm_groups"] = [g for g in s["bcm_groups"] if g["name"] != name]
        save_state(s)
        info(f"Successfully removed group {name}"); return 0
    return err("usage: bcmgroup [add|remove] <name>")


# ----- Monitoring (Practice 5) -----
def cmd_monitoring(args: list[str], s: dict) -> int:
    if not args:
        info("monitoring {add-producer|add-action|add-trigger|add-dashboard|list}"); return 0
    op = args[0]; rest = args[1:]
    if op == "add-producer":
        flags, _ = parse_flags(rest, {})
        name = flags.get("name", "ramp")
        kind = flags.get("type", "metric")
        script = flags.get("script", "")
        s["monitoring_data_producers"].append({"name": name, "type": kind,
                                                "script": script,
                                                "interval": int(flags.get("interval", 1))})
        save_state(s)
        info(f"Data producer '{name}' added (type={kind})"); return 0
    if op == "add-action":
        flags, _ = parse_flags(rest, {})
        name = flags.get("name", "rampaction")
        s["monitoring_actions"].append({"name": name,
                                         "script": flags.get("script", "")})
        save_state(s)
        info(f"Action '{name}' added"); return 0
    if op == "add-trigger":
        flags, _ = parse_flags(rest, {})
        name = flags.get("name", "Ramp Trigger")
        s["monitoring_triggers"].append({
            "name": name,
            "measurable": flags.get("measurable", "ramp"),
            "operator": flags.get("operator", ">"),
            "value": float(flags.get("value", 95)),
            "during": flags.get("during", "rampaction"),
        })
        save_state(s)
        info(f"Trigger '{name}' added"); return 0
    if op == "add-dashboard":
        flags, _ = parse_flags(rest, {})
        name = flags.get("name", "ramp-dashboard")
        s["monitoring_dashboards"].append({"name": name,
                                            "widgets": [flags.get("data", "ramp")]})
        save_state(s)
        info(f"Dashboard '{name}' added"); return 0
    if op == "list":
        info(f"Data producers ({len(s['monitoring_data_producers'])}):")
        for p in s["monitoring_data_producers"]:
            info(f"  {p['name']:20} type={p['type']:8} interval={p['interval']}s "
                 f"script={p.get('script','')}")
        info(f"Actions ({len(s['monitoring_actions'])}):")
        for a in s["monitoring_actions"]:
            info(f"  {a['name']:20} script={a.get('script','')}")
        info(f"Triggers ({len(s['monitoring_triggers'])}):")
        for t in s["monitoring_triggers"]:
            info(f"  {t['name']:20} {t['measurable']} {t['operator']} {t['value']} -> {t['during']}")
        info(f"Dashboards ({len(s['monitoring_dashboards'])}):")
        for d in s["monitoring_dashboards"]:
            info(f"  {d['name']:20} widgets={','.join(d['widgets'])}")
        return 0
    return err(f"monitoring: unknown op '{op}'")


# Hook the BCM-style cmsh device commands so user can drive lab from -c form
_original_cmsh_device = cmsh_device
def cmsh_device_v2(args: list[str], s: dict) -> int:
    if args and args[0] == "rshell":
        if len(args) < 2: return err("usage: rshell <node>")
        return cmd_rshell([args[1]], s)
    if args and args[0] == "imageupdate":
        return cmd_imageupdate(args[1:], s)
    if args and args[0] == "use":
        if len(args) < 2: return err("usage: use <node>")
        info(f"[bcm->device[{args[1]}]]   (interactive context simulated)"); return 0
    if args and args[0] == "set":
        # `device set <node> category Lite` form
        if len(args) >= 4 and args[2] == "category":
            target = args[1]
            s.setdefault("node_category", {})[target] = args[3]
            save_state(s)
            info(f"({target} category set to {args[3]}; commit to save)"); return 0
        if len(args) >= 4 and args[2] == "softwareimage":
            target = args[1]
            s.setdefault("node_softwareimage", {})[target] = args[3]
            save_state(s)
            info(f"({target} softwareimage set to {args[3]}; commit to save)"); return 0
    if args and args[0] == "get":
        # `device get <node> softwareimage`
        if len(args) >= 3:
            target = args[1]; field = args[2]
            if field == "softwareimage":
                cat = s.get("node_category", {}).get(target, "default")
                override = s.get("node_softwareimage", {}).get(target)
                if override:
                    info(override)
                else:
                    img = next((c["image"] for c in s.get("cloned_categories", [])
                                if c["name"] == cat), "default-image")
                    info(f"{img} (category:{cat})")
                return 0
    if args and args[0] == "listnodes":
        category = args[1] if len(args) > 1 else "default"
        nodes_in_cat = [n for n, cat in s.get("node_category", {}).items()
                        if cat == category]
        rows = []
        for nm in nodes_in_cat:
            n = next((x for x in s["nodes"] if x["name"].endswith(nm[-3:])), None)
            ip = "10.141.0." + nm[-1] if nm[:-1] == "node00" else ""
            rows.append(["PhysicalNode", nm,
                         f"4E:56:44:41:01:0{nm[-1]}", category, ip,
                         "Internalnet", "[ UP ]"])
        print(table(["Type", "Hostname (key)", "MAC", "Category", "IP",
                     "Network", "Status"], rows))
        return 0
    return _original_cmsh_device(args, s)


# Hook for cmsh_image
_original_cmsh_image = cmsh_image
def cmsh_image_v2(args: list[str], s: dict) -> int:
    if args and args[0] == "use":
        if len(args) < 2: return err("usage: use <image-name>")
        info(f"[bcm->softwareimage[{args[1]}]]   (interactive context simulated)"); return 0
    if args and args[0] == "kernelmodules":
        return cmsh_kernelmodules(args[1:], s)
    if args and args[0] == "list":
        # Augment the BCM_IMAGES baseline with anything cloned this session
        all_images = list(BCM_IMAGES)
        existing_names = {i["name"] for i in all_images}
        for cloned in s.get("cloned_images", []):
            if cloned not in existing_names:
                all_images.append({"name": cloned,
                                    "kernel": "5.15.0-91-generic",
                                    "size_gb": 24, "category": "default"})
        print(table(["NAME", "KERNEL", "SIZE_GB", "CATEGORY"],
                    [[i["name"], i["kernel"], i["size_gb"], i["category"]]
                     for i in all_images]))
        return 0
    if args and args[0] == "clone":
        if len(args) < 3: return err("usage: clone <src> <dst>")
        src, dst = args[1], args[2]
        if dst not in s.get("cloned_images", []):
            s.setdefault("cloned_images", []).append(dst)
            save_state(s)
        info(f"Image '{dst}' created (cloned from '{src}').")
        info(f"Initial ramdisk for image {dst} was generated successfully")
        return 0
    if args and args[0] == "add":
        # `softwareimage add <module>` — kernelmodules sub-mode shortcut
        if len(args) >= 2:
            return cmsh_kernelmodules(["add", args[1]], s)
    return _original_cmsh_image(args, s)


def cmsh_kernelmodules(args: list[str], s: dict) -> int:
    """Sub-mode of softwareimage[default-image]."""
    cmd = args[0] if args else "list"
    modules = list(DEFAULT_KERNEL_MODULES)
    if s.get("soundcore_added"):
        modules.append("soundcore")
    if cmd == "list":
        info(f"{'Module (key)':<24} {'Parameters':<48}")
        info("-" * 24 + " " + "-" * 48)
        for m in modules:
            info(f"{m:<24}")
        return 0
    if cmd == "add":
        if len(args) < 2: return err("usage: add <module>")
        if args[1] == "soundcore":
            s["soundcore_added"] = True
            save_state(s)
        info(f"({args[1]} queued; type 'commit' to save)"); return 0
    if cmd == "commit":
        info("Initial ramdisk for image default-image was generated successfully")
        return 0
    return err(f"cmsh kernelmodules: unknown command '{cmd}'")


# Hook cmsh_category for the `listnodes`, `use`, `clone`, `set softwareimage` lab path
_original_cmsh_category = cmsh_category
def cmsh_category_v2(args: list[str], s: dict) -> int:
    if args and args[0] == "list":
        # Show full table with our cloned categories
        # Count nodes per category
        cat_count: dict[str, int] = {}
        for nm, cat in s.get("node_category", {}).items():
            cat_count[cat] = cat_count.get(cat, 0) + 1
        rows = []
        for c in s.get("cloned_categories", []):
            rows.append([c["name"], c["image"], cat_count.get(c["name"], 0)])
        print(table(["Name (key)", "Software image", "Nodes"], rows))
        return 0
    if args and args[0] == "listnodes":
        category = args[1] if len(args) > 1 else "default"
        return cmsh_device_v2(["listnodes", category], s)
    if args and args[0] == "clone":
        if len(args) < 3: return err("usage: clone <src> <dst>")
        src, dst = args[1], args[2]
        existing_img = next((c["image"] for c in s.get("cloned_categories", [])
                            if c["name"] == src), "default-image")
        s.setdefault("cloned_categories", []).append({"name": dst,
                                                       "image": existing_img})
        save_state(s)
        info(f"(category '{dst}' cloned from '{src}'; type 'commit' to save)"); return 0
    if args and args[0] == "use":
        if len(args) < 2: return err("usage: use <category>")
        info(f"[bcm->category[{args[1]}]]   (interactive context simulated)"); return 0
    if args and args[0] == "set":
        # `set softwareimage <name>` after `use <category>`
        if len(args) >= 3 and args[1] == "softwareimage":
            # Apply to most recently cloned category as a heuristic
            if s.get("cloned_categories"):
                s["cloned_categories"][-1]["image"] = args[2]
                save_state(s)
            info(f"(softwareimage set to {args[2]}; commit to save)"); return 0
    if args and args[0] == "show":
        # Show last-used category
        if s.get("cloned_categories"):
            c = s["cloned_categories"][-1]
            info(f"Parameter                        Value")
            info(f"-" * 60)
            info(f"Name                             {c['name']}")
            info(f"Nodes                             0")
            info(f"Software image                   {c['image']}")
            info(f"Default category                 no")
            return 0
    return _original_cmsh_category(args, s)


# ===========================================================================
# NCP-AIO exam commands — ngc, nvidia-smi MIG/nvlink/dmon, ib_write_bw,
#                          docker login, perftest tools
# ===========================================================================
def cmd_ngc(args: list[str], s: dict) -> int:
    if not args:
        info("Usage: ngc {auth|config|registry|model|resource|user|orgs|version}")
        return 0
    sub = args[0]; rest = args[1:]
    if sub == "version":
        info("NGC CLI 3.42.0"); return 0
    if sub == "auth":
        if rest and rest[0] == "login":
            info("Successfully authenticated to NGC.")
            s["ngc_authenticated"] = True; save_state(s)
            return 0
        if rest and rest[0] == "logout":
            info("Logged out of NGC.")
            s["ngc_authenticated"] = False; save_state(s)
            return 0
    if sub == "config":
        if rest and rest[0] == "set":
            info("Configuration saved to ~/.ngc/config"); return 0
        info("apikey       ********")
        info("org          nvidia")
        info("team         no-team")
        info("ace          no-ace")
        info("format_type  ascii")
        return 0
    if sub == "user":
        info("Email      bbhasin@gmail.com")
        info("Org        nvidia (NVIDIA Corp)")
        info("Team       no-team")
        info("Roles      NGC_REGISTRY_USER, NGC_PRIVATE_REGISTRY_READ")
        return 0
    if sub == "orgs":
        info("Org Name        Display Name")
        info("nvidia          NVIDIA Corp")
        info("ea-bignlp       NLP Early Access")
        return 0
    if sub == "registry":
        if rest and rest[0] == "image":
            if len(rest) > 1 and rest[1] == "list":
                info("nvcr.io/nvidia/pytorch:24.03-py3       PyTorch + CUDA 12.4")
                info("nvcr.io/nvidia/tensorflow:24.03-tf2    TensorFlow 2 + CUDA 12.4")
                info("nvcr.io/nvidia/tritonserver:24.03      Triton Inference Server")
                return 0
            if len(rest) > 1 and rest[1] == "info":
                tag = rest[2] if len(rest) > 2 else "<image>"
                info(f"Image: {tag}")
                info("Architecture: amd64")
                info("Last updated: 2026-04-01")
                return 0
        if rest and rest[0] == "resource":
            info("Listing NGC catalog resources ...")
            info("nvidia/clara-imaging         1.2.0   medical-imaging models")
            info("nvidia/megatron-bert-345m    1.0     pretrained BERT base")
            return 0
    if sub == "model":
        if rest and rest[0] == "list":
            info("Name                              Version  Size")
            info("nvidia/megatron-bert-345m         1.0      2.4 GB")
            info("nvidia/clara-imaging              1.2.0    780 MB")
            return 0
        if rest and rest[0] == "download-version":
            target = rest[1] if len(rest) > 1 else "<model>:<v>"
            info(f"Downloading model: {target}")
            for f in ["config.json", "model.ckpt", "tokenizer.json", "vocab.txt"]:
                info(f"  Saving file: {f}")
            info(f"Successfully downloaded {target} to ./{target.split('/')[-1].split(':')[0]}")
            return 0
    return err(f"ngc: unknown subcommand '{sub}'")


def cmd_ib_write_bw(args: list[str], s: dict) -> int:
    info("---------------------------------------------------------------------------------------")
    info("                    RDMA_Write BW Test")
    info(" Dual-port       : OFF          Device         : mlx5_0")
    info(" Number of qps   : 1            Transport type : IB")
    info(" Connection type : RC           Using SRQ      : OFF")
    info(" TX depth        : 128          Mtu            : 4096[B]")
    info(" Link type       : IB           GID index      : 3")
    info("---------------------------------------------------------------------------------------")
    info(" #bytes  #iterations  BW peak[Gb/sec]  BW average[Gb/sec]  MsgRate[Mpps]")
    info(" 65536   1000         197.45           196.83              0.375")
    info(" 1048576 1000         199.12           198.61              0.024")
    info("---------------------------------------------------------------------------------------")
    return 0


def cmd_ib_read_bw(args: list[str], s: dict) -> int:
    info(" RDMA_Read BW Test  -->  peak 198.21 Gb/s   avg 197.62 Gb/s")
    return 0


def cmd_ib_send_bw(args: list[str], s: dict) -> int:
    info(" RDMA_Send BW Test  -->  peak 195.04 Gb/s   avg 194.32 Gb/s")
    return 0


def cmd_perftest(args: list[str], s: dict) -> int:
    info("Tools available: ib_write_bw  ib_read_bw  ib_send_bw  ib_atomic_bw  ib_write_lat  ib_read_lat")
    return 0


# ----- Extend nvidia-smi for MIG / nvlink / dmon / --gpu-reset -----
def nvidia_smi_mig_nvlink_extension(args: list[str], s: dict) -> Optional[int]:
    if not args: return None
    # nvidia-smi -mig 1   (enable MIG mode)
    if args[0] == "-mig":
        if len(args) > 1 and args[1] in ("1", "ENABLE"):
            info("Enabled MIG Mode for GPU 00000000:1B:00.0")
            info("All done.")
            s["mig_enabled"] = True; save_state(s); return 0
        if len(args) > 1 and args[1] in ("0", "DISABLE"):
            info("Disabled MIG Mode for GPU 00000000:1B:00.0")
            s["mig_enabled"] = False; save_state(s); return 0

    # nvidia-smi mig -lgip / -cgi / -cci / -lgi / -lci / -dgi / -dci
    if args[0] == "mig":
        sub = args[1] if len(args) > 1 else "-lgip"
        if sub in ("-lgip", "--list-gpu-instance-profiles"):
            info("+-------------------------------------------------------------------------+")
            info("| GPU instance profiles:                                                  |")
            info("| GPU  Name           ID  Instances     Memory  P2P    SM   DEC  ENC  CE  |")
            info("|                          Free/Total    GiB                    JPEG  OFA |")
            info("|=========================================================================|")
            info("|  0   MIG 1g.10gb    19      7/7        9.75   No   14     0    0    1   |")
            info("|  0   MIG 1g.10gb+me 20      1/1        9.75   No   14     1    0    1   |")
            info("|  0   MIG 1g.20gb    15      4/4       19.62   No   14     1    0    1   |")
            info("|  0   MIG 2g.20gb    14      3/3       19.50   No   28     1    0    2   |")
            info("|  0   MIG 3g.40gb     9      2/2       39.25   No   42     2    0    3   |")
            info("|  0   MIG 4g.40gb     5      1/1       39.25   No   56     2    0    4   |")
            info("|  0   MIG 7g.80gb     0      1/1       79.25   No   98     5    0    7   |")
            info("+-------------------------------------------------------------------------+")
            return 0
        if sub in ("-cgi", "--create-gpu-instance"):
            profile = args[2] if len(args) > 2 else "19"
            info(f"Successfully created GPU instance ID 1 on GPU 0 using profile "
                 f"MIG (ID {profile})")
            return 0
        if sub in ("-cci", "--create-compute-instance"):
            info("Successfully created compute instance ID 0 on GPU 0 "
                 "GPU instance ID 1 using profile MIG (ID 0)")
            return 0
        if sub in ("-lgi", "--list-gpu-instances"):
            info("+----------------------------------------------------------+")
            info("| GPU instances:                                           |")
            info("| GPU   Name           Profile  Instance     Placement     |")
            info("|                        ID       ID         Start:Size    |")
            info("|==========================================================|")
            info("|   0   MIG 1g.10gb     19         1            0:1        |")
            info("|   0   MIG 1g.10gb     19         2            1:1        |")
            info("+----------------------------------------------------------+")
            return 0
        if sub in ("-lci", "--list-compute-instances"):
            info("+--------------------------------------------------------+")
            info("| Compute instances:                                     |")
            info("| GPU   GPU-Inst   Name        Profile     Compute-Inst  |")
            info("|         ID                     ID            ID        |")
            info("|========================================================|")
            info("|   0      1     MIG 1g.10gb    0              0         |")
            info("+--------------------------------------------------------+")
            return 0
        if sub in ("-dgi", "--destroy-gpu-instance"):
            info("Successfully destroyed GPU instance ID 1 on GPU 0")
            return 0
        if sub in ("-dci", "--destroy-compute-instance"):
            info("Successfully destroyed compute instance ID 0")
            return 0

    # nvidia-smi nvlink --status / --errors
    if args[0] == "nvlink":
        if "--errors" in args or "-e" in args:
            info("GPU 0: NVIDIA H100 80GB HBM3")
            for link in range(18):
                info(f"   Link {link}: Replay Errors: 0   Recovery Errors: 0   Flit CRC Errors: 0")
            return 0
        # default --status
        info("GPU 0: NVIDIA H100 80GB HBM3 (UUID: GPU-3a51)")
        for link in range(18):
            info(f"   Link {link}: 25.781 GB/s   active")
        return 0

    # nvidia-smi dmon  (continuous device monitor)
    if args[0] == "dmon":
        info("# gpu   pwr  gtemp  mtemp    sm   mem   enc   dec  mclk  pclk")
        info("# Idx     W      C      C     %     %     %     %   MHz   MHz")
        for _ in range(5):
            info(f"    0   {random.randint(120, 380):>3}     "
                 f"{random.randint(45, 75):>2}     "
                 f"{random.randint(45, 78):>2}    "
                 f"{random.randint(0, 95):>3}    "
                 f"{random.randint(10, 90):>3}     0     0  1593  1980")
        return 0

    # nvidia-smi --gpu-reset
    if args[0] in ("--gpu-reset", "-r"):
        info("GPU 00000000:1B:00.0 was successfully reset.")
        return 0

    # nvidia-smi pmon  (process monitor)
    if args[0] == "pmon":
        info("# gpu        pid  type    sm   mem   enc   dec   command")
        info("# Idx          #   C/G     %     %     %     %   name")
        info("    0     204131     C    87    52     0     0   python")
        return 0

    return None


# Hook the new MIG/nvlink/dmon checks into nvidia-smi v2
_pre_v2_cmd_nvidia_smi = cmd_nvidia_smi_v2
def cmd_nvidia_smi_v3(args: list[str], s: dict) -> int:
    rc = nvidia_smi_mig_nvlink_extension(args, s)
    if rc is not None:
        return rc
    return _pre_v2_cmd_nvidia_smi(args, s)


# ----- Extend `docker` with `login` and `network inspect` -----
_original_cmd_docker = cmd_docker
def cmd_docker_v2(args: list[str], s: dict) -> int:
    if args and args[0] == "login":
        # Forms: docker login nvcr.io OR docker login nvcr.io -u $oauthtoken -p <KEY>
        registry = next((a for a in args[1:] if not a.startswith("-")), "docker.io")
        info(f"Login Succeeded — credentials saved in /root/.docker/config.json (registry={registry})")
        return 0
    if args and args[0] == "network":
        sub = args[1] if len(args) > 1 else "ls"
        if sub == "ls":
            info("NETWORK ID    NAME      DRIVER    SCOPE")
            info("a1b2c3d4e5    bridge    bridge    local")
            info("f6g7h8i9j0    host      host      local")
            info("k1l2m3n4o5    none      null      local")
            return 0
        if sub == "inspect":
            target = args[2] if len(args) > 2 else "bridge"
            info(f"[")
            info(f"    {{")
            info(f"        \"Name\": \"{target}\",")
            info(f"        \"Id\": \"a1b2c3d4...\",")
            info(f"        \"Driver\": \"bridge\",")
            info(f"        \"Scope\": \"local\",")
            info(f"        \"IPAM\": {{ \"Config\": [{{ \"Subnet\": \"172.17.0.0/16\" }}] }},")
            info(f"        \"Containers\": {{}}")
            info(f"    }}")
            info(f"]")
            return 0
    return _original_cmd_docker(args, s)


# ===========================================================================
# Exam-style additions: kubectl set, kubeadm, journalctl, runai suspend/resume,
#                       nvidia-smi -q -d, ipmitool sol, kubectl create secret
# ===========================================================================
def kubectl_set(args: list[str], s: dict) -> int:
    if not args: return err("usage: kubectl set {image|env|resources} ...")
    sub = args[0]
    if sub == "image":
        # kubectl set image deployment/triton triton=nvcr.io/.../tritonserver:24.06
        target = args[1] if len(args) > 1 else "deploy/<name>"
        kind, _, name = target.partition("/") if "/" in target else ("deployment", "/", target)
        change = args[2] if len(args) > 2 else "<container>=<image>"
        info(f"deployment.apps/{name or target} image updated")
        return 0
    if sub == "env":
        target = args[1] if len(args) > 1 else "<deploy>"
        info(f"deployment.apps/{target} env updated"); return 0
    if sub == "resources":
        target = args[1] if len(args) > 1 else "<deploy>"
        info(f"deployment.apps/{target} resources requirements updated"); return 0
    return err(f"kubectl set: unknown subcommand '{sub}'")


def cmd_kubeadm(args: list[str], s: dict) -> int:
    if not args: return err("usage: kubeadm {init|token|join|reset|version|upgrade}")
    sub = args[0]
    if sub == "version":
        info("kubeadm version: &version.Info{Major:\"1\", Minor:\"28\", "
             "GitVersion:\"v1.28.5\", BuildDate:\"2026-01-12T14:32:08Z\", Compiler:\"gc\"}")
        return 0
    if sub == "init":
        info("[init] Using Kubernetes version: v1.28.5")
        info("[preflight] Running pre-flight checks")
        info("[certs] Generating ca, apiserver, etcd certificates")
        info("[control-plane] Created static Pod manifest for kube-apiserver")
        info("[control-plane] Created static Pod manifest for kube-controller-manager")
        info("[control-plane] Created static Pod manifest for kube-scheduler")
        info("[etcd] Created static Pod manifest for local etcd")
        info("[apiclient] All control plane components are healthy after 18.502 seconds")
        info("[bootstrap-token] Using token: abcdef.0123456789abcdef")
        info("")
        info("Your Kubernetes control-plane has initialized successfully!")
        info("")
        info("To start using your cluster, you need to run the following as a regular user:")
        info("  mkdir -p $HOME/.kube")
        info("  sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config")
        info("  sudo chown $(id -u):$(id -g) $HOME/.kube/config")
        info("")
        info("Then you can join any number of worker nodes by running:")
        info("kubeadm join 10.141.0.1:6443 --token abcdef.0123456789abcdef \\")
        info("    --discovery-token-ca-cert-hash sha256:1234567890abcdef")
        return 0
    if sub == "token":
        op = args[1] if len(args) > 1 else "list"
        if op == "create":
            if "--print-join-command" in args:
                info("kubeadm join 10.141.0.1:6443 --token abcdef.0123456789abcdef \\")
                info("    --discovery-token-ca-cert-hash "
                     "sha256:1234567890abcdef1234567890abcdef1234567890abcdef")
            else:
                info("abcdef.0123456789abcdef")
            return 0
        if op == "list":
            info("TOKEN                     TTL         EXPIRES                   USAGES")
            info("abcdef.0123456789abcdef  23h         2026-05-01 12:00:00 UTC   authentication,signing")
            return 0
        if op == "delete":
            info(f"bootstrap token deleted: {args[2] if len(args) > 2 else '<token>'}"); return 0
    if sub == "join":
        info("[preflight] Running pre-flight checks")
        info("[preflight] Reading configuration from the cluster")
        info("[kubelet-start] Writing kubelet configuration to /var/lib/kubelet/config.yaml")
        info("[kubelet-start] Starting the kubelet")
        info("This node has joined the cluster:")
        info("* Certificate signing request was sent to apiserver and a response was received.")
        info("* The Kubelet was informed of the new secure connection details.")
        info("Run 'kubectl get nodes' on the control-plane to see this node join the cluster.")
        return 0
    if sub == "reset":
        info("[reset] Reading configuration from the cluster")
        info("[reset] Stopping the kubelet service")
        info("[reset] Removing the cluster from /etc/kubernetes/")
        info("[reset] Cleaning up etcd data")
        return 0
    if sub == "upgrade":
        info("[upgrade] Performing pre-flight checks")
        info("[upgrade] Successfully upgraded to v1.28.5")
        return 0
    return err(f"kubeadm: unknown subcommand '{sub}'")


def cmd_journalctl(args: list[str], s: dict) -> int:
    unit, follow, lines = None, False, 50
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("-u", "--unit") and i + 1 < len(args):
            unit = args[i+1]; i += 2; continue
        if a in ("-f", "--follow"):
            follow = True; i += 1; continue
        if a in ("-n", "--lines") and i + 1 < len(args):
            try: lines = int(args[i+1])
            except ValueError: pass
            i += 2; continue
        i += 1

    canned: dict[str, list[str]] = {
        "slurmctld": [
            "slurmctld: agent/is_node_resp: node node-007 not responding",
            "slurmctld: drain_nodes: node node-007 state set to DRAIN: NHC: GPU ECC threshold",
            "slurmctld: backfill scheduling complete: 4 jobs",
            "slurmctld: epilog complete for JobId=12471 NodeList=node-001",
            "slurmctld: error: Munge encode failed: Invalid credential",
        ],
        "slurmd": [
            "slurmd: gres/gpu device count of 8 lower than reported in conf",
            "slurmd: NHC: GPU ECC error rate 14/hr exceeds threshold",
            "slurmd: NHC: failure (rc=2) — node will be drained",
            "slurmd: error: Failed to create cgroup",
            "slurmd: prolog complete for JobId=12471",
        ],
        "kubelet": [
            "kubelet: pod gpu-pod admitted",
            "kubelet: ImagePull succeeded for nvcr.io/nvidia/cuda:12.6.2-base-ubuntu22.04",
            "kubelet: Started container cuda-container",
            "kubelet: Container nvidia-device-plugin restarted: ExitCode 1",
            "kubelet: PLEG is not healthy: pleg was last seen active 3m20s ago",
        ],
        "nvidia-fabricmanager": [
            "fabricmanager: detected NVSwitch domain 0",
            "fabricmanager: 6 NVSwitches initialized successfully",
            "fabricmanager: all GPUs are connected to NVSwitches",
            "fabricmanager: ready for service",
            "fabricmanager: GPU 0 NVLink port 2 reports recovery error",
        ],
        "runai-scheduler": [
            "runai-scheduler: project ml-research over guaranteed quota",
            "runai-scheduler: preempted Job 'interactive-bob' to free 8 GPUs",
            "runai-scheduler: scheduling Job 'big-train' (24 GPUs)",
            "runai-scheduler: training-priority class reclaim from interactive class",
        ],
        "docker": [
            "dockerd: Daemon has completed initialization",
            "dockerd: API listen on /var/run/docker.sock",
            "dockerd: error: Could not select device driver \"nvidia\"",
        ],
        "nvidia-container-toolkit": [
            "nvidia-container-toolkit: configured runtime for /etc/docker/daemon.json",
        ],
        "etcd": [
            "etcd: starting server",
            "etcd: published {Name:default ClientURLs:[https://10.141.0.1:2379]}",
            "etcd: ready to serve client requests",
        ],
    }
    if not unit:
        info(f"-- Logs begin at {NOW()} --")
        info(f"{NOW()} bcm systemd[1]: Started slurmctld Slurm controller daemon")
        info(f"{NOW()} bcm systemd[1]: Started kubelet kubelet")
        info(f"{NOW()} bcm systemd[1]: Started nvidia-fabricmanager")
        return 0
    log_lines = canned.get(unit,
                           [f"{unit}: no canned log available (service may not exist)"])
    for ln in log_lines[-lines:]:
        info(f"{NOW()} bcm {ln}")
    if follow:
        info("(--follow mode: would block until Ctrl-C in a real shell)")
    return 0


# Patch nvidia-smi v3 to also handle `-q`/`--query` (verbose queries)
_pre_v3_cmd_nvidia_smi = cmd_nvidia_smi_v3
def cmd_nvidia_smi_v4(args: list[str], s: dict) -> int:
    if args and args[0] in ("-q", "--query"):
        domains = []
        for i, a in enumerate(args):
            if a == "-d" and i + 1 < len(args):
                domains = [d.strip().upper() for d in args[i+1].split(",")]; break
        info("==============NVSMI LOG==============")
        info(f"Timestamp                                : {NOW()}")
        info("Driver Version                           : 570.86.15")
        info("CUDA Version                             : 12.8")
        info("Attached GPUs                            : 1")
        info("")
        info("GPU 00000000:1B:00.0")
        info("    Product Name                         : NVIDIA H100 NVL")
        info("    Persistence Mode                     : Disabled")
        if not domains or "ECC" in domains:
            info("    ECC Errors")
            info("        Volatile")
            info("            SRAM Correctable             : 0")
            info("            SRAM Uncorrectable           : 0")
            info("            DRAM Correctable             : 0")
            info("            DRAM Uncorrectable           : 0")
            info("        Aggregate")
            info("            SRAM Correctable             : 12")
            info("            SRAM Uncorrectable           : 0")
            info("            DRAM Correctable             : 0")
            info("            DRAM Uncorrectable           : 0")
        if not domains or "TEMPERATURE" in domains:
            info("    Temperature")
            info("        GPU Current Temp                 : 51 C")
            info("        GPU T.Limit Temp                 : 32 C")
            info("        GPU Shutdown Temp                : 95 C")
            info("        GPU Slowdown Temp                : 90 C")
            info("        Memory Current Temp              : 49 C")
        if not domains or "POWER" in domains:
            info("    Power Readings")
            info("        Power Draw                       : 124.42 W")
            info("        Power Limit                      : 400.00 W")
        if not domains or "CLOCK" in domains:
            info("    Clocks")
            info("        Graphics                         : 1980 MHz")
            info("        SM                               : 1980 MHz")
            info("        Memory                           : 2619 MHz")
        return 0
    return _pre_v3_cmd_nvidia_smi(args, s)


# Extend ipmitool with `sol activate/info/deactivate`
_original_cmd_ipmitool = cmd_ipmitool
def cmd_ipmitool_v2(args: list[str], s: dict) -> int:
    if "sol" in args:
        if "activate" in args:
            info("[SOL Session operational. Use ~? for help]")
            info("Welcome to the BCM head node serial console")
            info("(simulated SOL — type ~. to disconnect in a real shell)")
            return 0
        if "deactivate" in args:
            info("[SOL session deactivated]"); return 0
        if "info" in args:
            info("Set in progress                 : set-complete")
            info("Enabled                         : true")
            info("Force Encryption                : false")
            info("Privilege Level                 : USER")
            info("Volatile Bit Rate (kbps)        : 115.2")
            info("Non-Volatile Bit Rate (kbps)    : 115.2")
            info("Payload Channel                 : 1 (0x01)")
            return 0
    return _original_cmd_ipmitool(args, s)


# Extend kubectl_create to handle `secret docker-registry|generic|tls`
_original_kubectl_create = kubectl_create
def kubectl_create_v2(args: list[str], s: dict) -> int:
    if args and args[0] == "secret" and len(args) > 1:
        secret_type = args[1]   # docker-registry / generic / tls
        name = args[2] if len(args) > 2 else "<name>"
        info(f"secret/{name} created")
        return 0
    if args and args[0] == "configmap" and len(args) > 1:
        name = args[1]
        info(f"configmap/{name} created"); return 0
    if args and args[0] == "clusterrole":
        name = args[1] if len(args) > 1 else "<name>"
        info(f"clusterrole.rbac.authorization.k8s.io/{name} created"); return 0
    if args and args[0] == "clusterrolebinding":
        name = args[1] if len(args) > 1 else "<name>"
        info(f"clusterrolebinding.rbac.authorization.k8s.io/{name} created"); return 0
    if args and args[0] == "serviceaccount":
        name = args[1] if len(args) > 1 else "<name>"
        info(f"serviceaccount/{name} created"); return 0
    if args and args[0] == "ingress":
        name = args[1] if len(args) > 1 else "<name>"
        info(f"ingress.networking.k8s.io/{name} created"); return 0
    return _original_kubectl_create(args, s)


# Patch the existing kubectl dispatch to use create_v2 + set
def kubectl_extras(sub: str, rest: list[str], s: dict) -> Optional[int]:
    if sub == "set":
        return kubectl_set(rest, s)
    if sub == "create":
        return kubectl_create_v2(rest, s)
    if sub in ("annotate", "patch"):
        target = rest[0] if rest else "<resource>"
        info(f"{target} annotated" if sub == "annotate" else f"{target} patched")
        return 0
    return None


# Wrap cmd_kubectl to invoke kubectl_extras before falling back
_original_cmd_kubectl = cmd_kubectl
def cmd_kubectl_v2(args: list[str], s: dict) -> int:
    if args and not args[0].startswith("-"):
        rc = kubectl_extras(args[0], args[1:], s)
        if rc is not None:
            return rc
    # `kubectl delete pod x --force --grace-period=0`
    if args and args[0] == "delete" and "pod" in args:
        idx = args.index("pod")
        if idx + 1 < len(args):
            pod_name = args[idx + 1]
            s["pods"] = [p for p in s["pods"] if p["name"] != pod_name]
            save_state(s)
            info(f"pod \"{pod_name}\" force deleted (grace period 0)")
            return 0
    return _original_cmd_kubectl(args, s)


# Run:ai suspend / resume
_original_cmd_runai_extras = cmd_runai_extras
def cmd_runai_extras_v2(args: list[str], s: dict) -> Optional[int]:
    if args and args[0] in ("suspend", "resume"):
        op = args[0]
        if len(args) < 2: return err(f"usage: runai {op} <job>")
        name = args[1]
        j = next((x for x in s["runai_jobs"] if x["name"] == name), None)
        if not j: return err(f"job '{name}' not found")
        j["status"] = "Suspended" if op == "suspend" else "Running"
        save_state(s)
        verb = {"suspend": "suspended", "resume": "resumed"}[op]
        info(f"INFO[{NOW()}] Job '{name}' {verb} successfully")
        return 0
    if args and args[0] == "list" and len(args) > 1 and args[1] == "nodepools":
        rows = [["nvidia-h100",       "default", "16", "128", "available"],
                ["nvidia-l40s-vgpu",  "default", "1",  "1",   "available"]]
        print(table(["NAME", "ROLE", "NODES", "GPUs", "STATUS"], rows))
        return 0
    return _original_cmd_runai_extras(args, s)


# Override cmd_runai to use the new extras dispatcher
_existing_cmd_runai = cmd_runai
def cmd_runai_v2(args: list[str], s: dict) -> int:
    extras_rc = cmd_runai_extras_v2(args, s) if args else None
    if extras_rc is not None:
        return extras_rc
    return _existing_cmd_runai(args, s)


# Make `sbatch` tolerant of the rich CLI flag set the exam uses
_original_cmd_sbatch = cmd_sbatch
def cmd_sbatch_v2(args: list[str], s: dict) -> int:
    """Accepts --array, --constraint, --exclusive, --dependency, --gres, etc.
    Strips them down to the script path, then delegates to the original."""
    array = None
    script_path = None
    cleaned: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a.startswith("--array="):
            array = a.split("=", 1)[1]; i += 1; continue
        if a.startswith("--") and "=" in a:
            cleaned.append(a); i += 1; continue
        if a.startswith("--") and i + 1 < len(args) and not args[i+1].startswith("-"):
            cleaned.append(a); cleaned.append(args[i+1]); i += 2; continue
        if a.startswith("-") and len(a) == 2 and i + 1 < len(args) \
                and not args[i+1].startswith("-"):
            cleaned.append(a); cleaned.append(args[i+1]); i += 2; continue
        if a.startswith("-"):
            cleaned.append(a); i += 1; continue
        if script_path is None:
            script_path = a
        i += 1
    if not script_path:
        return err("usage: sbatch [flags] <script.sh>")
    rc = _original_cmd_sbatch([script_path], s)
    if array:
        info(f"(array job: tasks {array} — each runs the same script with $SLURM_ARRAY_TASK_ID)")
    return rc


# ===========================================================================
# Mock Exam — 120 minutes, 30 weighted multiple-choice + 3 hands-on labs
# Mirrors the official NCP-AIOL blueprint: 31/23/23/23 by domain.
# ===========================================================================
EXAM_QUESTIONS = [
    # ---------- Installation & Deployment (target weight 31%) ----------
    {"d": "I&D", "q": "Which BCM CLI command lists all physical nodes and their current MAC, IP and status in the cluster?",
     "o": ["A. cmsh -c 'category list'", "B. cmsh -c 'device list'",
           "C. mhcheck", "D. ndlist --status"],
     "a": "B", "e": "`device list` in cmsh shows hostname, MAC, category, IP, and status for every node."},
    {"d": "I&D", "q": "An admin runs `cm-kubernetes-setup` on the head node. What does this wizard configure?",
     "o": ["A. Slurm partitions and accounting", "B. Kubernetes control plane, network plugin, GPU operator",
           "C. BCM software image cloning", "D. Run:ai cluster CRDs only"],
     "a": "B", "e": "cm-kubernetes-setup is BCM's wizard that initializes Kubernetes (CNI, GPU operator, dashboards) on the cluster."},
    {"d": "I&D", "q": "After installing the NVIDIA Container Toolkit, which command configures Docker to use the nvidia runtime?",
     "o": ["A. nvidia-ctk runtime configure", "B. docker network create nvidia",
           "C. nvidia-smi --runtime=docker", "D. systemctl enable nvidia-runtime"],
     "a": "A", "e": "`nvidia-ctk runtime configure` writes /etc/docker/daemon.json so Docker can use the nvidia container runtime."},
    {"d": "I&D", "q": "Which file is the primary location for Slurm partition definitions on a BCM-managed cluster?",
     "o": ["A. /etc/slurm/gres.conf", "B. /etc/slurm/slurm.conf",
           "C. /etc/slurm/topology.conf", "D. /etc/cm/wlm.yaml"],
     "a": "B", "e": "slurm.conf is the master Slurm configuration file where partitions, nodes, and scheduling are defined."},
    {"d": "I&D", "q": "Which command initializes the Kubernetes control plane on a worker that BCM has provisioned?",
     "o": ["A. kubeadm init", "B. kubectl bootstrap",
           "C. nvidia-kube-init", "D. cm-kube-start"],
     "a": "A", "e": "kubeadm init is the upstream Kubernetes way to initialize a control plane node; BCM's wizard wraps this."},
    {"d": "I&D", "q": "Within cmsh you ran `softwareimage clone default-image gpu-image; commit`. What is the next step to make node004 use the new image?",
     "o": ["A. Reboot node004 immediately",
           "B. Set the node's category softwareimage to gpu-image, commit, then imageupdate -n node004 -w",
           "C. Edit /etc/hosts on the head node",
           "D. Run cm-kubernetes-setup --add-image"],
     "a": "B", "e": "The node inherits the image from its category; change the category's softwareimage and trigger imageupdate to provision."},
    {"d": "I&D", "q": "What does the `cm-chroot-sw-img /cm/images/default-image` command let you do?",
     "o": ["A. Boot a node into rescue mode",
           "B. Open a chroot shell INSIDE the software image so you can edit it before it's pushed to nodes",
           "C. SSH into a running compute node",
           "D. Generate a fresh ramdisk for the image"],
     "a": "B", "e": "cm-chroot-sw-img drops you into a chroot at the image root, so files you create or packages you install end up in the image."},
    {"d": "I&D", "q": "Which DOCA component is responsible for offloading and accelerating packet processing on a BlueField-3 DPU?",
     "o": ["A. DOCA DMA", "B. DOCA Flow", "C. DOCA Firefly", "D. DOCA Comm Channel"],
     "a": "B", "e": "DOCA Flow handles packet processing and network function offload on the DPU Arm."},
    {"d": "I&D", "q": "Which Helm command installs the Run:ai cluster chart from the runai repo into namespace runai?",
     "o": ["A. helm install runai-cluster runai/runai-cluster -n runai",
           "B. helm bootstrap runai --namespace runai",
           "C. kubectl apply -f https://run-ai-charts.../runai.yaml",
           "D. cmsh -c 'wlm; install runai'"],
     "a": "A", "e": "Standard Helm syntax — `helm install <release> <chart> -n <namespace>`."},
    {"d": "I&D", "q": "Which BCM command quickly verifies that DNS, auth, slurmctld and node health are all OK on the head node?",
     "o": ["A. cm-info", "B. cmha status", "C. mhcheck", "D. ndlist"],
     "a": "C", "e": "`mhcheck` is BCM's master health check, returning pass/fail for the cluster's core services."},
    {"d": "I&D", "q": "An administrator wants to check a single comprehensive cluster summary including BCM version, head node, node count, workload manager, and GPU details. Which command provides this?",
     "o": ["A. cm-info", "B. cmsh -c 'main status'", "C. cmha status", "D. mhcheck"],
     "a": "A", "e": "`cm-info` prints a one-screen cluster summary."},

    # ---------- Administration (target weight 23%) ----------
    {"d": "ADM", "q": "What is the FIRST step required before creating any MIG GPU instance on an H100?",
     "o": ["A. Reboot the node", "B. Run `nvidia-smi -mig 1` to enable MIG mode",
           "C. Install a separate MIG driver", "D. Run `dcgmi diag -r 4`"],
     "a": "B", "e": "MIG mode must be enabled with `nvidia-smi -mig 1` before any GPU instances can be created."},
    {"d": "ADM", "q": "Which MIG profile creates the maximum number of independent GPU instances on a single H100 80GB?",
     "o": ["A. 7g.80gb", "B. 4g.40gb", "C. 1g.10gb", "D. 3g.40gb"],
     "a": "C", "e": "1g.10gb is the smallest profile; seven of them fit on an H100 80GB, the maximum."},
    {"d": "ADM", "q": "In Run:ai, which feature lets a project consume MORE GPUs than its guaranteed quota when the cluster has spare capacity?",
     "o": ["A. NodePool pinning", "B. Over-quota (deserved GPU)",
           "C. Burst preemption", "D. Priority class"],
     "a": "B", "e": "Run:ai's over-quota mechanism allows projects to use spare GPUs above their guarantee, subject to fairshare."},
    {"d": "ADM", "q": "An administrator wants to drain Slurm node-007 for PSU replacement so existing jobs finish but no new jobs start there. Which command?",
     "o": ["A. scontrol update NodeName=node-007 State=DRAIN Reason=PSU",
           "B. scancel node-007",
           "C. squeue --drain node-007",
           "D. sacctmgr modify node node-007 set state=down"],
     "a": "A", "e": "`scontrol update ... State=DRAIN` puts a node into drain — running jobs continue, no new jobs are scheduled."},
    {"d": "ADM", "q": "Which sacctmgr command lists all configured QoS levels in Slurm?",
     "o": ["A. sacctmgr list qos", "B. sacctmgr show qos",
           "C. sinfo --qos", "D. scontrol show qos"],
     "a": "A", "e": "`sacctmgr list qos` enumerates the QoS records (priority, MaxJobs, MaxWall, etc.)."},
    {"d": "ADM", "q": "Which Kubernetes object enforces a per-team GPU allocation cap inside a namespace?",
     "o": ["A. PodSecurityPolicy", "B. ResourceQuota with `nvidia.com/gpu` limit",
           "C. NetworkPolicy", "D. ClusterRoleBinding"],
     "a": "B", "e": "ResourceQuota in a namespace can limit `nvidia.com/gpu` so a team can't exceed its allocation."},
    {"d": "ADM", "q": "Inside cmsh user mode you typed `add slurmy; commit`. What further step is required before slurmy can log into the cluster?",
     "o": ["A. Nothing — login is enabled immediately",
           "B. Run `set password <pw>; commit` to give the user a password",
           "C. Add slurmy to the wheel group via /etc/sudoers",
           "D. Reboot the head node"],
     "a": "B", "e": "BCM users have no password by default; the `set password` step is required before login is possible."},
    {"d": "ADM", "q": "An admin runs `cmsh -c 'device set node003 category Lite; commit'`. What's the effect?",
     "o": ["A. node003 reboots immediately",
           "B. node003's category changes to Lite — on next provision, it inherits Lite's software image",
           "C. node003 is put into maintenance",
           "D. node003 leaves the cluster"],
     "a": "B", "e": "Setting a node's category changes its inherited image; provisioning on next reboot or imageupdate applies it."},

    # ---------- Workload Management (target weight 23%) ----------
    {"d": "WLM", "q": "Which sbatch line correctly requests 4 nodes with 8 GPUs each (32 total)?",
     "o": ["A. #SBATCH --nodes=4 --gres=gpu:8",
           "B. #SBATCH --gpus=32",
           "C. #SBATCH --node-count=4 --gpu=8",
           "D. #SBATCH --reserve gpu*32"],
     "a": "A", "e": "`--nodes=4 --gres=gpu:8` is the canonical Slurm syntax — 4 nodes × 8 GPUs each."},
    {"d": "WLM", "q": "Which command authenticates the docker daemon to NGC's container registry?",
     "o": ["A. ngc auth login --apikey <KEY>",
           "B. docker login nvcr.io -u '$oauthtoken' -p <NGC_API_KEY>",
           "C. nvidia-docker authenticate <KEY>",
           "D. kubectl create secret nvcr <KEY>"],
     "a": "B", "e": "Docker uses the literal username `$oauthtoken` with the NGC API key as the password to authenticate to nvcr.io."},
    {"d": "WLM", "q": "Which NGC CLI command downloads a versioned model artifact?",
     "o": ["A. ngc model download-version <org>/<model>:<v>",
           "B. ngc pull model <model>",
           "C. ngc registry fetch <model>",
           "D. wget https://ngc.nvidia.com/models/<model>"],
     "a": "A", "e": "`ngc model download-version` is the official NGC CLI command for downloading a specific model version."},
    {"d": "WLM", "q": "How do you push a new image into a running Kubernetes Deployment with zero downtime?",
     "o": ["A. kubectl set image deployment/triton triton=<new-image>",
           "B. kubectl replace -f deployment.yaml --force",
           "C. kubectl restart deployment/triton",
           "D. kubectl edit deployment/triton --image=<new-image>"],
     "a": "A", "e": "`kubectl set image` triggers a rolling update — old pods drained as new pods come up healthy."},
    {"d": "WLM", "q": "Which Kubernetes object is correct for a stateless inference service that should always run exactly 3 replicas?",
     "o": ["A. StatefulSet", "B. Job", "C. Deployment", "D. DaemonSet"],
     "a": "C", "e": "Deployment manages a stateless replicated pod set, ensuring the desired replica count is maintained."},
    {"d": "WLM", "q": "A Run:ai admin needs to submit a multi-node distributed training job using MPI. Which command form is correct?",
     "o": ["A. runai submit-mpi train -p <proj> -g <n> --workers 4 -- mpirun python train.py",
           "B. runai submit train --mpi --nodes 4",
           "C. runai mpi train -g 32",
           "D. mpirun --runai train.py"],
     "a": "A", "e": "Run:ai's distributed-MPI form is `runai submit-mpi <name> -p <proj> -g <gpus-per-worker> --workers <n>`."},
    {"d": "WLM", "q": "After `kubectl set image` triggers a rollout, which command monitors progress?",
     "o": ["A. kubectl rollout status deployment/<name>",
           "B. kubectl logs deployment/<name>",
           "C. kubectl get events --watch",
           "D. helm status <release>"],
     "a": "A", "e": "`kubectl rollout status` blocks until the rollout completes (or fails)."},
    {"d": "WLM", "q": "Which Kubernetes object holds the NGC API key so pods can pull from nvcr.io?",
     "o": ["A. ConfigMap", "B. Secret of type docker-registry",
           "C. ServiceAccount", "D. PersistentVolumeClaim"],
     "a": "B", "e": "`kubectl create secret docker-registry` stores the NGC credentials so Kubernetes can pull from nvcr.io."},

    # ---------- Troubleshooting & Optimization (target weight 23%) ----------
    {"d": "T&O", "q": "A multi-GPU training job reports `NCCL WARN Connect to <ip> failed`. What should an operator check FIRST?",
     "o": ["A. GPU driver version",
           "B. Inter-node network reachability and firewall rules on the training network",
           "C. Slurm partition limit",
           "D. Docker image build date"],
     "a": "B", "e": "NCCL connection failures almost always trace back to network/firewall problems on the training fabric."},
    {"d": "T&O", "q": "An H100 logs `NVRM: Xid 79` in dmesg during training. Which severity and action is correct?",
     "o": ["A. Informational — no action",
           "B. CRITICAL — drain the node, run nvidia-bug-report.sh, replace if the error recurs",
           "C. WARNING — bump driver and continue",
           "D. CRITICAL — only reset the GPU; no escalation"],
     "a": "B", "e": "Xid 79 = uncorrectable ECC / GPU fell off bus — drain immediately, generate bug report, escalate to NVIDIA."},
    {"d": "T&O", "q": "The fabric manager service starts but NCCL tests still fail with NVLink errors. Which command shows per-link error counters?",
     "o": ["A. nvidia-smi nvlink --status",
           "B. nvidia-smi nvlink --errors",
           "C. nvidia-smi topo -m",
           "D. dcgmi diag -r 1"],
     "a": "B", "e": "`nvidia-smi nvlink --errors` shows replay/recovery/Flit-CRC counters for each NVLink lane."},
    {"d": "T&O", "q": "An operator suspects an H100 is thermal-throttling. Which command shows live temperature, power, SM and memory utilization?",
     "o": ["A. nvidia-smi dmon",
           "B. nvidia-smi --format=csv",
           "C. dcgmi profile --pause",
           "D. iostat -x 1"],
     "a": "A", "e": "`nvidia-smi dmon` is the device monitor — continuously prints power, temp, SM%, mem% etc."},
    {"d": "T&O", "q": "Which tool directly measures RDMA bandwidth between two nodes to verify GPUDirect RDMA is healthy?",
     "o": ["A. iperf3", "B. perftest suite (e.g. ib_write_bw)",
           "C. netperf", "D. fio"],
     "a": "B", "e": "ib_write_bw / ib_read_bw / ib_send_bw from the perftest suite are the standard RDMA bandwidth tools."},
    {"d": "T&O", "q": "After a failed Triton rollout you need to revert to the previous image quickly. Which command?",
     "o": ["A. kubectl rollout undo deployment/triton",
           "B. kubectl delete pod -l app=triton --force",
           "C. helm uninstall triton",
           "D. kubectl restart deployment/triton"],
     "a": "A", "e": "`kubectl rollout undo` rolls a Deployment back to the previous revision."},
    {"d": "T&O", "q": "A Slurm node is in `drain*` state with reason `NHC: failure`. Which log file is most likely to show the NHC error detail?",
     "o": ["A. journalctl -u slurmd",
           "B. journalctl -u slurmctld",
           "C. /var/log/cuda.log",
           "D. /var/log/syslog only"],
     "a": "A", "e": "NHC runs under slurmd on the compute node, so `journalctl -u slurmd` shows the NHC failure detail."},
    {"d": "T&O", "q": "A `docker run --gpus all` returns `could not select device driver \"nvidia\"`. What's the most likely cause?",
     "o": ["A. The container image is corrupted",
           "B. The NVIDIA Container Toolkit is not installed or Docker is not configured to use the nvidia runtime",
           "C. The GPU is in MIG mode",
           "D. The Docker registry is unreachable"],
     "a": "B", "e": "That error message means Docker can't find the nvidia runtime — usually a missing/unconfigured Container Toolkit."},
    {"d": "T&O", "q": "Storage I/O wait is causing GPU utilization to drop during training. Which Magnum IO component reduces this for read-heavy AI workloads?",
     "o": ["A. NCCL", "B. GPUDirect Storage (GDS)",
           "C. NVLink", "D. DOCA Flow"],
     "a": "B", "e": "GPUDirect Storage transfers data from storage directly into GPU memory, bypassing the CPU and reducing latency."},

    # Spares for randomization variety
    {"d": "I&D", "q": "Which command verifies all Kubernetes system pods came up correctly after install?",
     "o": ["A. kubectl get pods --all-namespaces",
           "B. cm-info --pods",
           "C. nvidia-smi --cluster",
           "D. kubeadm verify"],
     "a": "A", "e": "`kubectl get pods --all-namespaces` shows the state of every pod across the cluster."},
    {"d": "ADM", "q": "Which sacctmgr command modifies an existing user to add the 'high' QoS to their list?",
     "o": ["A. sacctmgr modify user <u> set qos+=high",
           "B. sacctmgr add qos high to user <u>",
           "C. scontrol update User=<u> QOS=high",
           "D. sacctmgr promote <u>"],
     "a": "A", "e": "`set qos+=<name>` is the additive form to extend a user's QoS list."},
    {"d": "WLM", "q": "Which Slurm flag forces a job to run interactively, attaching the operator's terminal to a compute node shell?",
     "o": ["A. srun --pty bash", "B. sbatch --interactive",
           "C. scontrol attach", "D. salloc --tty"],
     "a": "A", "e": "`srun --pty bash` allocates a job and drops you into a bash shell on the assigned node."},
    {"d": "T&O", "q": "An operator must force-delete a pod that's stuck terminating after a node failure. Which flags?",
     "o": ["A. kubectl delete pod <p> --force --grace-period=0",
           "B. kubectl drain <p>",
           "C. kubectl delete pod <p> --immediate",
           "D. kubectl rollout restart pod <p>"],
     "a": "A", "e": "`--force --grace-period=0` removes the pod object immediately, bypassing graceful shutdown."},
]


LAB_TASKS = [
    {
        "title": "Drain node-007 for PSU replacement and restore service afterwards",
        "story": "node-007 is reporting an under-spec 12V PSU rail. Drain it from "
                 "Slurm and Kubernetes so the hardware team can swap the PSU, "
                 "then bring it back online when they're done. Type the commands "
                 "you would run, one per line. Type 'done' when finished.",
        "expected": [
            ("scontrol", "DRAIN", "node-007"),     # drain in slurm
            ("kubectl", "cordon", "node-007"),     # cordon in k8s (or drain)
            ("scontrol", "RESUME", "node-007"),    # resume in slurm
            ("kubectl", "uncordon", "node-007"),   # uncordon in k8s
        ],
        "max_score": 4,
    },
    {
        "title": "Roll out Triton inference image tritonserver:24.06 with zero downtime",
        "story": "Push the new tritonserver:24.06 image to the production Triton "
                 "Deployment in the 'inference' namespace, monitor the rollout, "
                 "and roll back if pods crashloop. Type the commands you would "
                 "run, one per line. Type 'done' when finished.",
        "expected": [
            ("kubectl", "set", "image"),
            ("tritonserver:24.06",),
            ("kubectl", "rollout", "status"),
            ("kubectl", "rollout", "undo"),
        ],
        "max_score": 4,
    },
    {
        "title": "Diagnose XID 79 on node-007 during NCCL training",
        "story": "A training job died with NCCL errors and dmesg shows Xid 79 on "
                 "node-007. Triage: confirm the XID, check ECC counters, run a "
                 "GPU diagnostic, generate a bug report, and drain the node. "
                 "Type the commands, one per line. Type 'done' to submit.",
        "expected": [
            ("dmesg", "xid"),
            ("nvidia-smi",),                       # any nvidia-smi check
            ("dcgmi", "diag"),
            ("nvidia-bug-report.sh",),
            ("scontrol", "DRAIN"),
        ],
        "max_score": 5,
    },
    {
        "title": "Provision a new BCM node and assign it to the gpu-h100 category",
        "story": "A new compute node has just PXE-booted with MAC "
                 "4E:56:44:41:01:05. Bind that MAC to the next available node "
                 "slot, set its category to gpu-h100, and trigger an image "
                 "update. Type the commands, one per line. Type 'done'.",
        "expected": [
            ("cmsh", "device", "set"),
            ("category", "gpu-h100"),
            ("commit",),
            ("imageupdate",),
        ],
        "max_score": 4,
    },
    {
        "title": "Configure MIG on an H100 GPU as 7 × 1g.10gb instances",
        "story": "Partition GPU 0 on the local H100 into the maximum number of "
                 "1g.10gb MIG instances (7), then verify the layout. Type the "
                 "commands you would run. Type 'done' to submit.",
        "expected": [
            ("nvidia-smi", "-mig", "1"),
            ("nvidia-smi", "mig", "-cgi"),
            ("nvidia-smi", "mig", "-cci"),
            ("nvidia-smi", "mig", "-lgi"),
        ],
        "max_score": 4,
    },
    {
        "title": "Set up Run:ai project 'new-team' with a 16-GPU quota and submit a job",
        "story": "Onboard a new team. Create a Run:ai project named 'new-team', "
                 "set its GPU quota to 16, submit an 8-GPU PyTorch training job "
                 "named 'first-train' to it, then verify it's running. Type the "
                 "commands, one per line. Type 'done'.",
        "expected": [
            ("runai", "create", "project"),
            ("runai", "update", "project", "--gpu-quota"),
            ("runai", "submit"),
            ("runai", "list", "jobs"),
        ],
        "max_score": 4,
    },
]


EXAM_QUESTIONS_2 = [
    # ---------- Installation & Deployment ----------
    {"d": "I&D", "q": "What is the purpose of NVIDIA Mission Control?",
     "o": ["A. End-to-end lifecycle management of AI infrastructure clusters",
           "B. Training large language models",
           "C. Edge inference orchestration",
           "D. GPU temperature monitoring only"],
     "a": "A", "e": "Mission Control is NVIDIA's full-lifecycle management toolkit for AI clusters."},
    {"d": "I&D", "q": "Which BCM monitoring object actually executes a periodic script and emits a metric value?",
     "o": ["A. MonitoringDataProducerSingleLineMetricScript",
           "B. MonitoringTrigger",
           "C. MonitoringDashboard",
           "D. MonitoringScriptAction"],
     "a": "A", "e": "DataProducerSingleLineMetricScript is the type that runs a script on an interval and reports the line as a metric value."},
    {"d": "I&D", "q": "Which Slurm config file declares GPU device files and per-node GRES bindings?",
     "o": ["A. /etc/slurm/gres.conf",
           "B. /etc/slurm/slurm.conf",
           "C. /etc/slurm/cgroup.conf",
           "D. /etc/slurm/topology.conf"],
     "a": "A", "e": "gres.conf is where each node's generic resources (GPUs, NICs) and their device files are listed."},
    {"d": "I&D", "q": "Which command on a worker generates the join command for an existing Kubernetes control plane?",
     "o": ["A. kubeadm token create --print-join-command",
           "B. kubectl bootstrap --print-join",
           "C. kubeadm init --print-token",
           "D. kubectl get token"],
     "a": "A", "e": "From the control plane, this prints a ready-to-paste `kubeadm join …` command."},
    {"d": "I&D", "q": "An admin wants to clone the default category and assign a new image to it. Which is the correct sequence inside cmsh?",
     "o": ["A. category; clone default new-cat; set softwareimage <img>; commit",
           "B. category; new-cat; image <img>; commit",
           "C. softwareimage; clone default new-cat; commit",
           "D. device; create category new-cat"],
     "a": "A", "e": "Clone the category, set its software image, then commit — categories are managed under category mode."},
    {"d": "I&D", "q": "Which DOCA component provides a host↔DPU control channel for management traffic?",
     "o": ["A. DOCA Comm Channel",
           "B. DOCA Flow",
           "C. DOCA DMA",
           "D. DOCA Firefly"],
     "a": "A", "e": "DOCA Comm Channel is the bidirectional control-message channel between host and DPU."},
    {"d": "I&D", "q": "What does `mlxfwmanager` show?",
     "o": ["A. Mellanox/NVIDIA NIC and DPU firmware versions",
           "B. NVLink fabric routing tables",
           "C. Slurm jobs allocated to NICs",
           "D. NCCL communicator status"],
     "a": "A", "e": "mlxfwmanager queries device firmware on Mellanox/NVIDIA NICs and DPUs."},
    {"d": "I&D", "q": "After BCM provisions a new node it appears as PhysicalNode but its MAC is still 00:00:00:00:00:00. What does this mean?",
     "o": ["A. The node has been registered but has not yet been associated with a physical MAC",
           "B. The node is permanently broken",
           "C. BCM is misconfigured",
           "D. The node is intentionally a virtual node"],
     "a": "A", "e": "BCM creates the slot first; the MAC is bound during PXE discovery (manual select or readmac.sh)."},
    {"d": "I&D", "q": "Which command, after running cm-wlm-setup, initializes the Slurm module for the user's shell?",
     "o": ["A. module load slurm/slurm/23.02.8",
           "B. systemctl start slurmd",
           "C. cmsh -c 'wlm; activate slurm'",
           "D. source /etc/profile.d/slurm.sh"],
     "a": "A", "e": "cm-wlm-setup installs the slurm module; users load it via `module load slurm/...`."},
    {"d": "I&D", "q": "Which BCM utility writes the per-node `nvidia-driver-local` keyring after dpkg install?",
     "o": ["A. sudo cp /var/nvidia-driver-local-repo-.../keyring.gpg /usr/share/keyrings/",
           "B. nvidia-ctk install-keyring",
           "C. dpkg --reconfigure keyrings",
           "D. cmsh -c 'softwareimage install-keyring'"],
     "a": "A", "e": "Driver dpkg installs prompt the operator to copy the keyring — manual `cp` step from the dpkg notice."},
    {"d": "I&D", "q": "After a fresh K8s install via BCM, which command lists every pod across every namespace?",
     "o": ["A. kubectl get pods --all-namespaces",
           "B. kubectl get pods -A",
           "C. Both A and B are valid",
           "D. cm-info --pods"],
     "a": "C", "e": "Both `--all-namespaces` and the `-A` short form work."},
    {"d": "I&D", "q": "Which command shows the BCM cluster manager daemon (cmd) version?",
     "o": ["A. cm-version",
           "B. cmsh -c 'main version'",
           "C. cmd --version",
           "D. cm-info --short"],
     "a": "A", "e": "cm-version prints the Bright/BCM cmd daemon version."},

    # ---------- Administration ----------
    {"d": "ADM", "q": "Which Slurm config file controls cgroup-based resource isolation for jobs?",
     "o": ["A. /etc/slurm/cgroup.conf",
           "B. /etc/slurm/slurm.conf",
           "C. /etc/slurm/gres.conf",
           "D. /etc/slurm/topology.conf"],
     "a": "A", "e": "cgroup.conf configures Slurm's cgroup plugin for CPU/memory/devices isolation."},
    {"d": "ADM", "q": "What's the difference between a MIG GPU instance and a MIG compute instance?",
     "o": ["A. A GPU instance is a slice of memory + SMs; a compute instance is a context inside a GPU instance",
           "B. A GPU instance is the host driver; a compute instance is the kernel",
           "C. They are the same thing with different names",
           "D. A compute instance is a CUDA stream"],
     "a": "A", "e": "GPU instances divide the GPU; compute instances are CUDA execution contexts inside a GPU instance."},
    {"d": "ADM", "q": "An admin needs to release a Slurm job currently in JobHeldAdmin state. Which command?",
     "o": ["A. scontrol release <jobid>",
           "B. scancel --hold <jobid>",
           "C. sacctmgr release <jobid>",
           "D. squeue --release <jobid>"],
     "a": "A", "e": "`scontrol release` clears the held state so the scheduler can pick up the job."},
    {"d": "ADM", "q": "Which sacctmgr command sets a TRES limit of 16 GPUs on a Slurm account?",
     "o": ["A. sacctmgr modify account <a> set GrpTRES=gres/gpu=16",
           "B. sacctmgr add account <a> with limit 16",
           "C. scontrol update Account=<a> GPU=16",
           "D. sshare set <a> 16"],
     "a": "A", "e": "Generic Trackable Resources (TRES) limits like gres/gpu are set with `modify ... set GrpTRES=…`."},
    {"d": "ADM", "q": "Which Run:ai object lets a researcher launch an interactive notebook with a long-running shell?",
     "o": ["A. Workspace",
           "B. Training Job",
           "C. Inference",
           "D. Department"],
     "a": "A", "e": "Workspaces are Run:ai's interactive long-running sessions (e.g., Jupyter)."},
    {"d": "ADM", "q": "Which Kubernetes objects together let an admin set per-namespace GPU caps AND per-pod GPU defaults?",
     "o": ["A. ResourceQuota and LimitRange",
           "B. NetworkPolicy and PodSecurityPolicy",
           "C. PersistentVolume and PersistentVolumeClaim",
           "D. ConfigMap and Secret"],
     "a": "A", "e": "ResourceQuota = namespace cap; LimitRange = per-pod default and max."},
    {"d": "ADM", "q": "Which command renames a BCM software image from 'gpu-image' to 'h100-image'?",
     "o": ["A. softwareimage rename gpu-image h100-image; commit",
           "B. softwareimage clone gpu-image h100-image; commit  (then `remove gpu-image; commit`)",
           "C. mv /cm/images/gpu-image /cm/images/h100-image",
           "D. cmsh -c 'rename softwareimage gpu-image h100-image'"],
     "a": "B", "e": "BCM doesn't have a direct rename — clone to the new name, commit, then remove the original."},
    {"d": "ADM", "q": "Which command lists Run:ai NodePools registered with the cluster?",
     "o": ["A. runai list nodepools",
           "B. kubectl get nodepools.runai.ai",
           "C. Both A and B work",
           "D. runai cluster nodes"],
     "a": "C", "e": "Run:ai exposes nodepools both via its CLI and as Kubernetes CRDs."},
    {"d": "ADM", "q": "An admin must temporarily prevent any new Slurm jobs from being submitted cluster-wide while an upgrade is performed. Which command?",
     "o": ["A. scontrol update SubmitEnabled=no",
           "B. systemctl stop slurmctld",
           "C. scancel --all",
           "D. scontrol shutdown"],
     "a": "A", "e": "SubmitEnabled=no gates new submissions while existing jobs continue to run."},

    # ---------- Workload Management ----------
    {"d": "WLM", "q": "Which sbatch flag dispatches a job array of 100 tasks?",
     "o": ["A. --array=0-99",
           "B. --tasks=100",
           "C. --replicas=100",
           "D. --jobs=100"],
     "a": "A", "e": "`--array=0-99` schedules 100 array tasks indexed 0..99."},
    {"d": "WLM", "q": "Which Run:ai workload type is best suited for a stateless Triton inference server scaled by request rate?",
     "o": ["A. Inference (deployment-style)",
           "B. Training",
           "C. Workspace",
           "D. Distributed-MPI"],
     "a": "A", "e": "Run:ai's Inference workload type wraps a Kubernetes Deployment with HPA semantics for inference."},
    {"d": "WLM", "q": "Which NGC CLI sequence lists images in the catalog and inspects one image's tags?",
     "o": ["A. ngc registry image list ; ngc registry image info <image>",
           "B. ngc images ; ngc inspect <image>",
           "C. docker registry list ; docker inspect <image>",
           "D. nvcr.io/list ; nvcr.io/info"],
     "a": "A", "e": "`ngc registry image list` browses; `ngc registry image info <image>` shows tags + last update."},
    {"d": "WLM", "q": "Which Kubernetes resource expands or shrinks a Deployment replica count based on a custom metric like GPU utilization?",
     "o": ["A. Horizontal Pod Autoscaler (HPA) with custom metrics",
           "B. ReplicaSet",
           "C. CronJob",
           "D. PriorityClass"],
     "a": "A", "e": "HPA can scale replicas off custom metrics (e.g., DCGM Exporter metrics)."},
    {"d": "WLM", "q": "Which Kubernetes object is the right choice for a cluster-wide DCGM Exporter that must run exactly one pod per node?",
     "o": ["A. DaemonSet",
           "B. Deployment",
           "C. StatefulSet",
           "D. Job"],
     "a": "A", "e": "DaemonSet ensures one pod per node — perfect for node-level exporters."},
    {"d": "WLM", "q": "Which Kubernetes object exposes an inference deployment to clients OUTSIDE the cluster with HTTP routing rules?",
     "o": ["A. Ingress",
           "B. Service of type ClusterIP",
           "C. ConfigMap",
           "D. NetworkPolicy"],
     "a": "A", "e": "Ingress provides HTTP/HTTPS routing and TLS termination for external clients."},
    {"d": "WLM", "q": "Which Slurm setting turns on preemption based on job priority?",
     "o": ["A. PreemptType=preempt/job_prio in slurm.conf",
           "B. ScancelOnSubmit=yes",
           "C. EnforceLimits=yes",
           "D. AccountingStorageEnforce=qos,limits"],
     "a": "A", "e": "PreemptType=preempt/job_prio is the slurm.conf setting that enables priority-based preemption."},
    {"d": "WLM", "q": "An operator submits a Run:ai inference workload with `runai submit-inference --shm-size 32G`. What does --shm-size set?",
     "o": ["A. The /dev/shm tmpfs size for the container (used by NCCL/NVIDIA drivers)",
           "B. The Run:ai project's GPU quota",
           "C. The Kubernetes Service IP range",
           "D. The persistent volume claim size"],
     "a": "A", "e": "--shm-size matches `docker run --shm-size` — the container's /dev/shm size."},
    {"d": "WLM", "q": "Which sbatch flag asks for the entire node (no other jobs sharing it)?",
     "o": ["A. --exclusive",
           "B. --pty",
           "C. --solo",
           "D. --reserve"],
     "a": "A", "e": "`--exclusive` reserves the entire node so no other jobs co-locate."},

    # ---------- Troubleshooting & Optimization ----------
    {"d": "T&O", "q": "Which file contains detailed startup and runtime logs for nvidia-fabricmanager?",
     "o": ["A. /var/log/fabricmanager.log",
           "B. /var/log/cuda.log",
           "C. /var/log/nccl.log",
           "D. /var/log/messages"],
     "a": "A", "e": "fabricmanager writes its own log at /var/log/fabricmanager.log — first place to look for NVSwitch issues."},
    {"d": "T&O", "q": "Which dcgmi diag run level is fastest (basic software/integration tests only)?",
     "o": ["A. -r 1",
           "B. -r 3",
           "C. -r 4",
           "D. -r 5"],
     "a": "A", "e": "Level 1 = quick (~30s) software/init tests. 3 is medium-extensive, 4 is full memtest."},
    {"d": "T&O", "q": "Which command tails kernel-log lines in real time, filtered to NVRM/Xid messages?",
     "o": ["A. dmesg -wH | grep -i nvrm",
           "B. tail -f /var/log/cuda.log",
           "C. journalctl -u nvidia-driver",
           "D. nvidia-smi -q --xid"],
     "a": "A", "e": "`dmesg -w` follows the kernel ring buffer; pipe through grep to isolate NVRM/Xid lines."},
    {"d": "T&O", "q": "An H100 thermal-throttles at 92°C. Which one is the most likely root cause?",
     "o": ["A. Insufficient cooling (airflow, blocked fans, hot inlet)",
           "B. NCCL version mismatch",
           "C. MIG misconfiguration",
           "D. Slurm preemption"],
     "a": "A", "e": "92°C exceeds typical throttle thresholds (~85-90°C). Cooling problems are the leading cause."},
    {"d": "T&O", "q": "Which command tails a Kubernetes pod's logs continuously?",
     "o": ["A. kubectl logs -f <pod>",
           "B. kubectl describe -w <pod>",
           "C. kubectl events --watch <pod>",
           "D. kubectl get pod -o yaml -w"],
     "a": "A", "e": "`-f` follows the log stream just like `tail -f`."},
    {"d": "T&O", "q": "An admin runs `nvidia-smi -q -d ECC,TEMPERATURE`. What information does this command return?",
     "o": ["A. Volatile and aggregate ECC counters plus thermal state for each attached GPU",
           "B. Just the persistence-mode setting",
           "C. NVLink topology",
           "D. GPU clock speed only"],
     "a": "A", "e": "-q is verbose query; -d filters to specific domains (ECC + TEMPERATURE here)."},
    {"d": "T&O", "q": "Which event in BMC SEL most directly indicates a PSU sag?",
     "o": ["A. Power Unit | PSU 1: 12V rail below threshold",
           "B. Memory | DIMM B1: ECC error corrected",
           "C. Processor | CPU1 thermal trip",
           "D. Watchdog | NMI"],
     "a": "A", "e": "The Power Unit / PSU rail-below-threshold event in `ipmitool sel list` indicates voltage sag."},
    {"d": "T&O", "q": "Which command inspects per-link FEC error counters on a Mellanox NIC port?",
     "o": ["A. mlxlink",
           "B. nvidia-smi nvlink --errors",
           "C. ethtool -S",
           "D. ibstat"],
     "a": "A", "e": "mlxlink reports physical-layer state and BER/FEC stats for the NIC port."},
    {"d": "T&O", "q": "After Triton pods crashloop on a new image, what's the fastest way to revert?",
     "o": ["A. kubectl rollout undo deployment/triton",
           "B. helm uninstall triton",
           "C. kubectl delete pod -l app=triton",
           "D. kubectl restart deployment/triton"],
     "a": "A", "e": "rollout undo flips back to the previous revision."},
    {"d": "T&O", "q": "Which command brings up a serial console (SOL) on a remote BMC for boot-time troubleshooting?",
     "o": ["A. ipmitool -H <bmc> -U <u> -P <p> sol activate",
           "B. ssh -p 623 <bmc>",
           "C. ipmitool console attach",
           "D. virsh console <bmc>"],
     "a": "A", "e": "`ipmitool sol activate` opens the serial-over-LAN session — the canonical way to watch a node boot remotely."},
]


LAB_TASKS_2 = [
    {
        "title": "Investigate why a Slurm node-007 is in DRAIN state with NHC: ECC failure",
        "story": "node-007 went into drain* with reason 'NHC: GPU ECC threshold'. "
                 "Confirm the failure source via slurmd logs, query ECC counters "
                 "on the GPU, run a targeted DCGM diag, and bring the node back "
                 "to service if the ECC counters look benign. Type commands "
                 "one per line; type 'done' when finished.",
        "expected": [
            ("journalctl", "slurmd"),
            ("nvidia-smi", "ECC"),       # `-q -d ECC` etc.
            ("dcgmi", "diag"),
            ("scontrol", "RESUME"),
        ],
        "max_score": 4,
    },
    {
        "title": "Authenticate to NGC and pull the Triton inference container",
        "story": "Set up NGC CLI auth, then create the corresponding "
                 "imagePullSecret in Kubernetes so a pod in namespace 'inference' "
                 "can pull from nvcr.io. Type the commands one per line; "
                 "type 'done' to submit.",
        "expected": [
            ("ngc", "auth", "login"),
            ("docker", "login", "nvcr.io"),
            ("kubectl", "create", "secret", "docker-registry"),
            ("nvcr.io",),
        ],
        "max_score": 4,
    },
    {
        "title": "Slurm cluster maintenance: gate submissions, upgrade slurmctld, restore",
        "story": "Prepare the cluster for a slurmctld upgrade: stop accepting new "
                 "submissions, drain a single node for hardware work, restart "
                 "slurmctld, then restore service. Type the commands one per "
                 "line; type 'done' when finished.",
        "expected": [
            ("scontrol", "SubmitEnabled=no"),
            ("scontrol", "DRAIN"),
            ("systemctl", "slurmctld"),
            ("scontrol", "SubmitEnabled=yes"),
            ("scontrol", "RESUME"),
        ],
        "max_score": 5,
    },
    {
        "title": "Verify GPUDirect RDMA performance between two nodes",
        "story": "An ML team reports slow distributed training. Quickly verify "
                 "the RDMA fabric is healthy by checking link status, firmware, "
                 "FEC, and running a perftest bandwidth test. Type the commands "
                 "one per line; type 'done'.",
        "expected": [
            ("mst", "status"),
            ("mlxfwmanager",),
            ("mlxlink",),
            ("ib_write_bw",),
        ],
        "max_score": 4,
    },
    {
        "title": "Build a BCM custom monitoring metric, healthcheck, and trigger",
        "story": "Add a 'ramp' single-line metric script that runs every second, "
                 "a paired healthcheck script that runs every 20s, an action that "
                 "fires a 'rampaction' script, and a trigger that fires when "
                 "ramp > 95. Type the commands one per line; type 'done'.",
        "expected": [
            ("monitoring", "add-producer", "ramp"),
            ("monitoring", "add-producer", "healthcheck"),
            ("monitoring", "add-action"),
            ("monitoring", "add-trigger"),
        ],
        "max_score": 4,
    },
    {
        "title": "Roll out and verify a fresh DCGM Exporter via Helm",
        "story": "Add the NVIDIA helm repo, upgrade the GPU operator chart that "
                 "ships DCGM Exporter, and verify pods come up healthy in the "
                 "gpu-operator namespace. Type the commands one per line; type "
                 "'done'.",
        "expected": [
            ("helm", "repo", "add"),
            ("helm", "upgrade"),
            ("kubectl", "get", "pods", "gpu-operator"),
            ("kubectl", "rollout", "status"),
        ],
        "max_score": 4,
    },
]


EXAM_QUESTIONS_3 = [
    # ---------- Installation & Deployment ----------
    {"d": "I&D", "q": "Which BCM device-mode command pings a compute node from the head node?",
     "o": ["A. cmsh -c 'device ping node001'",
           "B. ping node001",
           "C. cmsh -c 'device alive node001'",
           "D. cmsh -c 'main ping node001'"],
     "a": "A", "e": "cmsh's device mode has a built-in ping verb that uses the cluster's internal network."},
    {"d": "I&D", "q": "Which command-line tool from the NVIDIA Network Operator family configures and manages MOFED installation across worker nodes?",
     "o": ["A. NVIDIA Network Operator (Helm-installed Kubernetes operator)",
           "B. nv-mofed-install",
           "C. mlxconfig --bulk",
           "D. mst start --all"],
     "a": "A", "e": "NVIDIA's Network Operator is a Kubernetes operator that installs MOFED, GPUDirect RDMA, and SR-IOV across nodes."},
    {"d": "I&D", "q": "What does the NVIDIA GPU Operator install that the device plugin alone does NOT?",
     "o": ["A. Driver, container toolkit, DCGM, MIG-manager and node-feature-discovery, all as DaemonSets",
           "B. Just the kubelet device plugin",
           "C. CUDA samples",
           "D. Kubernetes itself"],
     "a": "A", "e": "GPU Operator bundles the driver/toolkit/DCGM/MIG/NFD via DaemonSets, beyond what the bare device plugin gives you."},
    {"d": "I&D", "q": "Which is true about Magnum IO?",
     "o": ["A. It's a software stack — including NCCL, GDS, GPUDirect RDMA — for moving data efficiently to/from GPUs",
           "B. It's a hardware switch product",
           "C. It's a benchmark suite",
           "D. It's an inference framework"],
     "a": "A", "e": "Magnum IO is NVIDIA's data-movement stack — NCCL, GPUDirect Storage, GPUDirect RDMA, etc."},
    {"d": "I&D", "q": "Which BCM file pre-stages the lab MAC addresses for batch import via readmac.sh?",
     "o": ["A. nodes.csv",
           "B. /etc/cm/nodes.conf",
           "C. /var/cm/macs.txt",
           "D. /cm/shared/macs.json"],
     "a": "A", "e": "readmac.sh reads node,mac pairs from nodes.csv (lines starting with # are comments)."},
    {"d": "I&D", "q": "Which BCM cmsh sub-mode lets you list, clone, and update software images?",
     "o": ["A. softwareimage",
           "B. category",
           "C. device",
           "D. image"],
     "a": "A", "e": "Software images live under softwareimage mode in cmsh."},
    {"d": "I&D", "q": "When BCM provisions a new node, what is the typical state-machine sequence reported in cmsh?",
     "o": ["A. BOOTING → INSTALLING → INSTALLER_CALLINGINIT → UP",
           "B. CONFIGURING → INSTALLED → READY",
           "C. PXE → DHCP → DONE",
           "D. STARTING → ACTIVE"],
     "a": "A", "e": "BCM's node installer reports BOOTING (PXE), INSTALLING, INSTALLER_CALLINGINIT, then UP."},
    {"d": "I&D", "q": "Which cmsh command shows just the configured software-image-to-category mapping?",
     "o": ["A. cmsh -c 'category list'",
           "B. cmsh -c 'softwareimage map'",
           "C. cmsh -c 'device list'",
           "D. cmsh -c 'main category-image'"],
     "a": "A", "e": "`category list` prints each category with its bound software image and node count."},
    {"d": "I&D", "q": "Which BCM utility verifies cluster HA replication state between active and passive head nodes?",
     "o": ["A. cmha status",
           "B. cmsh -c 'main ha-state'",
           "C. drbdadm status",
           "D. mhcheck --ha"],
     "a": "A", "e": "cmha is BCM's HA orchestrator; `cmha status` shows active/standby and DRBD state."},
    {"d": "I&D", "q": "After BCM finishes installing Kubernetes, which command joins a non-root user (e.g., k8suser) as a kubectl-capable user?",
     "o": ["A. cm-kubernetes-setup --add-user k8suser",
           "B. kubeadm useradd k8suser",
           "C. kubectl create user k8suser",
           "D. cmsh -c 'user add-k8s k8suser'"],
     "a": "A", "e": "BCM's wizard subcommand provisions a non-root k8s user with their own kubeconfig."},
    {"d": "I&D", "q": "Which BCM utility lists the node-installer state for every node in the cluster?",
     "o": ["A. node-installer-status",
           "B. cmsh -c 'main installer'",
           "C. systemctl status node-installer",
           "D. cmsh -c 'softwareimage installer'"],
     "a": "A", "e": "node-installer-status prints per-node installation state and last-installed image."},
    {"d": "I&D", "q": "Which tool drops you into a chroot of a software image so you can install packages or create files that will be baked into the image?",
     "o": ["A. cm-chroot-sw-img /cm/images/<image>",
           "B. chroot /cm/images/<image>",
           "C. cmsh -c 'softwareimage chroot <image>'",
           "D. systemd-nspawn -D <image>"],
     "a": "A", "e": "cm-chroot-sw-img is BCM's wrapper that mounts the image with the right bindings before chroot-ing."},

    # ---------- Administration ----------
    {"d": "ADM", "q": "Which Slurm command shows the priority-decomposition factors (age, fairshare, jobsize, qos, partition) for pending jobs?",
     "o": ["A. sprio",
           "B. squeue --priority",
           "C. sshare",
           "D. sstat"],
     "a": "A", "e": "sprio breaks each pending job's priority into its individual factors."},
    {"d": "ADM", "q": "Which Slurm command shows raw vs effective fairshare usage per account/user?",
     "o": ["A. sshare",
           "B. sprio",
           "C. sacct",
           "D. sreport"],
     "a": "A", "e": "sshare shows RawShares, NormShares, RawUsage, EffectvUsage."},
    {"d": "ADM", "q": "Which command lists Slurm partitions and their per-partition limits?",
     "o": ["A. scontrol show partition",
           "B. sinfo --partition",
           "C. sacctmgr list partition",
           "D. squeue -p"],
     "a": "A", "e": "`scontrol show partition` prints each partition's full limit set (MaxTime, AllowGroups, etc.)."},
    {"d": "ADM", "q": "Inside cmsh device mode, which command operates on every node in a category at once?",
     "o": ["A. foreach -c <category> (<command>)",
           "B. apply --category <c> <command>",
           "C. cmsh -c 'foreach nodes <command>'",
           "D. device run -c <c> <command>"],
     "a": "A", "e": "`foreach -c <cat> (...)` runs the bracketed cmsh command across every device in the category."},
    {"d": "ADM", "q": "Which cmsh device-mode form acts on a numeric range of nodes (e.g., node001..node004)?",
     "o": ["A. range -n node001..node004",
           "B. for-each-node node001..node004",
           "C. cmsh -c 'devices in node001..node004'",
           "D. run-on node001-004"],
     "a": "A", "e": "`range -n <a>..<b>` enters a temporary multi-node context where set/get/clear apply to the whole range."},
    {"d": "ADM", "q": "An admin needs a node to inherit its software image from its category instead of an explicit override. Which command clears the override?",
     "o": ["A. clear softwareimage",
           "B. set softwareimage default",
           "C. reset softwareimage",
           "D. unset softwareimage"],
     "a": "A", "e": "`clear softwareimage` removes the per-node override so the category's image takes effect again."},
    {"d": "ADM", "q": "Which kubectl flag selects nodes by a label key/value when a command targets multiple nodes?",
     "o": ["A. -l (label selector)",
           "B. -n (namespace)",
           "C. -A (all namespaces)",
           "D. -o (output format)"],
     "a": "A", "e": "`-l app=triton` selects all matching pods/nodes — the standard kubectl label-selector flag."},
    {"d": "ADM", "q": "Which Run:ai admin command increases project ml-research's GPU quota from 32 to 64?",
     "o": ["A. runai update project ml-research --gpu-quota 64",
           "B. runai modify project ml-research quota=64",
           "C. kubectl patch project ml-research --gpu=64",
           "D. cmsh -c 'wlm quota ml-research 64'"],
     "a": "A", "e": "`runai update project … --gpu-quota …` is the canonical admin command."},

    # ---------- Workload Management ----------
    {"d": "WLM", "q": "Which sbatch flag delays a job until another job successfully completes?",
     "o": ["A. --dependency=afterok:<jobid>",
           "B. --after=<jobid>",
           "C. --wait-for=<jobid>",
           "D. --queue-after=<jobid>"],
     "a": "A", "e": "`--dependency=afterok:<jobid>` only releases the new job once the dependency succeeded."},
    {"d": "WLM", "q": "Which environment variable does NCCL use to choose which network interface to bind on a multi-NIC node?",
     "o": ["A. NCCL_SOCKET_IFNAME",
           "B. NCCL_NET",
           "C. NCCL_INTERFACE",
           "D. NCCL_BIND"],
     "a": "A", "e": "NCCL_SOCKET_IFNAME selects the IP-network interface NCCL uses for rendezvous and TCP fall-back."},
    {"d": "WLM", "q": "What is the canonical Kubernetes Service type for an inference deployment that needs an external IP?",
     "o": ["A. LoadBalancer (or NodePort if no cloud LB available)",
           "B. ClusterIP",
           "C. ExternalName",
           "D. Headless"],
     "a": "A", "e": "LoadBalancer provisions an external IP via the cloud provider; NodePort is the on-prem equivalent."},
    {"d": "WLM", "q": "Which kubectl resource type allows pods to remember persistent storage across restarts?",
     "o": ["A. PersistentVolumeClaim (PVC)",
           "B. ConfigMap",
           "C. Secret",
           "D. EmptyDir"],
     "a": "A", "e": "PVCs are how pods request and bind to PersistentVolumes."},
    {"d": "WLM", "q": "Which Run:ai CLI command shows GPU/memory utilization across currently-running jobs?",
     "o": ["A. runai top job",
           "B. runai metrics",
           "C. runai stats",
           "D. kubectl top runai-jobs"],
     "a": "A", "e": "`runai top job` displays live util/mem per running job, mirroring `kubectl top`."},
    {"d": "WLM", "q": "Which is the correct way to authenticate the docker daemon for pulls from nvcr.io?",
     "o": ["A. docker login nvcr.io -u '$oauthtoken' -p <NGC_API_KEY>",
           "B. docker login nvcr.io with org user/pass",
           "C. ngc docker auth",
           "D. nvidia-docker login <KEY>"],
     "a": "A", "e": "Username is literally `$oauthtoken`, password is the NGC API key."},
    {"d": "WLM", "q": "Which command annotates a Kubernetes namespace with a key/value the admission controllers can read?",
     "o": ["A. kubectl annotate namespace default <key>=<value>",
           "B. kubectl set annotation namespace default key value",
           "C. kubectl label namespace default key=value",
           "D. kubectl edit ns default"],
     "a": "A", "e": "`kubectl annotate <resource> <name> key=val` adds or updates an annotation."},

    # ---------- Troubleshooting & Optimization ----------
    {"d": "T&O", "q": "Which `nvidia-smi` flag resets the GPU after the last process detaches (post XID 79)?",
     "o": ["A. --gpu-reset",
           "B. --recover",
           "C. --reload",
           "D. --device-init"],
     "a": "A", "e": "`nvidia-smi --gpu-reset` does a full GPU reset; requires no active processes."},
    {"d": "T&O", "q": "Which is the correct first step for an inference pod stuck in Pending after a node drain?",
     "o": ["A. kubectl describe pod <p>  (look at FailedScheduling events)",
           "B. kubectl delete pod <p> --force",
           "C. helm uninstall the inference release",
           "D. Reboot every node"],
     "a": "A", "e": "Always read `kubectl describe pod` events first — it tells you exactly why the scheduler rejected the pod."},
    {"d": "T&O", "q": "Which DCGM tool runs an interactive memory + compute health check on a specific GPU?",
     "o": ["A. dcgmi diag -r 3 -i <gpu_index>",
           "B. dcgmi memcheck",
           "C. dcgmi compute-test",
           "D. nvidia-bug-report.sh"],
     "a": "A", "e": "Level 3 with `-i` targets one GPU and exercises compute, memory, PCIe, NVLink, and power."},
    {"d": "T&O", "q": "What's the difference between volatile and aggregate ECC counters?",
     "o": ["A. Volatile resets at driver reload; aggregate persists for the GPU's lifetime",
           "B. Volatile is for memory; aggregate is for SM",
           "C. Volatile is double-bit; aggregate is single-bit",
           "D. They are aliases for the same value"],
     "a": "A", "e": "Volatile counters reset on driver reload/reboot; aggregate counters survive."},
    {"d": "T&O", "q": "Which command shows live event-log entries for a specific systemd unit?",
     "o": ["A. journalctl -u <unit> -f",
           "B. tail -f /var/log/<unit>.log",
           "C. systemctl events <unit>",
           "D. dmesg -u <unit>"],
     "a": "A", "e": "`journalctl -u <unit> -f` is the standard live-tail of a systemd service's logs."},
    {"d": "T&O", "q": "Which monitoring source feeds Kubernetes HPA when scaling on GPU utilization?",
     "o": ["A. DCGM Exporter (Prometheus-formatted GPU metrics)",
           "B. nvidia-smi exporter built into kubelet",
           "C. GPU operator's HPA controller",
           "D. Slurm sacct"],
     "a": "A", "e": "DCGM Exporter is the standard way to surface GPU metrics into Prometheus and HPA."},
    {"d": "T&O", "q": "Which command-line tool takes a comprehensive snapshot of driver state, dmesg, NVLink topology and XID history for NVIDIA support?",
     "o": ["A. nvidia-bug-report.sh",
           "B. dcgmi snapshot",
           "C. nvidia-smi --dump",
           "D. systemctl status nvidia-driver"],
     "a": "A", "e": "nvidia-bug-report.sh is the canonical support-bundle generator from NVIDIA."},
    {"d": "T&O", "q": "Which BCM monitoring object actually fires a remediation script when a trigger condition is met?",
     "o": ["A. MonitoringScriptAction",
           "B. MonitoringTrigger",
           "C. MonitoringDashboard",
           "D. MonitoringDataProducer"],
     "a": "A", "e": "Triggers detect; ScriptActions are what they invoke (enter/leave/during)."},
    {"d": "T&O", "q": "An admin sees `Failed to initialize NVML` from `nvidia-smi`. Most likely cause?",
     "o": ["A. Driver isn't loaded or doesn't match the kernel",
           "B. CUDA samples aren't installed",
           "C. NCCL version is wrong",
           "D. Container runtime mismatch"],
     "a": "A", "e": "NVML init failure usually means the kernel module isn't loaded (driver/kernel version mismatch, MOFED conflict, etc.)."},
    {"d": "T&O", "q": "Which command shows a node's labels including `nvidia.com/gpu.product` (so you can verify the GPU operator detected the right model)?",
     "o": ["A. kubectl describe node <node>",
           "B. kubectl get pod -l nvidia",
           "C. nvidia-smi --node",
           "D. dcgmi cluster"],
     "a": "A", "e": "Node labels (nvidia.com/gpu.product, count, mig.strategy) appear in `describe node` output."},
]


LAB_TASKS_3 = [
    {
        "title": "Promote a Run:ai project to over-quota and verify a borrowing job runs",
        "story": "Increase project 'ml-research' GPU guarantee to 16 and quota "
                 "to 64, submit a 24-GPU training job that should consume "
                 "guaranteed + over-quota GPUs, and verify it's running. Type "
                 "the commands one per line; type 'done'.",
        "expected": [
            ("runai", "update", "project", "--gpu-quota"),
            ("runai", "submit"),
            ("-g", "24"),
            ("runai", "list", "jobs"),
        ],
        "max_score": 4,
    },
    {
        "title": "Trace a 'pull access denied' error on a Triton pod",
        "story": "A Triton pod is in ImagePullBackOff with 'pull access denied "
                 "for nvcr.io/...'. Re-authenticate, recreate the pull secret, "
                 "and force the pod to retry. Type the commands one per line.",
        "expected": [
            ("kubectl", "describe", "pod"),
            ("docker", "login", "nvcr.io"),
            ("kubectl", "create", "secret", "docker-registry"),
            ("kubectl", "delete", "pod"),
        ],
        "max_score": 4,
    },
    {
        "title": "Build a 4-node Slurm batch job and inspect runtime efficiency",
        "story": "Submit a 4-node × 8-GPU sbatch job, monitor it in the queue, "
                 "and inspect its CPU/memory efficiency after it completes. "
                 "Type the commands one per line.",
        "expected": [
            ("sbatch",),
            ("--gres=gpu:8",),
            ("squeue",),
            ("seff",),
        ],
        "max_score": 4,
    },
    {
        "title": "Pin a Kubernetes inference deployment to MIG GPUs",
        "story": "After enabling MIG on the H100 nodes, label them and pin "
                 "the inference Deployment to MIG-enabled nodes. Type the "
                 "commands one per line; type 'done'.",
        "expected": [
            ("nvidia-smi", "-mig", "1"),
            ("kubectl", "label", "nodes"),
            ("kubectl", "set"),     # set image / set resources etc.
            ("kubectl", "rollout", "status"),
        ],
        "max_score": 4,
    },
    {
        "title": "Quarantine a failing node from BCM, drain it, and re-image",
        "story": "node-007 is reporting persistent ECC errors. Move it to "
                 "MAINTENANCE state in BCM, drain Slurm and Kubernetes, then "
                 "re-image it with default-image-backup. Type the commands "
                 "one per line; type 'done'.",
        "expected": [
            ("scontrol", "DRAIN"),
            ("kubectl", "cordon"),
            ("imageupdate", "node-007"),
            ("scontrol", "RESUME"),
        ],
        "max_score": 4,
    },
    {
        "title": "Verify NVLink + NVSwitch fabric health on an H100 HGX node",
        "story": "Before launching a 8-GPU NCCL test, confirm fabricmanager is "
                 "healthy, every NVLink is active with no errors, and the "
                 "topology shows the expected NV18 connections. Type the "
                 "commands one per line; type 'done'.",
        "expected": [
            ("systemctl", "nvidia-fabricmanager"),
            ("nvidia-smi", "topo"),
            ("nvidia-smi", "nvlink", "--status"),
            ("nvidia-smi", "nvlink", "--errors"),
        ],
        "max_score": 4,
    },
]


# Domain weights — must match the official 31/23/23/23 blueprint.
DOMAIN_TARGETS = {"I&D": 9, "ADM": 7, "WLM": 7, "T&O": 7}     # = 30 total
DOMAIN_NAME = {"I&D": "Installation & Deployment",
               "ADM": "Administration",
               "WLM": "Workload Management",
               "T&O": "Troubleshooting & Optimization"}


EXAM_QUESTIONS_4 = [
    # ---------- Installation & Deployment ----------
    {"d": "I&D", "q": "Which BCM cmsh form rolls a software-image change to a node?",
     "o": ["A. imageupdate -n <node> -w from device mode",
           "B. apt-get update --node <node> --image now",
           "C. systemctl reload bcm-image @<node>.service",
           "D. cmsh -c 'softwareimage push <node> --immediate'"],
     "a": "A", "e": "imageupdate -n <node> -w pushes the assigned image to the node and writes."},
    {"d": "I&D", "q": "Which BCM file pre-stages /etc/slurm/slurm.conf when nodes provision?",
     "o": ["A. The software image bound to the node's category",
           "B. /etc/cm/slurm.conf.d/100-defaults.conf manually",
           "C. The kubelet bootstrap config in /var/lib/kubelet",
           "D. /etc/cm/wlm/template.conf only at first install"],
     "a": "A", "e": "The category's image carries /etc/slurm/slurm.conf — provisioning syncs the image to the node."},
    {"d": "I&D", "q": "Which command boots a fresh head-node K8s control plane via BCM?",
     "o": ["A. cm-kubernetes-setup wizard from the head node CLI",
           "B. kubectl init --bcm /etc/cm/kube.conf in user shell",
           "C. systemctl start bcm-kubelet on every worker node",
           "D. apt install kubernetes-control-plane && start it up"],
     "a": "A", "e": "BCM wraps `kubeadm init` in cm-kubernetes-setup, which configures CNI/GPU operator/dashboard."},
    {"d": "I&D", "q": "Which DOCA service offloads OVS bridge processing onto a BlueField DPU?",
     "o": ["A. DOCA OVS Hardware Offload service on the DPU Arm",
           "B. DOCA HBN — Host-Based Networking on the host CPU",
           "C. DOCA Telemetry running inside the kernel modules",
           "D. DOCA Comm Channel only — for management traffic"],
     "a": "A", "e": "DOCA's OVS hardware offload service runs on the DPU Arm to accelerate Open vSwitch packet processing."},
    {"d": "I&D", "q": "Which command verifies BCM categories against bound software images?",
     "o": ["A. cmsh -c 'category list' from the head node terminal",
           "B. cmsh -c 'softwareimage map' (this command does not exist)",
           "C. cm-info --categories prints summary only without images",
           "D. cmsh -c 'device categories' lists devices not categories"],
     "a": "A", "e": "`category list` shows category name, software image, and node count side by side."},
    {"d": "I&D", "q": "Which Linux command checks the GPU PCIe address before driver install?",
     "o": ["A. lspci -nn | grep -i nvidia (or `lspci -Q | grep NVIDIA`)",
           "B. dmesg --pci nvidia (deprecated wrapper, removed 2024)",
           "C. nvidia-pci-list (only available with the driver loaded)",
           "D. cat /proc/driver/nvidia/pci.txt before driver install"],
     "a": "A", "e": "lspci is the standard pre-install enumeration; nvidia-smi requires the driver."},
    {"d": "I&D", "q": "Which procedure correctly removes prior NVIDIA driver packages?",
     "o": ["A. sudo apt-get remove --purge '^nvidia-.*' then reboot",
           "B. dpkg -r nvidia-driver only — leaves config files behind",
           "C. modprobe -r nvidia (just unloads, does not uninstall)",
           "D. systemctl disable nvidia (no apt removal performed)"],
     "a": "A", "e": "Pattern-purge with apt-get is the canonical clean-uninstall before installing a fresh driver."},
    {"d": "I&D", "q": "Which is BCM's HA replication state that you check with cmha?",
     "o": ["A. UpToDate / UpToDate between active and passive heads",
           "B. Synchronised / Pending — partial commit not yet flushed",
           "C. Active / Standby — DRBD has not yet finished init phase",
           "D. Online / Online — both heads serve simultaneously now"],
     "a": "A", "e": "DRBD reports UpToDate / UpToDate when replication is healthy."},
    {"d": "I&D", "q": "Which install order is correct for an end-to-end AI cluster?",
     "o": ["A. BCM head → provision nodes → K8s/Slurm → GPU operator/Run:ai",
           "B. Run:ai first → Kubernetes second → BCM third on top of stack",
           "C. Driver first on every host, then Kubernetes, then BCM later",
           "D. SLURM controller first, then BCM, then K8s, then DPU stack"],
     "a": "A", "e": "BCM provisions the OS and runtimes; orchestrators stack on top."},
    {"d": "I&D", "q": "Which command checks BCM CMD daemon version on the head node?",
     "o": ["A. cm-version prints the cluster manager daemon (cmd) version",
           "B. cmsh -c 'main version' returns the version of the cmsh CLI",
           "C. systemctl status cmd | head — no version reported there",
           "D. dpkg -l cmd-daemon | grep ii (only if installed via apt)"],
     "a": "A", "e": "cm-version is the dedicated utility; the others either don't exist or won't show the daemon's version cleanly."},
    {"d": "I&D", "q": "Which file contains the lab's lab-script-driven MAC mapping?",
     "o": ["A. nodes.csv with `node,mac` lines used by readmac.sh helper",
           "B. /etc/cm/nodes.json bound at provisioning time by daemon",
           "C. /var/cm/lab-macs.txt (only generated after PXE discovery)",
           "D. /cm/shared/lab/macs.csv (no helper script reads this file)"],
     "a": "A", "e": "BCM's lab pattern reads node,mac from nodes.csv via readmac.sh."},
    {"d": "I&D", "q": "Which Magnum IO component accelerates GPU↔storage data flows?",
     "o": ["A. GPUDirect Storage (GDS) — direct path from NVMe to GPU mem",
           "B. NVLink — provides peer-to-peer GPU links inside one server",
           "C. NCCL — multi-GPU collective communication library only",
           "D. DOCA Flow — packet processing pipeline on DPUs not storage"],
     "a": "A", "e": "GDS bypasses the CPU bounce buffer for storage-to-GPU transfers."},

    # ---------- Administration ----------
    {"d": "ADM", "q": "Which command snapshots the etcd database used by Kubernetes?",
     "o": ["A. etcdctl snapshot save /backup/etcd.db --endpoints=...",
           "B. kubectl backup etcd /backup/etcd.db --since=now-1h",
           "C. systemctl stop etcd && cp /var/lib/etcd /backup/etcd",
           "D. cmsh -c 'main snapshot etcd' from the BCM CLI shell"],
     "a": "A", "e": "etcdctl snapshot save is the standard etcd backup command (with cacert/cert/key)."},
    {"d": "ADM", "q": "Which Slurm command shows partition limits including AllowGroups?",
     "o": ["A. scontrol show partition prints all per-partition limits",
           "B. sinfo --partition (only header + node states, not limits)",
           "C. squeue --partition shows queued jobs not partition info",
           "D. sacctmgr list partition (lists accounts not partitions)"],
     "a": "A", "e": "scontrol show partition prints AllowGroups, MaxTime, etc."},
    {"d": "ADM", "q": "Which command applies a Slurm reservation to a set of nodes?",
     "o": ["A. scontrol create reservation NodeCnt=4 StartTime=now Duration=2h",
           "B. sbatch --reserve nodes=4 --time=2h --start=now (no such flag)",
           "C. sacctmgr add reservation node-001..004 --duration 2h --now",
           "D. sinfo --reserve --duration 2h --nodes 4 (sinfo cannot edit)"],
     "a": "A", "e": "scontrol create reservation is the canonical create-reservation form."},
    {"d": "ADM", "q": "Which cmsh command restores a previous BCM revision/snapshot?",
     "o": ["A. cmsh -c 'main revision restore <id>' from CLI on head node",
           "B. systemctl restart bcm-revision-restore on the head node now",
           "C. cm-info --restore <id> (no such option exists in cm-info)",
           "D. cmsh -c 'main rollback <id>' (use revision restore, not roll)"],
     "a": "A", "e": "Revisions are managed under main mode; the verb is `revision restore`."},
    {"d": "ADM", "q": "Which Run:ai admin object groups projects under one quota tree?",
     "o": ["A. Department contains projects; quotas roll up to department",
           "B. NodePool contains nodes only; not projects (resource group)",
           "C. Workspace is per-user job; not for grouping admin policies",
           "D. Cluster groups installations; not the object you'd use here"],
     "a": "A", "e": "Run:ai's Department aggregates projects for hierarchical quotas."},
    {"d": "ADM", "q": "Which kubectl command grants cluster-admin role to a service account?",
     "o": ["A. kubectl create clusterrolebinding admin-sa --clusterrole=cluster-admin --serviceaccount=ns:sa",
           "B. kubectl create rolebinding admin-sa --role=cluster-admin --serviceaccount=ns:sa --namespace=ns",
           "C. kubectl annotate clusterrole cluster-admin grants=admin-sa to bind it to a service account",
           "D. kubectl edit serviceaccount admin-sa --add-cluster-admin (no such flag in kubectl edit)"],
     "a": "A", "e": "ClusterRoleBinding (not RoleBinding) is required for cluster-wide privileges."},
    {"d": "ADM", "q": "Which sinfo flag formats output with custom columns?",
     "o": ["A. sinfo -N --Format=NodeHost,CPUsState,Gres,GresUsed,Reason",
           "B. sinfo --columns=node,cpu,gres (no such flag in sinfo)",
           "C. sinfo --custom (no such flag — use --Format with literal)",
           "D. sinfo --output=node,gres,reason (use --Format= not --output)"],
     "a": "A", "e": "sinfo's --Format= takes a comma-separated column spec with capital F."},
    {"d": "ADM", "q": "Which scontrol command updates the cluster's submit-enabled state?",
     "o": ["A. scontrol update SubmitEnabled=no/yes (cluster-wide submit gating)",
           "B. systemctl pause slurmctld (stops scheduler, not submission gate)",
           "C. sacctmgr modify cluster set submit=no (no such field/syntax)",
           "D. sbatch --paused on every job — does not block new submissions"],
     "a": "A", "e": "scontrol update SubmitEnabled=... gates new job submissions cluster-wide."},

    # ---------- Workload Management ----------
    {"d": "WLM", "q": "Which Run:ai workload type wraps a Kubernetes Deployment + HPA?",
     "o": ["A. Inference workload — exposed as Deployment, autoscaled by HPA",
           "B. Workspace — interactive, not stateless or autoscaled (Jupyter)",
           "C. Distributed-MPI workload — multi-node MPI training, batch only",
           "D. Training workload — single-shot batch job, not autoscaled at all"],
     "a": "A", "e": "Run:ai's inference workload is the long-running stateless service form with HPA."},
    {"d": "WLM", "q": "Which sbatch flag provides a job a literal interactive shell on its node?",
     "o": ["A. srun --pty bash (pseudo-tty + bash on the allocated node)",
           "B. sbatch --interactive (no such flag in canonical Slurm CLI)",
           "C. salloc --tty (allocates only; need srun to actually start sh)",
           "D. scontrol attach (no such verb — there is no attach in scontrol)"],
     "a": "A", "e": "srun --pty bash is the canonical Slurm interactive shell command."},
    {"d": "WLM", "q": "Which kubectl flag shows logs since a specific time window?",
     "o": ["A. kubectl logs <pod> --since=24h (or --since-time=<RFC3339>)",
           "B. kubectl logs <pod> --time 24h (no such flag in kubectl logs)",
           "C. kubectl logs <pod> --window 1d (no such flag in kubectl logs)",
           "D. kubectl logs <pod> --last 24 (no such flag in kubectl logs)"],
     "a": "A", "e": "--since= takes a duration like 24h; --since-time= takes RFC3339."},
    {"d": "WLM", "q": "Which command lists Kubernetes PVCs across every namespace?",
     "o": ["A. kubectl get pvc --all-namespaces (or `kubectl get pvc -A`)",
           "B. kubectl describe pvc -A (describe doesn't accept -A by itself)",
           "C. kubectl get pv --all (lists volumes not claims; not the same)",
           "D. kubectl pvc list (kubectl uses get verb, no such pvc verb)"],
     "a": "A", "e": "--all-namespaces (or -A) shows PVCs cluster-wide."},
    {"d": "WLM", "q": "Which sbatch flag sets a job's runtime limit?",
     "o": ["A. -t 02:00:00 (or --time=02:00:00) — wall-clock limit format",
           "B. --runtime 2h (no such flag) — Slurm uses --time= directive",
           "C. --max-time 7200s (no such flag) — Slurm uses --time= dir",
           "D. --duration 2:00 (no such flag) — Slurm uses --time= directly"],
     "a": "A", "e": "Slurm's runtime limit flag is -t / --time= in HH:MM:SS or D-HH:MM."},
    {"d": "WLM", "q": "Which Slurm script directive requests 4 GPUs on each task?",
     "o": ["A. #SBATCH --gres=gpu:4 (and --ntasks for total task count)",
           "B. #SBATCH --gpus-per-task 4 (different semantic in newer Slurm)",
           "C. #SBATCH --resources=gpu:4 (no such flag in canonical Slurm)",
           "D. #SBATCH --request gpus 4 (no such flag in canonical Slurm)"],
     "a": "A", "e": "The classic GRES form is --gres=gpu:N; new --gpus-per-task=N exists too but the question asked for the canonical syntax."},
    {"d": "WLM", "q": "Which command exposes a deployment outside the cluster as a Service?",
     "o": ["A. kubectl expose deployment triton --type=LoadBalancer --port=8000",
           "B. kubectl create service load-balancer triton --port 8000 (rare)",
           "C. kubectl set service triton --external (no such kubectl verb)",
           "D. kubectl deploy triton --expose (no such kubectl verb either)"],
     "a": "A", "e": "kubectl expose is the canonical create-service form for deployments."},

    # ---------- Troubleshooting & Optimization ----------
    {"d": "T&O", "q": "Which command collects driver, dmesg, NVLink, and XID for support?",
     "o": ["A. nvidia-bug-report.sh (writes a tarball with full state dump)",
           "B. nvidia-smi --dump (no such flag in current driver versions)",
           "C. dcgmi snapshot (no such verb in canonical dcgmi versions)",
           "D. dmesg > nvidia.log (only kernel ring buffer, not full state)"],
     "a": "A", "e": "nvidia-bug-report.sh is the canonical NVIDIA support-bundle tool."},
    {"d": "T&O", "q": "Which counter type resets when the NVIDIA driver reloads?",
     "o": ["A. Volatile ECC counters reset on driver reload — aggregate stays",
           "B. Aggregate ECC counters reset on driver reload — volatile stays",
           "C. Both reset on driver reload — counters always start fresh now",
           "D. Neither reset — counters always persist for the GPU's lifetime"],
     "a": "A", "e": "Volatile ECC counters are tied to driver lifetime; aggregate persists."},
    {"d": "T&O", "q": "Which RDMA tool measures one-way write bandwidth to a remote NIC?",
     "o": ["A. ib_write_bw (verb tests one-way RDMA Write at line rate now)",
           "B. iperf3 (does TCP/UDP; not RDMA — not for verifying GDR-RDMA)",
           "C. fio (does block-level I/O; not network RDMA verbs at all)",
           "D. netperf (TCP/UDP throughput tool; not RDMA verbs at all)"],
     "a": "A", "e": "ib_write_bw from the perftest suite measures one-way RDMA Write bandwidth."},
    {"d": "T&O", "q": "Which fabric-manager log file is the FIRST place to look for issues?",
     "o": ["A. /var/log/fabricmanager.log — service-specific log file path",
           "B. /var/log/cuda.log — CUDA runtime log, not fabric-manager log",
           "C. /var/log/nccl.log — NCCL collective log, not fabric-manager log",
           "D. /var/log/messages — generic syslog; fabricmanager has its own"],
     "a": "A", "e": "fabricmanager logs to its own file first; messages may have a small subset."},
    {"d": "T&O", "q": "Which Run:ai log shows job preemption decisions?",
     "o": ["A. journalctl -u runai-scheduler — scheduler logs include preempt",
           "B. journalctl -u runai-engine (this unit name does not exist)",
           "C. /var/log/runai/jobs.log (not the canonical path; varies)",
           "D. kubectl logs deployment/runai-engine-overseer (wrong app)"],
     "a": "A", "e": "Run:ai's scheduler is the component that preempts; its logs are under runai-scheduler."},
    {"d": "T&O", "q": "Which command shows the kubelet's current pod-eviction events?",
     "o": ["A. journalctl -u kubelet -f | grep -i evict (live tail with grep)",
           "B. kubectl get events -n kube-system (events but not evict only)",
           "C. systemctl status kubelet (only the unit status, not eviction)",
           "D. cat /var/log/kubelet.log (path varies; journald is canonical)"],
     "a": "A", "e": "Kubelet logs to journald; -f tails it live, grep filters."},
    {"d": "T&O", "q": "Which command triggers a forced pod removal bypassing graceful term?",
     "o": ["A. kubectl delete pod <pod> --force --grace-period=0 (immediate)",
           "B. kubectl delete pod <pod> --immediate=true (no such flag exists)",
           "C. kubectl rollout restart pod <pod> (rolling, doesn't force kill)",
           "D. kubectl evict pod <pod> --now (no such kubectl subcommand)"],
     "a": "A", "e": "--force --grace-period=0 deletes the pod object immediately."},
    {"d": "T&O", "q": "Which Mellanox tool inspects ConnectX/BlueField firmware?",
     "o": ["A. mlxfwmanager — queries Mellanox/NVIDIA NIC + DPU firmware",
           "B. mlxlink — physical-layer state (BER/FEC), not firmware version",
           "C. mst status — lists devices for tools, not firmware queries",
           "D. ibstat — InfiniBand port state, not firmware version status"],
     "a": "A", "e": "mlxfwmanager prints firmware versions for Mellanox/NVIDIA NICs and DPUs."},
    {"d": "T&O", "q": "Which command triggers a DCGM extended hardware diagnostic?",
     "o": ["A. dcgmi diag -r 3 (medium-extensive: SM, mem, PCIe, NVLink test)",
           "B. dcgmi check --hw (no such verb in standard dcgmi commands)",
           "C. dcgmi extended-test --hw (no such verb in canonical dcgmi)",
           "D. dcgmi profile --start (profiles, not a hardware diagnostic)"],
     "a": "A", "e": "Level 3 runs the extended hardware suite (compute, mem, PCIe, NVLink, power)."},
]


LAB_TASKS_4 = [
    {
        "title": "Restore a Slurm node from drain after an ECC NHC failure",
        "story": "node-007 went into drain* with 'NHC: GPU ECC threshold' a "
                 "few hours ago; the GPU has since cleared. Confirm the "
                 "current state, verify ECC, then resume scheduling.",
        "expected": [
            ("scontrol", "show", "node"),
            ("nvidia-smi", "ECC"),
            ("dcgmi", "diag"),
            ("scontrol", "RESUME"),
        ],
        "max_score": 4,
    },
    {
        "title": "Build a 4-node GPU job and verify efficiency",
        "story": "Submit a 4-node × 8-GPU sbatch job named 'big-train' with an "
                 "8-hour limit, wait for it to start, and inspect its CPU/mem "
                 "efficiency with seff.",
        "expected": [
            ("sbatch", "--gres=gpu:8"),
            ("--time",),
            ("squeue",),
            ("seff",),
        ],
        "max_score": 4,
    },
    {
        "title": "Restore Kubernetes etcd from a snapshot",
        "story": "The etcd database is corrupted on a single-control-plane "
                 "cluster. Snapshot, stop the API server, restore from the "
                 "saved snapshot, and bring the control plane back up.",
        "expected": [
            ("etcdctl", "snapshot", "save"),
            ("systemctl", "stop", "kubelet"),
            ("etcdctl", "snapshot", "restore"),
            ("systemctl", "start", "kubelet"),
        ],
        "max_score": 4,
    },
    {
        "title": "Set up an NGC imagePullSecret and patch a Deployment to use it",
        "story": "A Triton Deployment in 'inference' is failing image pulls. "
                 "Create the docker-registry secret with NGC creds, patch the "
                 "deployment to reference it, and watch the rollout.",
        "expected": [
            ("kubectl", "create", "secret", "docker-registry"),
            ("nvcr.io",),
            ("kubectl", "patch", "deployment"),
            ("kubectl", "rollout", "status"),
        ],
        "max_score": 4,
    },
    {
        "title": "Tail kubelet + slurmd live to spot a flapping node",
        "story": "node-005 keeps oscillating between Ready and NotReady in "
                 "Kubernetes. Tail kubelet logs and slurmd logs side-by-side, "
                 "then drain it from both schedulers when you confirm.",
        "expected": [
            ("journalctl", "kubelet"),
            ("journalctl", "slurmd"),
            ("kubectl", "drain"),
            ("scontrol", "DRAIN"),
        ],
        "max_score": 4,
    },
    {
        "title": "Roll out a new GPU operator chart and verify DCGM Exporter",
        "story": "Bump the gpu-operator Helm release to a newer chart and "
                 "confirm the DCGM Exporter DaemonSet pods are healthy.",
        "expected": [
            ("helm", "repo", "update"),
            ("helm", "upgrade", "gpu-operator"),
            ("kubectl", "get", "pods", "gpu-operator"),
            ("kubectl", "rollout", "status"),
        ],
        "max_score": 4,
    },
]


EXAM_QUESTIONS_5 = [
    # ---------- Installation & Deployment ----------
    {"d": "I&D", "q": "Which BCM tool is a wizard for adding a non-root k8s user?",
     "o": ["A. cm-kubernetes-setup --add-user <name> on the head node CLI",
           "B. kubeadm useradd <name> (no such kubeadm verb in canonical k8s)",
           "C. kubectl create user <name> --kubeconfig (no such kubectl verb)",
           "D. cmsh -c 'user add-k8s <name>' (no such cmsh subcommand exists)"],
     "a": "A", "e": "BCM's wizard creates the user, namespace, and per-user kubeconfig."},
    {"d": "I&D", "q": "Which DOCA component encrypts/processes IPsec on the DPU Arm?",
     "o": ["A. DOCA Strongswan / IPsec offload service on the DPU Arm CPU",
           "B. DOCA Flow — packet-processing pipeline; not crypto by itself",
           "C. DOCA Comm Channel — host↔DPU control channel only, no crypto",
           "D. DOCA Telemetry — hardware metrics collection; not crypto today"],
     "a": "A", "e": "DOCA's IPsec service runs Strongswan on the DPU and offloads crypto to the BlueField hardware."},
    {"d": "I&D", "q": "Which BCM file gates the cluster's external network defaults?",
     "o": ["A. cmsh main mode 'externalnet' setting via `set externalnet`",
           "B. /etc/cm/external.conf — directly edited file on head node FS",
           "C. cmsh -c 'softwareimage external' (no such mode in cmsh today)",
           "D. /etc/network/interfaces is sole authority for net config now"],
     "a": "A", "e": "Network defaults are properties of the cmsh main partition."},
    {"d": "I&D", "q": "Which BCM revision-management verb saves a named snapshot?",
     "o": ["A. cmsh -c 'main revision create <name>' from head node CLI",
           "B. cmsh -c 'main snapshot save <name>' (incorrect verb naming)",
           "C. cmsh -c 'main backup <name>' (incorrect verb naming today)",
           "D. cm-info --snapshot <name> (no such option in cm-info today)"],
     "a": "A", "e": "Revisions live under main mode; the verb is `revision create <name>`."},
    {"d": "I&D", "q": "Which YAML key in /etc/docker/daemon.json switches default runtime?",
     "o": ["A. \"default-runtime\": \"nvidia\" — switches docker to nvidia runtime",
           "B. \"runtime\": \"nvidia\" — incorrect key name; default-runtime",
           "C. \"runtime-default\": \"nvidia\" — incorrect key name in JSON",
           "D. \"nvidia-runtime\": true — incorrect key/value combination too"],
     "a": "A", "e": "Docker reads `default-runtime` to choose the runtime when --runtime= isn't passed."},
    {"d": "I&D", "q": "Which Linux package suite supplies the perftest RDMA tools?",
     "o": ["A. perftest (apt install perftest on Ubuntu — installs ib_*_bw)",
           "B. mellanox-tools (legacy bundle on RHEL; superseded by perftest)",
           "C. ofed-tools (no such canonical package — see ofed/MLNX install)",
           "D. ib-tools (no such canonical package — see ofed/MLNX install)"],
     "a": "A", "e": "perftest is the apt/yum package; binaries are ib_write_bw / ib_read_bw / ib_send_bw."},
    {"d": "I&D", "q": "Which step is required before a container can use GPUs?",
     "o": ["A. Install nvidia-container-toolkit + run nvidia-ctk runtime config",
           "B. Set --gpus all on docker run is sufficient — no toolkit needed",
           "C. Install CUDA on host — host driver is enough for any container",
           "D. Manually mount /dev/nvidia* into the container (works fine too)"],
     "a": "A", "e": "Without the Container Toolkit, Docker has no nvidia runtime to inject the GPU into the container."},
    {"d": "I&D", "q": "Which command lists Kubernetes pods in 'inference' namespace?",
     "o": ["A. kubectl get pods -n inference (or `kubectl -n inference get pods`)",
           "B. kubectl pods --namespace inference (kubectl uses 'get' verb)",
           "C. kubectl get -n inference (must specify which resource get)",
           "D. kubectl ns inference get pods (incorrect ordering of flags)"],
     "a": "A", "e": "-n is the namespace flag; the verb is get, the resource is pods."},
    {"d": "I&D", "q": "Which provisioning state means BCM has not yet bound a MAC?",
     "o": ["A. PhysicalNode with MAC 00:00:00:00:00:00 — slot exists, no MAC",
           "B. HeadNode unassigned — applies to head, not compute, nodes ever",
           "C. UnknownNode discovered — discovered but no slot or category yet",
           "D. PXE-pending — informal label; real BCM does not use this state"],
     "a": "A", "e": "Empty MAC means the slot is registered but PXE binding hasn't happened yet."},
    {"d": "I&D", "q": "Which ConfigMap names the GPU operator's policy resource?",
     "o": ["A. clusterpolicies.nvidia.com — single CRD instance per cluster",
           "B. gpu-operator-config — incorrect; that's a sub-ConfigMap inside",
           "C. nvidia-gpu-config — incorrect; not the canonical CRD plural",
           "D. nvidia-config — incorrect; not the canonical CRD plural too"],
     "a": "A", "e": "GPU operator config is a ClusterPolicy resource (clusterpolicies.nvidia.com)."},
    {"d": "I&D", "q": "Which step provisions a new compute node in BCM via PXE?",
     "o": ["A. Worker boots PXE, contacts head, registers MAC + state UP",
           "B. Worker boots from local disk image — no PXE involved at all",
           "C. Worker downloads image directly from S3 then registers itself",
           "D. Head node SSHes to worker and pushes the image then reboots"],
     "a": "A", "e": "BCM's standard provisioning relies on PXE/DHCP/TFTP from the head."},
    {"d": "I&D", "q": "Which file maps Kubernetes RBAC roles to namespaces?",
     "o": ["A. RoleBinding YAML (and ClusterRoleBinding for cluster scope)",
           "B. PodSecurityPolicy YAML (deprecated; not RBAC role mapping)",
           "C. ServiceAccount YAML alone (does not bind roles by itself)",
           "D. NetworkPolicy YAML (network rules; not role assignment)"],
     "a": "A", "e": "RoleBinding maps a Role to subjects within a namespace; ClusterRoleBinding does it cluster-wide."},

    # ---------- Administration ----------
    {"d": "ADM", "q": "Which command modifies a Slurm user's QoS list additively?",
     "o": ["A. sacctmgr modify user <u> set qos+=<name>",
           "B. sacctmgr add qos <name> to user <u> directly through CLI",
           "C. scontrol update User=<u> QoS=<name> (scontrol can't set QoS)",
           "D. sacctmgr promote user <u> with qos <name> (no such verb)"],
     "a": "A", "e": "`set qos+=<name>` appends; `qos=<name>` would replace."},
    {"d": "ADM", "q": "Which Run:ai operation activates a paused project's quota?",
     "o": ["A. runai update project <p> --gpu-quota <n> (or set --status=open)",
           "B. runai resume project <p> (no such project-level resume verb)",
           "C. runai patch project <p> with state open (no such patch verb)",
           "D. runai start project <p> with quota number (no such start verb)"],
     "a": "A", "e": "Project state is changed via `runai update project` (quota or open/closed flags)."},
    {"d": "ADM", "q": "Which sacctmgr command lists Slurm-defined accounts (departments)?",
     "o": ["A. sacctmgr list account (or `sacctmgr show account`)",
           "B. sinfo --accounts (no such flag in sinfo cluster summary)",
           "C. squeue --account-list (squeue doesn't list defined accts)",
           "D. scontrol show account (scontrol doesn't show accounts)"],
     "a": "A", "e": "Slurm accounts (the accounting hierarchy) are listed via sacctmgr."},
    {"d": "ADM", "q": "Which kubectl flag scales a Deployment to N replicas?",
     "o": ["A. kubectl scale deployment <name> --replicas=N (instant scale)",
           "B. kubectl set replicas <name> N (no such kubectl set verb)",
           "C. kubectl edit deployment <name> --replicas=N (no such flag)",
           "D. kubectl deployment scale <name> N (incorrect kubectl form)"],
     "a": "A", "e": "kubectl scale ... --replicas=N is the canonical form."},
    {"d": "ADM", "q": "Which BCM monitoring trigger condition fires when ramp > 95?",
     "o": ["A. Trigger expression: measurable=ramp, operator=>, value=95",
           "B. Trigger expression: ramp.value greater 95 (incorrect syntax)",
           "C. Action expression: ramp >= 95 (incorrect — operator is >)",
           "D. Healthcheck: ramp PASS when >95 (semantics are inverted)"],
     "a": "A", "e": "Triggers attach an Expression with measurable, operator, and threshold value."},
    {"d": "ADM", "q": "Which Linux user/group operation creates a Slurm-able account?",
     "o": ["A. sacctmgr add account <a>; useradd; sacctmgr create user <u>",
           "B. useradd alone is sufficient (Slurm accounting is independent)",
           "C. sacctmgr add user <u> alone (an account must exist first now)",
           "D. groupadd alone is sufficient (Slurm accounting needs more)"],
     "a": "A", "e": "Slurm accounting requires an account record AND a user record bound to it."},
    {"d": "ADM", "q": "Which cmsh device verb forces a node to repower-on remotely?",
     "o": ["A. cmsh -c 'device power on <node>' (uses BMC IPMI under the hood)",
           "B. cmsh -c 'device start <node>' (no such verb in cmsh today)",
           "C. cmsh -c 'device boot <node>' (no such verb in cmsh today)",
           "D. systemctl start <node> from head node (no such systemd unit)"],
     "a": "A", "e": "cmsh's device mode has power on/off/reset that wraps IPMI."},
    {"d": "ADM", "q": "Which command lists Slurm jobs blocked behind a dependency?",
     "o": ["A. squeue --states=PENDING --reason=Dependency (filtered queue)",
           "B. scontrol show dependencies (no such scontrol subcommand)",
           "C. sacct --depblocked (no such sacct flag in canonical Slurm)",
           "D. sinfo --dependency (no such flag — sinfo is for nodes)"],
     "a": "A", "e": "squeue can filter by state and reason; Dependency is a recognised PENDING reason string."},

    # ---------- Workload Management ----------
    {"d": "WLM", "q": "Which Run:ai distributed-launcher submits MPI multi-node training?",
     "o": ["A. runai submit-mpi <name> -p <proj> -g <n> --workers <w> -- ...",
           "B. runai submit --mpi --nodes <n> (no such combination of flags)",
           "C. runai mpi <name> -p <proj> -g <n> (no such mpi subcommand)",
           "D. runai submit --dist mpi --workers <w> (no such flag combo)"],
     "a": "A", "e": "Run:ai's distributed-MPI form is `runai submit-mpi`."},
    {"d": "WLM", "q": "Which kubectl command rolls a deployment forward by replacing image?",
     "o": ["A. kubectl set image deployment/<name> <container>=<new-image>",
           "B. kubectl deploy --image=<new-image> (no such kubectl verb)",
           "C. kubectl rollout image <name> --new=<image> (no such flag)",
           "D. kubectl restart deployment/<name> --image (no such flag)"],
     "a": "A", "e": "set image triggers a rolling update with the new image."},
    {"d": "WLM", "q": "Which Slurm flag causes a job to fail-fast on first task error?",
     "o": ["A. srun --kill-on-bad-exit (kills the step when one task exits)",
           "B. sbatch --abort-on-error (no such flag in canonical Slurm CLI)",
           "C. sbatch --strict (no such flag exists in canonical Slurm CLI)",
           "D. srun --fail (no such flag — see --kill-on-bad-exit instead)"],
     "a": "A", "e": "srun --kill-on-bad-exit terminates the entire step when any task exits non-zero."},
    {"d": "WLM", "q": "Which kubectl resource controls per-pod CPU/memory defaults?",
     "o": ["A. LimitRange (per-namespace defaults + min/max for pod resources)",
           "B. ResourceQuota (caps total resources, not per-pod defaults at all)",
           "C. PriorityClass (orders scheduling, not per-pod resource defaults)",
           "D. PodDisruptionBudget (eviction policy; not resource defaults)"],
     "a": "A", "e": "LimitRange sets per-pod defaults and min/max; ResourceQuota caps the namespace total."},
    {"d": "WLM", "q": "Which sbatch flag sets a job array with concurrency cap?",
     "o": ["A. --array=0-99%10 (100 array tasks but max 10 running at once)",
           "B. --array=0-99 --concurrent 10 (no such flag exists in Slurm)",
           "C. --array=0-99,parallel=10 (incorrect syntax for the cap value)",
           "D. --array=0-99 --max-running 10 (no such flag exists in Slurm)"],
     "a": "A", "e": "The %N suffix on --array caps simultaneous array tasks at N."},
    {"d": "WLM", "q": "Which Run:ai feature lets a user borrow over-quota GPUs?",
     "o": ["A. Over-quota (deserved-GPU) scheduling — borrows free cluster GPUs",
           "B. Burst priority class — no such Run:ai feature with that name",
           "C. Quota exemption flag — no such Run:ai project-level flag exists",
           "D. NodePool override list — no such Run:ai feature for borrowing"],
     "a": "A", "e": "Run:ai's deserved-GPU mechanism allows projects to use spare cluster GPUs above their quota."},
    {"d": "WLM", "q": "Which kubectl verb prints CPU/memory live usage for pods?",
     "o": ["A. kubectl top pod (also: kubectl top node for node-level usage)",
           "B. kubectl describe pod (shows events not live resource use)",
           "C. kubectl get pod -o wide (just lists with placement detail)",
           "D. kubectl stats pod (no such kubectl verb in canonical CLI)"],
     "a": "A", "e": "kubectl top is the canonical live-resource-usage view (requires metrics-server)."},

    # ---------- Troubleshooting & Optimization ----------
    {"d": "T&O", "q": "Which command shows kernel-log XID errors filtered live?",
     "o": ["A. dmesg -wH | grep -i 'NVRM: Xid' (live tail filtered to NVRM)",
           "B. journalctl -u nvidia-driver (no such systemd unit on hosts)",
           "C. tail -f /var/log/cuda.log (CUDA runtime, not kernel ring)",
           "D. cat /proc/driver/nvidia/xid (no such procfs entry by default)"],
     "a": "A", "e": "dmesg -w follows the kernel ring buffer; -H makes timestamps human-readable."},
    {"d": "T&O", "q": "Which command counts ECC errors aggregated since GPU lifetime start?",
     "o": ["A. nvidia-smi --query-gpu=ecc.errors.uncorrected.aggregate.total",
           "B. nvidia-smi --query-gpu=ecc.errors.lifetime (no such field today)",
           "C. nvidia-smi -q -d ECC | grep volatile (volatile resets on reload)",
           "D. dcgmi diag --ecc-aggregate (no such dcgmi flag in current docs)"],
     "a": "A", "e": "Aggregate ECC counters survive driver reload; volatile do not."},
    {"d": "T&O", "q": "Which command checks if Docker can use the nvidia runtime?",
     "o": ["A. docker info | grep -i runtimes (or run a sanity nvidia-smi pod)",
           "B. nvidia-smi --runtime=docker (no such nvidia-smi flag exists)",
           "C. cat /etc/docker/daemon.json (only shows config, not effective)",
           "D. systemctl status docker | grep nvidia (status doesn't show)"],
     "a": "A", "e": "docker info enumerates registered runtimes; presence of 'nvidia' confirms toolkit setup."},
    {"d": "T&O", "q": "Which Mellanox check reports per-link physical-layer state?",
     "o": ["A. mlxlink (state, speed, FEC, BER, signal levels — physical layer)",
           "B. ibstat (port state at link layer; less physical detail)",
           "C. mlxconfig (configures NIC params; not physical-layer query)",
           "D. mst status (lists devices for tools, not physical-layer info)"],
     "a": "A", "e": "mlxlink is the standard tool for physical-layer state on ConnectX/BlueField."},
    {"d": "T&O", "q": "Which command resets a GPU after XID 79 with no active processes?",
     "o": ["A. nvidia-smi --gpu-reset -i <gpu_index> (or just --gpu-reset)",
           "B. nvidia-smi --recover -i <gpu_index> (no such recover option)",
           "C. modprobe -r nvidia ; modprobe nvidia (full driver reset only)",
           "D. systemctl restart nvidia-persistenced (only the daemon)"],
     "a": "A", "e": "--gpu-reset works when no processes hold the device."},
    {"d": "T&O", "q": "Which BCM monitoring object FIRES when a trigger condition matches?",
     "o": ["A. MonitoringScriptAction — runs when entering/leaving the trigger",
           "B. MonitoringTrigger — defines the condition; doesn't run a script",
           "C. MonitoringDataProducer — emits metric/health values; not action",
           "D. MonitoringDashboard — displays only; never executes anything"],
     "a": "A", "e": "Triggers detect; ScriptActions are the things that actually run."},
    {"d": "T&O", "q": "Which `kubectl delete` flags force-evict a stuck pod?",
     "o": ["A. --force --grace-period=0 (immediate; bypasses graceful term)",
           "B. --immediate (no such flag in canonical kubectl delete CLI)",
           "C. --evict (no such flag — eviction is via the eviction API)",
           "D. --now (no such flag — kubectl uses --grace-period=0 instead)"],
     "a": "A", "e": "--force AND --grace-period=0 together immediately remove the pod object."},
    {"d": "T&O", "q": "Which is the FIRST place to look when NCCL says 'Connect failed'?",
     "o": ["A. Inter-node IP/firewall reachability and NCCL_SOCKET_IFNAME var",
           "B. The CUDA driver version number on every node in the cluster",
           "C. The Slurm scheduler logs for the affected partition queue now",
           "D. The Docker image build date for the training container layer"],
     "a": "A", "e": "NCCL connect failures almost always mean the rendezvous network is unreachable or wrong-interface."},
    {"d": "T&O", "q": "Which command tail-monitors GPU live state in 5-line bursts?",
     "o": ["A. nvidia-smi dmon (continuous device monitor; cols power/temp/sm)",
           "B. nvidia-smi --query-gpu=... --format=csv (snapshot only one row)",
           "C. dcgmi profile --start (profiling, not live continuous metrics)",
           "D. iostat -x 1 (block-IO live, not GPU metrics in any way today)"],
     "a": "A", "e": "nvidia-smi dmon continuously prints power/temp/SM/mem/clocks per GPU."},
]


LAB_TASKS_5 = [
    {
        "title": "Set up a multi-team Run:ai cluster with hierarchical quotas",
        "story": "Create a department 'research', two projects 'nlp' and 'cv' "
                 "under it with 32 GPUs each (16 guaranteed, 16 over-quota), "
                 "then submit a 24-GPU MPI job in 'nlp' to test borrowing.",
        "expected": [
            ("runai", "create", "department"),
            ("runai", "create", "project"),
            ("runai", "update", "project", "--gpu-quota"),
            ("runai", "submit-mpi"),
        ],
        "max_score": 4,
    },
    {
        "title": "Recover from a failed Helm upgrade of GPU operator",
        "story": "A helm upgrade of gpu-operator left some pods in CrashLoop. "
                 "Inspect history, roll back to the last good revision, and "
                 "verify the rollback succeeded.",
        "expected": [
            ("helm", "history", "gpu-operator"),
            ("helm", "rollback", "gpu-operator"),
            ("kubectl", "get", "pods", "gpu-operator"),
            ("kubectl", "rollout", "status"),
        ],
        "max_score": 4,
    },
    {
        "title": "Bake an NCCL test image into a fresh BCM software image",
        "story": "Clone default-image to nccl-test-image, chroot in, install "
                 "perftest + nccl-tests, exit chroot, and roll the image to "
                 "a category. Type the commands one per line.",
        "expected": [
            ("softwareimage", "clone"),
            ("cm-chroot-sw-img",),
            ("apt", "install"),
            ("commit",),
        ],
        "max_score": 4,
    },
    {
        "title": "Investigate a stuck pod with kubectl + journalctl",
        "story": "Pod 'inference-1' is stuck in CrashLoopBackOff. Inspect its "
                 "events and logs, check the kubelet on its node, and either "
                 "fix it or force-delete + redeploy.",
        "expected": [
            ("kubectl", "describe", "pod"),
            ("kubectl", "logs"),
            ("journalctl", "kubelet"),
            ("kubectl", "delete", "pod", "--force"),
        ],
        "max_score": 4,
    },
    {
        "title": "Audit Slurm priorities for a fairness complaint",
        "story": "A team complains their jobs are starving. Inspect the queue, "
                 "fairshare, and per-job priorities. Adjust the team's QoS so "
                 "they can use a higher-priority class.",
        "expected": [
            ("squeue",),
            ("sshare",),
            ("sprio",),
            ("sacctmgr", "modify", "user"),
        ],
        "max_score": 4,
    },
    {
        "title": "Recover a node BCM stopped seeing during an image rollout",
        "story": "node-005 hasn't checked back in since the last image rollout. "
                 "Verify status, attach SOL to watch the boot, force a power "
                 "reset, and confirm it returns.",
        "expected": [
            ("cmsh", "device", "status"),
            ("ipmitool", "sol"),
            ("cmsh", "device", "power"),
            ("cmsh", "device", "list"),
        ],
        "max_score": 4,
    },
]


EXAM_QUESTIONS_6 = [
    # ---------- Installation & Deployment ----------
    {"d": "I&D", "q": "Which command verifies a freshly-installed driver works?",
     "o": ["A. nvidia-smi (prints driver version, CUDA version, GPU table)",
           "B. cat /proc/driver/nvidia/version (only the driver version line)",
           "C. dmesg | grep nvidia (just kernel-load messages, not full data)",
           "D. lspci | grep -i nvidia (PCI presence; no driver-state info)"],
     "a": "A", "e": "nvidia-smi is the canonical post-install sanity check; it queries the loaded driver."},
    {"d": "I&D", "q": "Which BCM cmsh verb saves changes you've made interactively?",
     "o": ["A. commit (must be issued in the mode where changes were made)",
           "B. save (no such verb in cmsh — `commit` is the canonical word)",
           "C. apply (no such cmsh verb — Kubernetes uses `apply`, not BCM)",
           "D. write (no such verb — `commit` is the canonical word in cmsh)"],
     "a": "A", "e": "Commit is the cmsh verb that flushes pending changes to the database."},
    {"d": "I&D", "q": "Which BCM command lists current revisions/snapshots?",
     "o": ["A. cmsh -c 'main revision list' (under main mode in cmsh shell)",
           "B. cmsh -c 'softwareimage revisions' (incorrect mode for verb)",
           "C. cm-info --revisions (no such option in cm-info utility today)",
           "D. cmsh -c 'snapshots' (no such mode/verb in canonical BCM CLI)"],
     "a": "A", "e": "Revisions are managed under main mode."},
    {"d": "I&D", "q": "Which DOCA service inspects packets for telemetry?",
     "o": ["A. DOCA Telemetry Service collecting hardware counters from DPU",
           "B. DOCA Comm Channel — host↔DPU control bus only, no telemetry",
           "C. DOCA Flow — packet processing pipeline, not the telem service",
           "D. DOCA DMA — RDMA-style memory copy engine on DPU not telemetry"],
     "a": "A", "e": "DOCA Telemetry Service collects hardware/network counters from the DPU."},
    {"d": "I&D", "q": "Which BCM property enforces a node's role in the cluster?",
     "o": ["A. The node's category (binds image, partition, role, etc.)",
           "B. The node's MAC address (just a hardware identifier alone)",
           "C. The node's IP (network locator; no role info attached)",
           "D. The node's hostname (label only; not a configuration policy)"],
     "a": "A", "e": "Categories carry the policy: image, role, configuration."},
    {"d": "I&D", "q": "Which Linux command prints the running kernel build?",
     "o": ["A. uname -a (or `uname -r` for just the release string)",
           "B. cat /proc/version (also works; full build info available)",
           "C. dmesg | head -1 (only kernel ring start; not always present)",
           "D. lsb_release -a (distro info, not running-kernel build info)"],
     "a": "A", "e": "uname -a includes kernel name, host, release, version, arch."},
    {"d": "I&D", "q": "Which BCM monitoring metric type computes one number per script?",
     "o": ["A. MonitoringDataProducerSingleLineMetricScript runs script→1 num",
           "B. MonitoringDataProducerMultiLine — emits multiple lines/values",
           "C. MonitoringTrigger only fires actions; not a producer at all",
           "D. MonitoringScriptAction runs scripts on alert; not a producer"],
     "a": "A", "e": "SingleLineMetricScript is the canonical 'one script, one number' producer."},
    {"d": "I&D", "q": "Which Helm subcommand lists all releases across namespaces?",
     "o": ["A. helm list -A (or `helm list --all-namespaces`)",
           "B. helm list (only the current/default namespace by itself)",
           "C. helm releases --all (no such verb 'releases' in helm CLI)",
           "D. helm get -A (kubectl-style, not canonical helm CLI form)"],
     "a": "A", "e": "helm list -A shows every release in every namespace."},
    {"d": "I&D", "q": "Which step adds a Helm chart repository?",
     "o": ["A. helm repo add <name> <url> (and `helm repo update` to refresh)",
           "B. helm install repo <name> <url> (incorrect form for adding)",
           "C. helm repo new <name> <url> (incorrect verb — use `add`)",
           "D. helm chart add <name> <url> (incorrect verb — use `repo`)"],
     "a": "A", "e": "helm repo add is the canonical add-repo command."},
    {"d": "I&D", "q": "Which prerequisite is required before a Run:ai install?",
     "o": ["A. A working Kubernetes cluster + GPU operator/device plugin",
           "B. A Slurm cluster with at least one GPU partition active first",
           "C. NVIDIA Mission Control deployed on the head node first now",
           "D. A BlueField DPU on every node — Run:ai requires DPU offload"],
     "a": "A", "e": "Run:ai layers on top of Kubernetes; the GPU operator (or device plugin) is the GPU layer."},
    {"d": "I&D", "q": "Which command adds nvidia repo and pulls nvidia-driver?",
     "o": ["A. apt-get update && apt-get install nvidia-driver-<version>",
           "B. apt-get install nvidia-driver only (without an update first)",
           "C. yum install nvidia-driver — wrong package mgr on Ubuntu host",
           "D. snap install nvidia-driver — Ubuntu doesn't snap the driver"],
     "a": "A", "e": "Apt update + install with the explicit package name is the standard install."},
    {"d": "I&D", "q": "Which BCM utility audits cluster-wide health from CLI?",
     "o": ["A. mhcheck (master health check; pass/fail per service or item)",
           "B. cmsh -c 'main check' (no such verb in canonical cmsh CLI)",
           "C. cm-info --health (no such option in cm-info utility today)",
           "D. systemctl status bcm — only one unit not cluster-wide audit"],
     "a": "A", "e": "mhcheck is BCM's cluster-wide health audit."},

    # ---------- Administration ----------
    {"d": "ADM", "q": "Which command-line tool defines GPU resources for Slurm?",
     "o": ["A. /etc/slurm/gres.conf — per-node GPU device file declarations",
           "B. /etc/slurm/slurm.conf — partitions/jobs, but not device file",
           "C. /etc/slurm/cgroup.conf — only resource isolation cfg, not GPU",
           "D. /etc/slurm/topology.conf — switch topology, not GPU device"],
     "a": "A", "e": "gres.conf declares each node's GPUs and their device files."},
    {"d": "ADM", "q": "Which Run:ai concept tags a workload's compute share?",
     "o": ["A. Project — the resource container holding GPU/CPU quota and team",
           "B. Workspace — interactive job shell type, not the resource share",
           "C. Cluster — installation; multiple projects share one cluster now",
           "D. Department — groups projects; not the per-workload tag itself"],
     "a": "A", "e": "Workloads are submitted to a Project; the project carries the share."},
    {"d": "ADM", "q": "Which kubectl flag adds a node to a non-schedulable state?",
     "o": ["A. kubectl cordon <node> (drain optional; cordon marks unsched)",
           "B. kubectl drain <node> only (also drains; cordon is the gate)",
           "C. kubectl taint <node> NoSchedule:= (incorrect taint syntax)",
           "D. kubectl annotate <node> unschedulable=yes (annotation alone)"],
     "a": "A", "e": "cordon marks the node unschedulable; drain additionally evicts pods."},
    {"d": "ADM", "q": "Which Slurm subsystem stores accounting historical data?",
     "o": ["A. slurmdbd (the accounting daemon backed by MySQL/MariaDB)",
           "B. slurmctld (the controller; processes jobs, doesn't archive)",
           "C. slurmd (per-node daemon; doesn't keep history database)",
           "D. munged (auth daemon; not related to accounting at all)"],
     "a": "A", "e": "slurmdbd is the accounting database daemon."},
    {"d": "ADM", "q": "Which kubectl flag binds a Role to a user/SA in a namespace?",
     "o": ["A. kubectl create rolebinding <name> --role=<r> --user=<u> -n <ns>",
           "B. kubectl create clusterrolebinding ... --user=<u> (cluster scope)",
           "C. kubectl annotate user <u> role=<r> (no such kubectl form/path)",
           "D. kubectl bind <u> to <r> (no such kubectl form in canonical CLI)"],
     "a": "A", "e": "RoleBinding is the namespace-scoped binding; ClusterRoleBinding is cluster-wide."},
    {"d": "ADM", "q": "Which BCM cmsh verb inspects a node BMC's IPMI sensors?",
     "o": ["A. ipmitool (top-level Linux tool used from the head node CLI)",
           "B. cmsh -c 'device sensors <node>' (no such cmsh verb today)",
           "C. nvidia-smi --bmc (no such flag — nvidia-smi is for GPUs not BMC)",
           "D. cmsh -c 'main sensors' (no such cmsh verb — use ipmitool)"],
     "a": "A", "e": "BMC sensor inspection is via ipmitool; cmsh has no native sensors verb."},
    {"d": "ADM", "q": "Which cmsh device-mode form drains a node from Kubernetes?",
     "o": ["A. kubectl cordon <node> (and `kubectl drain` for graceful evict)",
           "B. cmsh -c 'device drain <node>' (no such verb in BCM cmsh CLI)",
           "C. cmsh -c 'device cordon <node>' (no such verb in BCM cmsh CLI)",
           "D. cmsh -c 'device k8s-drain <node>' (no such verb in cmsh CLI)"],
     "a": "A", "e": "Kubernetes drain/cordon are kubectl commands; BCM doesn't wrap them."},
    {"d": "ADM", "q": "Which kubectl annotation pins a Run:ai workload to a NodePool?",
     "o": ["A. nodeSelector / runai.nodepool annotation in the pod spec yaml",
           "B. spec.affinity.nodepool YAML key (incorrect canonical key path)",
           "C. metadata.labels.runai-nodepool (incorrect; labels not selector)",
           "D. spec.priority.nodepool YAML key (incorrect canonical key name)"],
     "a": "A", "e": "Run:ai uses node selectors/annotations to pin to a NodePool."},

    # ---------- Workload Management ----------
    {"d": "WLM", "q": "Which command pulls an NGC container without docker login?",
     "o": ["A. ngc registry image pull <image> (uses NGC CLI auth, not docker)",
           "B. docker pull nvcr.io/<image> (requires docker login first now)",
           "C. kubectl image pull <image> (kubectl has no image-pull verb)",
           "D. ngc model download <image> (model registry not image registry)"],
     "a": "A", "e": "The NGC CLI's image pull uses NGC creds, not docker login."},
    {"d": "WLM", "q": "Which sbatch flag dispatches a job ONLY after dependency OK?",
     "o": ["A. --dependency=afterok:<jobid> (release iff dep finished success)",
           "B. --after <jobid> (no such flag in canonical Slurm sbatch CLI)",
           "C. --wait-for <jobid> (no such flag in canonical Slurm CLI)",
           "D. --depends <jobid> (no such flag — use --dependency=afterok)"],
     "a": "A", "e": "Slurm's --dependency=afterok:<jobid> is the canonical AND-success form."},
    {"d": "WLM", "q": "Which command launches a Slurm interactive bash session?",
     "o": ["A. srun --pty bash (or `srun -N1 --pty --gres=gpu:1 bash` for GPU)",
           "B. salloc --interactive (no such flag, allocates only no shell)",
           "C. sbatch --shell (no such flag — sbatch is for batch scripts)",
           "D. scontrol login <jobid> (no such verb in canonical Slurm)"],
     "a": "A", "e": "srun --pty bash starts a pseudo-terminal session on the assigned node."},
    {"d": "WLM", "q": "Which Slurm flag waits for ALL tasks ready before launching?",
     "o": ["A. srun --wait-all-nodes=1 (gang-style: don't start until all up)",
           "B. srun --gang (no such flag in canonical Slurm — see --wait-all)",
           "C. sbatch --barrier (no such flag in canonical Slurm CLI today)",
           "D. srun --sync (no such flag — use --wait-all-nodes=1 instead)"],
     "a": "A", "e": "--wait-all-nodes=1 forces gang-style start when all nodes are ready."},
    {"d": "WLM", "q": "Which kubectl object enables an external HTTPS endpoint?",
     "o": ["A. Ingress (with a TLS section; or LoadBalancer Service for raw)",
           "B. Service of type ClusterIP (internal only, not external HTTPS)",
           "C. ConfigMap (configuration, not an external endpoint surface)",
           "D. NetworkPolicy (firewall rules, not an external endpoint)"],
     "a": "A", "e": "Ingress provides HTTP(S) routing and TLS; LoadBalancer is the L4 alternative."},
    {"d": "WLM", "q": "Which kubectl flag sets a pod's desired GPU model?",
     "o": ["A. nodeSelector with nvidia.com/gpu.product label in pod spec",
           "B. spec.affinity.gpu (no such canonical Pod-spec key in K8s API)",
           "C. spec.runtimeClass=nvidia-h100 (incorrect — runtimeClass not GPU)",
           "D. spec.priorityClass=h100-only (incorrect — priority not GPU)"],
     "a": "A", "e": "GPU operator labels nodes with nvidia.com/gpu.product so pods can target a model."},
    {"d": "WLM", "q": "Which mpirun flag controls processes per node for NCCL?",
     "o": ["A. --npernode <N> (sets processes per node for the MPI launch)",
           "B. -n <N> (total process count across all nodes; not per-node)",
           "C. --hosts <N> (host count; not processes per host setting)",
           "D. --threads <N> (threads per process; not processes per node)"],
     "a": "A", "e": "--npernode tells mpirun how many ranks to start on each host."},

    # ---------- Troubleshooting & Optimization ----------
    {"d": "T&O", "q": "Which command shows topology + NVLink connections in matrix?",
     "o": ["A. nvidia-smi topo -m (NV18 = NVLink, PIX = PCIe, X = self)",
           "B. nvidia-smi nvlink --status (per-link state, not full matrix)",
           "C. nvidia-smi --topology (no such canonical nvidia-smi flag)",
           "D. dcgmi topo (no such canonical dcgmi verb in current docs)"],
     "a": "A", "e": "topo -m prints the GPU↔GPU connection matrix."},
    {"d": "T&O", "q": "Which dcgmi diag level is fastest for a quick smoke test?",
     "o": ["A. dcgmi diag -r 1 (basic software/integration tests, ~30 sec)",
           "B. dcgmi diag -r 4 (extended memtest; takes 2+ hours per GPU)",
           "C. dcgmi diag -r 5 (no such level in canonical dcgmi releases)",
           "D. dcgmi diag --quick (no such flag — use -r 1 for the quickest)"],
     "a": "A", "e": "Level 1 is the fast smoke test; 3 is medium-extensive, 4 is exhaustive memtest."},
    {"d": "T&O", "q": "Which lsmod result indicates the NVIDIA driver is loaded?",
     "o": ["A. nvidia, nvidia_uvm, nvidia_drm modules listed in lsmod output",
           "B. cuda module listed (CUDA is userspace; not a kernel module)",
           "C. mlx5_core listed (Mellanox NIC driver; nothing about NVIDIA)",
           "D. drm only listed (DRM core; not specifically NVIDIA at all)"],
     "a": "A", "e": "nvidia + nvidia_uvm + nvidia_drm are the NVIDIA kernel modules."},
    {"d": "T&O", "q": "Which file gates which NIC NCCL uses for rendezvous?",
     "o": ["A. NCCL_SOCKET_IFNAME env var (e.g., NCCL_SOCKET_IFNAME=eth0)",
           "B. NCCL_BIND env var (no such NCCL env var in canonical NCCL)",
           "C. NCCL_NET env var (selects net plugin, not socket interface)",
           "D. NCCL_IF env var (no such NCCL env var in canonical NCCL)"],
     "a": "A", "e": "NCCL_SOCKET_IFNAME chooses the socket-rendezvous interface."},
    {"d": "T&O", "q": "Which command shows real-time CPU/memory load on each node?",
     "o": ["A. kubectl top node (live; needs metrics-server installed first)",
           "B. cat /proc/loadavg (one node only; CPU load average alone)",
           "C. uptime (one node only; CPU load average; no memory shown)",
           "D. dmesg | grep memory (kernel events; not live load values)"],
     "a": "A", "e": "kubectl top node prints live CPU/mem usage per node from metrics-server."},
    {"d": "T&O", "q": "Which command verifies fabric-manager is initialized cleanly?",
     "o": ["A. systemctl status nvidia-fabricmanager (or read its log file)",
           "B. lsmod | grep fabric (kernel modules don't show fabricmgr today)",
           "C. dcgmi cluster (no such verb in canonical dcgmi today CLI)",
           "D. nvidia-smi --fabric (no such flag in nvidia-smi canonical)"],
     "a": "A", "e": "Fabric Manager is a systemd unit; status + /var/log/fabricmanager.log."},
    {"d": "T&O", "q": "Which iostat field is most relevant for storage I/O wait?",
     "o": ["A. %iowait (CPU time spent waiting on I/O — direct indicator)",
           "B. tps (transactions per second; throughput not wait time)",
           "C. r/s (reads per second; throughput not wait time directly)",
           "D. svctm (service time; deprecated; %iowait is the right field)"],
     "a": "A", "e": "%iowait shows CPU time spent waiting on I/O — the canonical signal of storage bottlenecks."},
    {"d": "T&O", "q": "Which Run:ai log records preemption decisions live?",
     "o": ["A. journalctl -u runai-scheduler -f (live; preempt = scheduler)",
           "B. journalctl -u runai-engine (no such systemd unit on hosts)",
           "C. /var/log/runai/jobs.log (path varies; not the canonical home)",
           "D. kubectl logs deployment/runai-overseer (incorrect app name)"],
     "a": "A", "e": "Run:ai's scheduler is what preempts; its journal is the place to look."},
    {"d": "T&O", "q": "Which command-line tool tests host↔DPU PCIe-VF connectivity?",
     "o": ["A. ssh student@<dpu-arm> (DPU exposes ssh over PCIe-VF interface)",
           "B. mlxconfig (configures NIC params; not connectivity testing)",
           "C. iperf3 (TCP/UDP throughput; not PCIe-VF specific check)",
           "D. nvidia-smi --dpu (no such flag in nvidia-smi canonical CLI)"],
     "a": "A", "e": "BlueField DPU exposes a virtual function that the host SSHes into."},
]


LAB_TASKS_6 = [
    {
        "title": "Migrate workloads from one MIG layout to another",
        "story": "The cluster currently runs 7×1g.10gb on each H100. Switch "
                 "GPU 0 of node-001 to 3g.40gb × 2 to host larger inference "
                 "models, then verify and re-label the node so the scheduler "
                 "sees the new layout.",
        "expected": [
            ("nvidia-smi", "mig", "-dgi"),
            ("nvidia-smi", "mig", "-cgi"),
            ("nvidia-smi", "mig", "-lgi"),
            ("kubectl", "label", "nodes"),
        ],
        "max_score": 4,
    },
    {
        "title": "Build and audit a custom Slurm partition for inference",
        "story": "Create a 'inference' partition gated to the inference team, "
                 "verify nodes/limits, then audit AllowGroups and update the "
                 "QoS so only members can submit there.",
        "expected": [
            ("scontrol", "show", "partition"),
            ("sacctmgr", "list"),
            ("sacctmgr", "modify"),
            ("sinfo",),
        ],
        "max_score": 4,
    },
    {
        "title": "Push a security patch to all BCM-managed nodes",
        "story": "Chroot into the default image, apt-upgrade, exit, commit, "
                 "and trigger a fleet imageupdate on every node in the "
                 "default category.",
        "expected": [
            ("cm-chroot-sw-img",),
            ("apt", "upgrade"),
            ("commit",),
            ("imageupdate",),
        ],
        "max_score": 4,
    },
    {
        "title": "Move a Triton inference Deployment to a new namespace",
        "story": "Create namespace 'serving', move the Triton Deployment + "
                 "Service + secret + ResourceQuota across, then verify the "
                 "external endpoint still serves.",
        "expected": [
            ("kubectl", "create", "namespace"),
            ("kubectl", "create", "secret", "docker-registry"),
            ("kubectl", "apply",),
            ("kubectl", "get", "service"),
        ],
        "max_score": 4,
    },
    {
        "title": "Bring up a Run:ai inference workload with HPA on GPU util",
        "story": "Submit a Run:ai inference workload pinned to the 'serving' "
                 "project with min 2 / max 8 replicas and HPA on GPU util > "
                 "70%, then verify replicas scale.",
        "expected": [
            ("runai", "submit"),
            ("--inference",),
            ("kubectl", "get", "hpa"),
            ("kubectl", "top"),
        ],
        "max_score": 4,
    },
    {
        "title": "Diagnose 'pod stuck Pending' on a multi-tenant cluster",
        "story": "A user complains their pod has been Pending for 20 minutes. "
                 "Check the pod, the node taints, the GPU operator labels, "
                 "and the Run:ai project quota — fix whichever is wrong.",
        "expected": [
            ("kubectl", "describe", "pod"),
            ("kubectl", "describe", "node"),
            ("runai", "list", "projects"),
            ("kubectl", "get", "nodes"),
        ],
        "max_score": 4,
    },
]


# Six independent question/lab banks, all aligned to the official 31/23/23/23
# blueprint domain weighting.
EXAM_BANKS = {
    "1": {"questions": EXAM_QUESTIONS,   "labs": LAB_TASKS,
          "name": "Exam 1 — General coverage (BCM, Slurm, K8s, Run:ai, NGC)"},
    "2": {"questions": EXAM_QUESTIONS_2, "labs": LAB_TASKS_2,
          "name": "Exam 2 — Mission Control, DOCA, RDMA, monitoring, Helm"},
    "3": {"questions": EXAM_QUESTIONS_3, "labs": LAB_TASKS_3,
          "name": "Exam 3 — GPU operator, Slurm accounting, networking, MIG, fabric"},
    "4": {"questions": EXAM_QUESTIONS_4, "labs": LAB_TASKS_4,
          "name": "Exam 4 — Image rollout, ECC recovery, etcd snapshot, log triage"},
    "5": {"questions": EXAM_QUESTIONS_5, "labs": LAB_TASKS_5,
          "name": "Exam 5 — Multi-tenant Run:ai, Helm rollback, RBAC, fairness"},
    "6": {"questions": EXAM_QUESTIONS_6, "labs": LAB_TASKS_6,
          "name": "Exam 6 — MIG migration, partitions, security patch, inference"},
}


def _select_questions(question_pool: list[dict] = None) -> list[dict]:
    """Pick 30 questions with weighted random sampling per domain
    from the supplied pool (defaults to EXAM_QUESTIONS for backwards compat)."""
    if question_pool is None:
        question_pool = EXAM_QUESTIONS
    pool = {d: [q for q in question_pool if q["d"] == d]
            for d in DOMAIN_TARGETS}
    chosen: list[dict] = []
    for d, n in DOMAIN_TARGETS.items():
        avail = pool[d]
        if len(avail) >= n:
            chosen.extend(random.sample(avail, n))
        else:
            chosen.extend(avail)
            extras = [q for q in question_pool if q not in chosen]
            chosen.extend(random.sample(extras, n - len(avail)))
    random.shuffle(chosen)
    return chosen


def _grade_lab(typed_lines: list[str], lab: dict) -> tuple[int, list[str]]:
    """Score a lab: count how many expected token-tuples are matched by any line."""
    hit_indices: set[int] = set()
    for line in typed_lines:
        low = line.lower()
        for idx, tokens in enumerate(lab["expected"]):
            if idx in hit_indices:
                continue
            if all(t.lower() in low for t in tokens):
                hit_indices.add(idx)
    score = len(hit_indices)
    missed = [
        " + ".join(lab["expected"][i])
        for i in range(len(lab["expected"]))
        if i not in hit_indices
    ]
    return score, missed


def _fmt_remaining(start_ts: float, total_min: int) -> str:
    elapsed = time.time() - start_ts
    remaining = max(0, int(total_min * 60 - elapsed))
    m, s = divmod(remaining, 60)
    return f"{m:02d}:{s:02d}"


def _pad_options(opts: list[str]) -> list[str]:
    """Pad option texts (after the 'A. '/'B. ' prefix) to the longest length
    so option length doesn't telegraph the answer (length-bias mitigation)."""
    parts = [(o[0:3], o[3:].strip()) for o in opts]   # ('A. ', '<text>')
    longest = max(len(p[1]) for p in parts)
    return [f"{prefix}{text.ljust(longest)}" for prefix, text in parts]


def _shuffle_and_equalize(q: dict) -> dict:
    """Return a fresh copy of question q with:
       - options shuffled        (eliminates 'correct answer is always A' bias)
       - re-lettered correct ans (the new correct letter is wherever the
                                  original correct text ended up)
       - content-length equalized using neutral middle-dot filler so the
         correct answer cannot be identified by being visibly longer.

    Run on every question presentation, so even the same item shows
    differently if it ever recurs.
    """
    # cleaned: list of (original_letter, text_without_prefix)
    cleaned = [(o[0], o[3:].rstrip()) for o in q["o"]]
    correct_text = next(t for letter, t in cleaned if letter == q["a"])

    # Shuffle the (letter, text) pairs in place.
    shuffled = list(cleaned)
    random.shuffle(shuffled)

    target = max(len(t) for _, t in shuffled)

    new_opts: list[str] = []
    new_ans = ""
    for new_letter, (orig_letter, text) in zip("ABCD", shuffled):
        # Identify the correct letter BEFORE we apply padding so the
        # match is on the unmodified text — no string-stripping ambiguity.
        if text == correct_text:
            new_ans = new_letter
        gap = target - len(text)
        if gap > 1:
            display_text = f"{text} " + ("·" * (gap - 1))
        else:
            display_text = text + (" " * gap)
        new_opts.append(f"{new_letter}. {display_text}")

    return {**q, "o": new_opts, "a": new_ans}


def _grade_lab_detailed(typed_lines: list[str], lab: dict) -> tuple[int, list[dict]]:
    """Like _grade_lab but returns per-expected-command results so the
    lab phase can give detailed feedback right away."""
    results: list[dict] = []
    typed_low = [t.lower() for t in typed_lines]
    for tokens in lab["expected"]:
        hit_line = None
        for line in typed_low:
            if all(t.lower() in line for t in tokens):
                hit_line = line; break
        results.append({
            "expected": " + ".join(tokens),
            "hit": hit_line is not None,
            "matched_line": hit_line or "",
        })
    score = sum(1 for r in results if r["hit"])
    return score, results


def cmd_mock_exam(args: list[str], s: dict) -> int:
    """Run a 120-minute NCP-AIOL mock exam (30 MCQ + 3 labs).

    USAGE:
        mock-exam [BANK]            interactive mode (BANK = 1..6)
        mock-exam <BANK> start      skip the confirmation prompt
        mock-exam --help            print this help

    There are six independent question banks. Each bank uses the official
    NCP-AIOL blueprint weighting (31% I&D · 23% ADM · 23% WLM · 23% T&O)
    and presents 30 multiple-choice questions plus 3 hands-on lab tasks.

    Feedback is given immediately after each MCQ answer and after each lab
    is submitted, so you learn as you go.
    """
    # Parse: figure out which bank (default = 1) and whether 'start' present
    bank_id = "1"
    auto_start = False
    if args and args[0] in ("-h", "--help", "help"):
        # Print the docstring as a real OS-style help screen
        for line in cmd_mock_exam.__doc__.splitlines():
            info(line.lstrip())
        info("AVAILABLE EXAM BANKS:")
        for k, b in EXAM_BANKS.items():
            info(f"    {k}    {b['name']}")
        return 0
    for tok in args:
        if tok in EXAM_BANKS:
            bank_id = tok
        elif tok == "start":
            auto_start = True

    bank = EXAM_BANKS[bank_id]
    print("=" * 70)
    print(f" NVIDIA NCP-AIOL Mock Exam — Bank {bank_id}")
    print(f" {bank['name']}")
    print(" 30 multiple-choice questions + 3 hands-on labs · 120 minutes")
    print(" Domain weighting: 31% I&D · 23% ADM · 23% WLM · 23% T&O")
    print("=" * 70)
    print(" This is a SIMULATED practice exam. The real exam questions are NDA-")
    print(" protected. These items exercise the same skills as the official")
    print(" blueprint at https://www.nvidia.com/en-us/learn/certification/ai-")
    print(" operations-professional/")
    print(" Available banks: " + ", ".join(EXAM_BANKS.keys()))
    print("=" * 70)
    if auto_start:
        confirm = "yes"
    else:
        try:
            confirm = input(" Type 'start' to begin, anything else to cancel: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return 0
    if confirm not in ("start", "yes", "y", "begin"):
        info("Mock exam cancelled.")
        return 0

    questions = _select_questions(bank["questions"])
    labs = random.sample(bank["labs"], 3)
    answers: dict[int, str] = {}
    lab_results: list[dict] = []
    start_ts = time.time()
    TOTAL_MIN = 120

    # ----- Multiple-choice phase: immediate per-question feedback -----
    # NOTE: questions are stored with original A/B/C/D mapping but we
    # shuffle + length-equalize each one before display — so the user
    # never sees a length or position pattern.
    for qi, q_raw in enumerate(questions, 1):
        q = _shuffle_and_equalize(q_raw)
        rem = _fmt_remaining(start_ts, TOTAL_MIN)
        if rem == "00:00":
            print("\n*** Time's up — auto-submitting ***")
            break
        print(f"\n[Q{qi}/30  {DOMAIN_NAME[q['d']]}  ⏱ {rem} remaining]")
        print(f" {q['q']}")
        for opt in _pad_options(q["o"]):
            print(f"   {opt}")
        try:
            ans = input(" Your answer (A/B/C/D, 'skip', 'quit'): ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            print("\nExam aborted.")
            return 0
        if ans == "QUIT":
            print("Exam aborted.")
            return 0
        if ans == "SKIP" or ans == "":
            print(f" → Skipped. (Correct answer was {q['a']}.)")
            print(f"   Explanation: {q['e']}")
            continue
        if ans in ("A", "B", "C", "D"):
            answers[qi - 1] = ans
            if ans == q["a"]:
                print(f" ✓ Correct.")
            else:
                correct_text = next(
                    (o for o in q["o"] if o.startswith(q["a"] + ".")), q["a"])
                print(f" ✗ Incorrect. The correct answer was {q['a']}.")
                print(f"   {correct_text.strip()}")
            print(f"   Explanation: {q['e']}")
        else:
            print(f" → Unrecognised input '{ans}' — counted as skipped.")
            print(f"   Correct answer was {q['a']}: {q['e']}")

    # ----- Lab phase: per-lab immediate feedback -----
    print("\n" + "=" * 70)
    print(" LAB PHASE — 3 hands-on exercises")
    print(" Type the commands you would run, one per line. Type 'done' to submit.")
    print(" During a lab you can also type:")
    print("   man <cmd>           — view a real-shape man page")
    print("   <cmd> --help        — same effect for known commands")
    print("   help                — show this menu")
    print("   list                — show what you've typed so far")
    print("   undo                — remove the most recent command")
    print(" Anything else is recorded as a command attempt for grading.")
    print("=" * 70)
    LAB_HELPABLE = list(MANPAGES.keys())
    for li, lab in enumerate(labs, 1):
        rem = _fmt_remaining(start_ts, TOTAL_MIN)
        if rem == "00:00":
            print("*** Time's up before all labs were attempted ***")
            break
        print(f"\n--- Lab {li} of 3   ⏱ {rem} remaining ---")
        print(f" Title:  {lab['title']}\n")
        print(f" {lab['story']}\n")
        typed: list[str] = []
        while True:
            try:
                line = input(f" lab{li}> ").strip()
            except (EOFError, KeyboardInterrupt):
                print(); break
            if not line:
                continue
            tokens = line.split()
            head = tokens[0].lower()

            # ----- Lab-only meta commands (don't get recorded for grading) -----
            if head in ("done", "submit"):
                break
            if head in ("quit", "abort"):
                return 0
            if head == "help":
                print("   man <cmd>           — view a real-shape man page")
                print("   <cmd> --help        — same effect for known commands")
                print(f"   commands with man pages: {', '.join(LAB_HELPABLE)}")
                print("   list                — show what you've typed so far")
                print("   undo                — remove your last typed command")
                print("   done                — submit answers + see grading")
                print("   quit                — abandon the exam (no grading)")
                continue
            if head == "list":
                if not typed:
                    print("   (you haven't typed any commands yet)")
                else:
                    print(f"   You have typed {len(typed)} command(s):")
                    for i, t in enumerate(typed, 1):
                        print(f"     {i}. {t}")
                continue
            if head == "undo":
                if typed:
                    removed = typed.pop()
                    print(f"   removed: '{removed}'   (now {len(typed)} command(s))")
                else:
                    print("   nothing to undo")
                continue
            if head == "man":
                if len(tokens) >= 2:
                    _print_manpage(tokens[1])
                else:
                    print("   usage: man <command>")
                continue
            if (len(tokens) >= 2 and tokens[1] in ("--help", "-h")
                    and tokens[0] in MANPAGES):
                _print_manpage(tokens[0])
                continue

            # ----- Otherwise: record as a graded command attempt -----
            typed.append(line)
        score, details = _grade_lab_detailed(typed, lab)
        lab_results.append({"title": lab["title"], "score": score,
                             "max": lab["max_score"], "details": details,
                             "typed": typed})
        # Immediate detailed feedback for THIS lab
        print(f"\n Lab {li} graded: {score}/{lab['max_score']} expected commands matched.")
        print(" Per-step feedback:")
        for d in details:
            mark = "✓" if d["hit"] else "✗"
            line = f"   {mark} {d['expected']}"
            if d["hit"]:
                line += f"      (matched line: '{d['matched_line']}')"
            else:
                line += "      (no typed command contained these tokens)"
            print(line)
        if score < lab["max_score"]:
            print(" 📚 Tip: review the corresponding scenario walk-throughs "
                  "(`scenarios` then `scenario <n>`) to see the canonical commands.")

    # ----- Score -----
    elapsed = int(time.time() - start_ts)
    em, es = divmod(elapsed, 60)
    print("\n" + "=" * 70)
    print(" RESULTS")
    print("=" * 70)
    correct = 0
    domain_stats: dict[str, dict[str, int]] = {d: {"t": 0, "c": 0} for d in DOMAIN_TARGETS}
    for qi, q in enumerate(questions):
        domain_stats[q["d"]]["t"] += 1
        if answers.get(qi) == q["a"]:
            correct += 1
            domain_stats[q["d"]]["c"] += 1
    total_qs = len(questions)
    mcq_pct = round(correct / total_qs * 100, 1) if total_qs else 0.0

    lab_correct = sum(r["score"] for r in lab_results)
    lab_max = sum(r["max"] for r in lab_results) or 1
    lab_pct = round(lab_correct / lab_max * 100, 1)

    overall = round((mcq_pct + lab_pct) / 2, 1)
    pass_threshold = 70.0
    verdict = "PASS" if overall >= pass_threshold else "FAIL"

    print(f"\n MCQ score:    {correct}/{total_qs}     ({mcq_pct}%)")
    print(f" Lab score:    {lab_correct}/{lab_max}      ({lab_pct}%)")
    print(f" Combined:     {overall}%   →   {verdict} (cut: 70%)")
    print(f" Time used:    {em}m {es}s of 120m\n")

    print(" Domain breakdown")
    print(" " + "-" * 60)
    for d, stats in domain_stats.items():
        pct = round(stats["c"] / stats["t"] * 100, 1) if stats["t"] else 0.0
        bar = "█" * int(pct / 5)
        print(f"  {DOMAIN_NAME[d]:30}  {stats['c']:>2}/{stats['t']:<2}  "
              f"{pct:>5.1f}%  {bar}")

    print("\n Lab review")
    for r in lab_results:
        status = "✓" if r["score"] == r["max"] else "✗"
        print(f"  {status} {r['title']}")
        print(f"     scored {r['score']}/{r['max']}")
        missed = [d["expected"] for d in r.get("details", []) if not d["hit"]]
        if missed:
            print(f"     missed: {' · '.join(missed)}")

    if not lab_results:
        print("  (no labs completed)")

    print("\n" + "=" * 70)
    print(" Review your weakest domains and re-run with: mock-exam")
    print(" Type `scenarios` for the 44 study scenarios.")
    print("=" * 70)
    return 0


# ===========================================================================
# Help, scenarios, reset
# ===========================================================================
TOP_HELP = """\
Cluster CLI Simulator — practice for NCP-AIIOL

RUN:AI
  runai whoami | cluster info
  runai list projects | departments | jobs | workspaces
  runai config project <name>
  runai submit <name> -p <proj> -g <n> -i <image> -- <cmd>
  runai submit-mpi <name> -p <proj> -g <n> --workers <w> -- <cmd>
  runai describe job <name>     | runai logs <name>
  runai delete job <name>       | runai delete project <name>
  runai create project <name>   | runai create department <name>
  runai update project <name> --gpu-quota <n>
  runai top job

SLURM
  sinfo [-N]               sbatch <script>          srun -N <n> --gres=gpu:<g> ...
  squeue [-u <user>]       salloc -N <n>            scancel <jobid>
  scontrol show node|job   scontrol update NodeName=... State=DRAIN|RESUME
  sacct [-u <user>]        sshare       sprio
  seff <jobid>             sreport      sacctmgr {list|add} {account|user|qos}

KUBERNETES
  kubectl get {nodes|pods|namespaces|clusterpolicy|pvc} [-A] [-o wide]
  kubectl describe {node|pod} <name>
  kubectl logs <pod>       kubectl exec <pod> -- <cmd>
  kubectl top {node|pod}   kubectl drain|cordon|uncordon <node>
  kubectl create {namespace|deployment|configmap|secret} <name>
  kubectl scale|rollout|run|label|taint|edit|explain|apply|port-forward
  helm list [-A]   helm install <release> <chart>   helm status|repo|history

BASE COMMAND MANAGER (BCM)
  cmsh                            interactive shell — nested mode prompts
  cmsh -c "device list"           one-shot form
  cmsh -c "device status node-007"
  cmsh -c "category list"         cmsh -c "softwareimage list"
  cmsh -c "monitoring alerts"     cmsh -c "wlm; status"
  cm-info        cm-version        cmha status        mhcheck
  ndlist         node-installer-status
  cm-wlm-setup                    BCM Slurm setup wizard
  cm-kubernetes-setup [--add-user <name>] [--list-users]

ENVIRONMENT MODULES
  module available                list all installed modules
  module load slurm/slurm/23.02.8
  module load kubernetes/default/1.30.10-1.1
  module list / unload / show / purge

SHELL / IDENTITY
  su - k8suser                    switch user — prompt becomes k8suser@bcm:~$
  ssh student@dgx02               jump to DGX, prompt becomes student@dgx02:~$
  cat /cm/shared/apps/lp/gpu.yaml shows the lab pod manifest

DCGMI (extended)
  dcgmi discovery -l              list GPUs + NvSwitches
  dcgmi group -l                  list groups
  dcgmi group -c student_group    create group (returns group ID)
  dcgmi group -g 9 -a 0,1,2,nvswitch:12     add entities
  dcgmi group -g 9 -r 0,1,2,nvswitch:12     remove
  dcgmi group -d 9                delete
  dcgmi config -g 9 --get         show group config

GPU / LINUX CLI TOOLS
  nvidia-smi [topo|nvlink|--query-gpu=...|-L]
  nvidia-bug-report.sh
  dcgmi {diag -r 3 | health | profile | stats | group | discovery}
  dmesg [ | grep xid ]
  ipmitool {sdr|sel|chassis status|lan print}
  mlxlink        mlxfwmanager        mst status

BUILT-IN
  help                   this message
  scenarios              list exam-style practice scenarios
  scenario <n>           walk through scenario n
  reset                  wipe cluster state and reseed
  state                  dump current state (debug)
  quit / exit / Ctrl-D   leave the REPL
"""


SCENARIOS = [
    {
        "title": "S1: Drain a node for PSU replacement",
        "story": "Node-007 is reporting 11.55V on the 12V rail. You need "
                 "to drain it from the scheduler before swapping the PSU.",
        "tasks": [
            "scontrol show node node-007    # confirm current state",
            "scontrol update NodeName=node-007 State=DRAIN Reason=PSU-swap",
            "sinfo -N | grep node-007       # verify it's drained",
            "kubectl cordon node-007        # also cordon from k8s",
            "# (after PSU replaced)",
            "scontrol update NodeName=node-007 State=RESUME",
            "kubectl uncordon node-007",
        ],
    },
    {
        "title": "S2: Submit a Run:ai training job and check status",
        "story": "Submit a 16-GPU PyTorch training job to the ml-research "
                 "project, watch it start, and check its logs.",
        "tasks": [
            "runai config project ml-research",
            "runai submit my-bert -p ml-research -g 16 "
              "-i nvcr.io/nvidia/pytorch:24.03-py3 -- python train.py",
            "runai list jobs -p ml-research",
            "runai describe job my-bert",
            "runai logs my-bert",
        ],
    },
    {
        "title": "S3: Investigate a Pending job",
        "story": "Job 'embeddings-batch' has been Pending for several minutes. "
                 "Find out why and decide what to do.",
        "tasks": [
            "runai describe job embeddings-batch",
            "runai list projects                 # check ml-research quota",
            "sinfo                                # any partition full?",
            "scontrol show job 12473",
            "# remediation: wait for resources, raise quota, or run with smaller -g",
        ],
    },
    {
        "title": "S4: Submit a Slurm batch job",
        "story": "Write a Slurm batch script and submit it.  Verify it lands "
                 "on a healthy node.",
        "tasks": [
            "# create a script train.sbatch with these directives:",
            "#   #SBATCH -J my-train",
            "#   #SBATCH -p gpu",
            "#   #SBATCH -N 2",
            "#   #SBATCH --gres=gpu:8",
            "#   #SBATCH -t 04:00:00",
            "sbatch train.sbatch",
            "squeue -u $USER",
            "scontrol show job <jobid>",
        ],
    },
    {
        "title": "S5: Inspect GPU usage in a Kubernetes pod",
        "story": "An ML engineer says their pod is GPU-starved.  Inspect the "
                 "node, pod, and exec nvidia-smi inside.",
        "tasks": [
            "kubectl get pods -n ml-team -o wide",
            "kubectl describe pod bert-pretrain-0-0 -n ml-team",
            "kubectl describe node node-001",
            "kubectl exec bert-pretrain-0-0 -- nvidia-smi",
            "kubectl top node",
        ],
    },
    {
        "title": "S6: Cancel a runaway job",
        "story": "A job is consuming GPUs but its owner reports it's stuck.  "
                 "Cancel it.",
        "tasks": [
            "squeue",
            "scancel 12471",
            "squeue                                # confirm it's gone",
            "# Run:ai equivalent:",
            "runai delete job bert-pretrain",
        ],
    },
    {
        "title": "S7: Verify the GPU operator stack on Kubernetes",
        "story": "Check that the NVIDIA GPU operator is healthy and DCGM "
                 "exporter is running.",
        "tasks": [
            "kubectl get clusterpolicy",
            "kubectl get pods -n gpu-operator",
            "kubectl describe node node-001 | grep -i nvidia",
            "helm list -A                          # see all helm releases",
            "helm status gpu-operator",
        ],
    },
    {
        "title": "S8: BCM — inspect a node and view its software image",
        "story": "From the BCM head node, audit node-007's status, the "
                 "category it belongs to, and the software image installed.",
        "tasks": [
            "cm-info                                 # cluster summary",
            "cm-version",
            "cmha status                             # HA pair healthy?",
            "cmsh -c 'device list'                   # all devices",
            "cmsh -c 'device status node-007'        # is the node UP?",
            "cmsh -c 'category list'                 # which category?",
            "cmsh -c 'softwareimage list'            # available images",
            "mhcheck                                  # cluster health",
            "ndlist                                   # provisioning list",
        ],
    },
    {
        "title": "S9: BCM — clone an image and roll it to a new category",
        "story": "Create a new GPU image variant by cloning the existing "
                 "h100-cuda12.4-image, then verify it appears.",
        "tasks": [
            "cmsh -c 'softwareimage list'",
            "cmsh -c 'softwareimage clone h100-cuda12.4-image h100-cuda12.5-image'",
            "cmsh -c 'softwareimage list'",
            "cmsh -c 'softwareimage commit'",
            "node-installer-status",
        ],
    },
    {
        "title": "S10: Diagnose a hardware failure with dcgmi + nvidia-smi",
        "story": "An H100 on node-007 is reporting XID 79.  Triage it from "
                 "the command line: check dmesg, run dcgmi diag -r 3, "
                 "generate a bug report.",
        "tasks": [
            "dmesg | grep xid                        # find the XID event",
            "nvidia-smi --query-gpu=index,uuid,ecc.errors.uncorrected.aggregate.total --format=csv",
            "nvidia-smi topo                          # NVLink topology",
            "dcgmi health                             # DCGM health check",
            "dcgmi diag -r 3 -i 0                     # extended diagnostic on GPU 0",
            "nvidia-bug-report.sh                     # collect for NVIDIA support",
            "scontrol update NodeName=node-007 State=DRAIN Reason=XID79",
            "kubectl cordon node-007",
        ],
    },
    {
        "title": "S11: BMC / PSU investigation with ipmitool",
        "story": "Node-007 PSU was flagged.  Use ipmitool to confirm the "
                 "12V rail and check the BMC event log.",
        "tasks": [
            "ipmitool sdr list | grep -i 12v",
            "ipmitool sel list | grep -i psu",
            "ipmitool chassis status",
            "cmsh -c 'monitoring alerts'",
        ],
    },
    {
        "title": "S12: NIC / RoCEv2 troubleshooting (BlueField-3 / ConnectX-7)",
        "story": "NCCL bandwidth dropped 12% on a training cluster.  Check "
                 "the NIC firmware, link state, and FEC corrections.",
        "tasks": [
            "mst status                                  # find the device",
            "mlxfwmanager                                # firmware version",
            "mlxlink                                     # physical layer",
            "srun -N 4 --gres=gpu:8 nccl all_reduce_perf  # repro",
        ],
    },
    {
        "title": "S13: Run:ai admin — create project, set quota, submit MPI job",
        "story": "Onboard a new team: create their project, set a 16-GPU "
                 "quota, then submit a distributed PyTorch MPI job.",
        "tasks": [
            "runai create project new-team",
            "runai update project new-team --gpu-quota 16",
            "runai list projects",
            "runai submit-mpi distributed-train -p new-team -g 8 --workers 4 "
              "-i nvcr.io/nvidia/pytorch:24.03-py3 -- mpirun python train.py",
            "runai list jobs -p new-team",
        ],
    },
    {
        "title": "S14: Slurm fairshare investigation",
        "story": "A team complains their jobs sit Pending while another "
                 "team's run.  Check fair-share and priorities.",
        "tasks": [
            "squeue                                  # who's running, who's waiting",
            "sshare                                  # raw vs effective usage",
            "sprio                                   # priority breakdown",
            "sacctmgr list qos",
            "sreport                                 # monthly utilization",
        ],
    },
    {
        "title": "S15: Helm — upgrade the GPU operator",
        "story": "Bump the NVIDIA GPU operator chart to a new revision and "
                 "verify rollout.",
        "tasks": [
            "helm list -A",
            "helm repo update",
            "helm upgrade gpu-operator nvidia/gpu-operator -n gpu-operator",
            "helm history gpu-operator",
            "kubectl get pods -n gpu-operator",
            "kubectl get clusterpolicy",
        ],
    },
    {
        "title": "S16: NCP-AIIOL Lab — Provision a Slurm cluster from BCM",
        "story": "Starting from a blank BCM head node, clone an image, "
                 "create a category, configure four nodes, then run "
                 "the cm-wlm-setup wizard.",
        "tasks": [
            "module available",
            "cmsh                                  # enter interactive cmsh",
            "softwareimage",
            "clone default-image slurm-image",
            "commit",
            "..                                    # back to [bcm]%",
            "category",
            "clone default slurm",
            "set softwareimage slurm-image",
            "commit",
            "..",
            "device",
            "foreach -c slurm -n slurm-01..slurm-04 (set category slurm)",
            "commit",
            "ls                                    # confirm 4 slurm nodes",
            "exit                                  # leave cmsh",
            "cm-wlm-setup",
            "module load slurm/slurm/23.02.8",
            "sinfo",
            "srun -N 4 hostname",
            "srun -N 1 --gres=gpu:1 hostname slurm-04",
        ],
    },
    {
        "title": "S17: NCP-AIIOL Lab — Provision a Kubernetes cluster from BCM",
        "story": "Provision K8s control-plane and worker nodes via BCM, "
                 "then add a non-root user and deploy a GPU pod.",
        "tasks": [
            "cmsh",
            "softwareimage",
            "clone default-image k8s-control-plane-image",
            "commit",
            "..",
            "clone default-image k8s-worker-image",
            "commit",
            "..",
            "category",
            "clone default k8s-control-plane",
            "set softwareimage k8s-control-plane-image",
            "commit",
            "..",
            "clone default k8s-worker",
            "set softwareimage k8s-worker-image",
            "commit",
            "..",
            "device",
            "foreach -c k8s-control-plane -n k8s-control-plane-01..03 (set category k8s-control-plane)",
            "foreach -c k8s-worker -n k8s-worker-01 (set category k8s-worker)",
            "commit",
            "ls",
            "exit",
            "cm-kubernetes-setup",
            "cm-kubernetes-setup --add-user k8suser",
            "cm-kubernetes-setup --list-users",
            "su - k8suser",
            "module load kubernetes/default/1.30.10-1.1",
            "kubectl get nodes",
            "cat /cm/shared/apps/lp/gpu.yaml",
            "kubectl apply -f /cm/shared/apps/lp/gpu.yaml",
            "kubectl get pods                       # ContainerCreating",
            "kubectl get pods                       # then Completed",
            "kubectl logs gpu-pod                   # see nvidia-smi inside pod",
        ],
    },
    {
        "title": "S18: NCP-AIIOL Lab — DCGM group management on DGX02",
        "story": "SSH from the head node to a DGX, then create a DCGM group "
                 "with GPUs and an NvSwitch, configure it, and tear it down.",
        "tasks": [
            "ssh student@dgx02",
            "dcgmi discovery -l                     # 8 GPUs + 6 NvSwitches",
            "dcgmi group -l                         # default groups",
            "dcgmi group -c student_group           # creates group ID 9",
            "dcgmi group -g 9 -a 0,1,2,nvswitch:12  # add GPUs + NvSwitch",
            "dcgmi config -g 9 --get                # inspect group settings",
            "dcgmi group -g 9 -r 0,1,2,nvswitch:12  # remove entities",
            "dcgmi group -d 9                       # delete the group",
            "exit                                    # back to bcm",
        ],
    },
    {
        "title": "S19: Lab — Install the NVIDIA Container Toolkit",
        "story": "Set up the NVIDIA Container Toolkit on a fresh Ubuntu 22.04 host "
                 "so Docker can pass GPUs through to containers.",
        "tasks": [
            "# Configure the production repository (one long curl pipeline):",
            "curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey",
            "curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list",
            "sudo apt-get update",
            "sudo apt-get install -y nvidia-container-toolkit",
            "sudo nvidia-ctk runtime configure",
            "sudo systemctl restart docker",
            "# Verify GPU passthrough:",
            "sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi",
        ],
    },
    {
        "title": "S20: Lab — Train a model in a PyTorch Docker container",
        "story": "Pull the NVIDIA PyTorch container, drop into an interactive "
                 "shell, train a digit recognizer, and compare GPU vs CPU speed.",
        "tasks": [
            "sudo docker pull nvcr.io/nvidia/pytorch:25.01-py3",
            "mkdir docker_example",
            "# Run the container WITH GPUs:",
            "sudo docker run --gpus all --ipc=host -it --rm "
              "-v /home/student/AI_Infra/module8:/workspace nvcr.io/nvidia/pytorch:25.01-py3",
            "# (Inside container — prompt becomes root@<id>:/workspace#)",
            "nvidia-smi                              # baseline before training",
            "python train.py                         # ~1m22s on H100",
            "nvidia-smi                              # GPU now busy at ~94%",
            "python test.py                          # default 7.png",
            "python test.py 8.png",
            "exit                                    # leave container",
            "# Now CPU-only (omit --gpus):",
            "sudo docker run --ipc=host -it --rm "
              "-v /home/student/AI_Infra/module8:/workspace nvcr.io/nvidia/pytorch:25.01-py3",
            "python train.py                         # 12+ minutes — much slower",
            "exit",
        ],
    },
    {
        "title": "S21: Lab — Install the NVIDIA driver from scratch",
        "story": "Survey the host, gather details for the NVIDIA driver "
                 "download page, then install the driver and verify.",
        "tasks": [
            "# 1) Confirm there's no driver yet:",
            "nvidia-smi                              # 'command not found'",
            "# 2) Gather GPU + OS + arch info for the website:",
            "lspci -Q | grep NVIDIA                  # H100 detected",
            "cat /etc/os-release                     # Ubuntu 22.04",
            "uname -a                                # x86_64",
            "# 3) Install the driver .deb that's pre-staged on the lab host:",
            "sudo apt-get remove --purge '^nvidia-.*'",
            "sudo dpkg -i nvidia-driver-local-repo-ubuntu2204-570.86.15_1.0-1_amd64.deb",
            "sudo cp /var/nvidia-driver-local-repo-ubuntu2204-570.86.15/nvidia-driver-local-081EF1BD-keyring.gpg /usr/share/keyrings/",
            "sudo apt update",
            "sudo apt install nvidia-driver-570 -y",
            "nvidia-smi                              # driver now reported",
        ],
    },
    {
        "title": "S22: Lab — nvidia-smi query forms",
        "story": "Use --query-gpu= to extract specific fields in CSV form, "
                 "and -l to loop continuously.",
        "tasks": [
            "nvidia-smi --query-gpu=utilization.gpu --format=csv",
            "nvidia-smi --query-gpu=memory.used --format=csv",
            "nvidia-smi --query-gpu=utilization.gpu,memory.total,memory.used,"
              "memory.free,temperature.gpu,power.draw --format=csv -l 2",
        ],
    },
    {
        "title": "S23: Lab — BMC tour (DGX A100 Baseboard Management Controller)",
        "story": "Log into the BMC, look at the dashboard, sensors, and GPU info "
                 "to answer the lab worksheet questions.",
        "tasks": [
            "bmc login                               # student / studentpassword",
            "bmc dashboard                           # uptime, deassertions, firmware",
            "bmc sensors                             # PWR_GB_GPU# rows",
            "bmc gpu-info                            # marketing name + total memory",
            "bmc fru-info                            # chassis FRU",
            "bmc system-inventory",
            "bmc signout",
        ],
    },
    {
        "title": "S24: BCM Admin Lab P0/P1 — Setup script + node provisioning",
        "story": "Run the lab setup script, then provision four worker nodes — "
                 "one by manual PXE selection, one by Base View MAC entry, "
                 "and two via the readmac.sh CSV.",
        "tasks": [
            "./setup_lab.sh",
            "cmsh -c 'device list'                   # 4 nodes, all DOWN, MAC=00:..",
            "# (Worker01 GUI: select node001 manually — simulated as:)",
            "cmsh -c 'device use node001; set mac 4E:56:44:41:01:01; commit'",
            "# (Base View: enter MAC for node002:)",
            "cmsh -c 'device use node002; set mac 4E:56:44:41:01:02; commit'",
            "# Run the readmac script for the remaining two nodes:",
            "./readmac.sh nodes.csv",
            "cmsh -c 'device list'                   # all 4 should be UP",
        ],
    },
    {
        "title": "S25: BCM Admin Lab P2 — Software images + chroot + kernel modules",
        "story": "Clone the default image, chroot in to add a test file, then "
                 "add a kernel module via cmsh and reboot to verify.",
        "tasks": [
            "cmsh -c 'softwareimage list'",
            "cmsh -c 'softwareimage clone default-image default-image-backup; commit'",
            "cmsh -c 'softwareimage list'",
            "# Now chroot into the default image and create a test file:",
            "cm-chroot-sw-img /cm/images/default-image",
            "touch test.txt",
            "ls                                      # see test.txt",
            "exit                                    # leave chroot",
            "# Add a kernel module via cmsh interactive:",
            "cmsh                                    # then inside cmsh:",
            "  softwareimage",
            "  use default-image",
            "  kernelmodules",
            "  list                                  # soundcore NOT yet listed",
            "  add soundcore",
            "  commit                                # ramdisk regenerated",
            "  list                                  # soundcore now present",
            "  exit ; exit ; exit ; quit",
            "# Verify on node001 BEFORE reboot — neither file nor module visible:",
            "rshell node001",
            "ls /                                    # no test.txt",
            "lsmod | grep soundcore                  # empty",
            "exit                                    # leave rshell",
            "# Reboot node001 to sync the image, then verify AFTER:",
            "reboot node001",
            "rshell node001",
            "ls /                                    # test.txt now present",
            "lsmod | grep soundcore                  # soundcore loaded",
            "exit",
        ],
    },
    {
        "title": "S26: BCM Admin Lab P3 — Categories, foreach, range, imageupdate",
        "story": "Clone a category, move a node into it with a different image, "
                 "drive a fleet-wide change with foreach, and revert with range.",
        "tasks": [
            "cmsh -c 'category list'",
            "cmsh -c 'category listnodes default'    # all 4 nodes",
            "cmsh -c 'category clone default Lite; commit'",
            "cmsh -c 'category use Lite; set softwareimage default-image-backup; commit'",
            "cmsh -c 'category show'                 # verify",
            "cmsh -c 'device set node003 category Lite; commit'",
            "cmsh -c 'device list'                   # node003 shows Lite",
            "imageupdate -n node003 -w               # provision new image",
            "cmsh -c 'device get node001 softwareimage'   # default-image",
            "cmsh -c 'device get node003 softwareimage'   # default-image-backup",
            "rshell node001 ; ls / ; exit            # test.txt visible",
            "rshell node003 ; ls / ; exit            # test.txt NOT visible",
            "# Bulk operation across the default category:",
            "cmsh                                    # inside cmsh:",
            "  device",
            "  foreach -c default (set softwareimage default-image-backup)",
            "  commit",
            "  foreach -c default (get softwareimage)",
            "  range -n node001..node004",
            "  clear softwareimage",
            "  commit",
            "  get softwareimage                     # back to category-inherited",
            "  reboot                                # all 4 reboot",
            "  exit ; exit ; quit",
        ],
    },
    {
        "title": "S27: BCM Admin Lab P4 — Users and groups via cmsh + Base View",
        "story": "Add the lab user 'slurmy' with a password, add 'group2' and "
                 "make slurmy a member, then remove the test user.",
        "tasks": [
            "cmsh                                    # interactive:",
            "  user",
            "  add slurmy",
            "  commit",
            "  set password Welcome123!",
            "  commit",
            "  list                                  # slurmy now shown",
            "  show                                  # parameters of current user",
            "  exit",
            "  group",
            "  add group2",
            "  commit",
            "  append members slurmy ; commit",
            "  list",
            "  exit ; quit",
            "# Or one-shot from outside cmsh:",
            "bcmuser add test1",
            "bcmuser",
            "bcmuser remove test1",
            "bcmgroup add admins",
            "bcmgroup",
        ],
    },
    {
        "title": "S28: BCM Admin Lab P5 — Custom metric, health check, action, dashboard",
        "story": "Build a Base-View-style ramp metric with a paired health check "
                 "and trigger that fires an action when the value exceeds 95.",
        "tasks": [
            "monitoring add-producer --name ramp --type metric "
              "--script /cm/shared/ramp.sh --interval 1",
            "monitoring add-producer --name ramphealthcheck --type healthcheck "
              "--script /cm/shared/rampcheck.sh --interval 20",
            "monitoring add-action --name rampaction --script /cm/shared/rampaction.sh",
            "monitoring add-trigger --name 'Ramp Trigger' --measurable ramp "
              "--operator '>' --value 95 --during rampaction",
            "monitoring add-dashboard --name ramp-dashboard --data ramp",
            "monitoring list                         # confirm everything wired up",
        ],
    },
    # =====================================================================
    # NCP-AIO exam-aligned scenarios (Domain 1: Installation & Deployment,
    # Domain 2: Administration, Domain 3: Workload Management,
    # Domain 4: Troubleshooting & Optimization).
    # =====================================================================
    {
        "title": "S29: NCP-AIO D1 — End-to-end cluster bring-up (BCM + K8s + Slurm + Run:ai)",
        "story": "From a fresh BCM head node, provision compute nodes, deploy "
                 "Kubernetes, deploy Slurm, then lay Run:ai on top. Mirrors the "
                 "exam's 'Mission Control / BCM lifecycle' question theme.",
        "tasks": [
            "cm-info                                 # confirm BCM head healthy",
            "cmsh -c 'device list'                   # 4 nodes pending",
            "./readmac.sh nodes.csv                  # batch-set MACs",
            "cm-wlm-setup                            # Slurm install wizard",
            "cm-kubernetes-setup                     # Kubernetes install wizard",
            "kubectl get pods --all-namespaces       # verify post-install",
            "helm repo add runai https://run-ai-charts.storage.googleapis.com",
            "helm install runai-cluster runai/runai-cluster -n runai",
            "kubectl get clusterpolicy               # GPU operator status",
            "runai cluster info                      # final smoke test",
        ],
    },
    {
        "title": "S30: NCP-AIO D1 — DOCA Services on the BlueField-3 DPU Arm",
        "story": "Stand up DOCA Flow on a BlueField-3 DPU and verify host-to-DPU "
                 "PCIe-VF communication. (Maps to the exam's DOCA Flow / DPU "
                 "questions.)",
        "tasks": [
            "lspci | grep -i Mellanox                # DPU detected",
            "mst status                              # Mellanox Software Tools",
            "mlxfwmanager                            # check ConnectX-7 / BF-3 FW",
            "mlxlink                                 # 200G LinkUp, FEC active",
            "ssh student@dgx02                       # jump to host with the DPU",
            "cmsh -c 'device list'",
            "# DOCA Flow handles packet processing offload for host-to-DPU traffic",
            "# Communication uses PCIe Virtual Functions (VFs) exposed by BF-3",
            "exit                                    # back to BCM head",
        ],
    },
    {
        "title": "S31: NCP-AIO D2 — MIG configuration on H100",
        "story": "Enable MIG and partition one H100 GPU into seven 1g.10gb "
                 "instances, the maximum count for an 80GB H100. Then list, "
                 "create a compute instance, and tear down. (Maps to the "
                 "exam's MIG profile questions.)",
        "tasks": [
            "nvidia-smi -mig 1                        # enable MIG mode",
            "nvidia-smi mig -lgip                     # list profiles",
            "nvidia-smi mig -cgi 19,19,19,19,19,19,19 # 7 × 1g.10gb",
            "nvidia-smi mig -lgi                      # list GPU instances",
            "nvidia-smi mig -cci                       # create compute instance",
            "nvidia-smi mig -lci                       # list compute instances",
            "nvidia-smi                                # see MIG layout",
            "# Tear down:",
            "nvidia-smi mig -dci                       # destroy compute inst",
            "nvidia-smi mig -dgi                       # destroy GPU inst",
            "nvidia-smi -mig 0                         # disable MIG",
        ],
    },
    {
        "title": "S32: NCP-AIO D2 — Run:ai quotas (department + project + over-quota)",
        "story": "Set up a department-level guarantee, a project-level quota, "
                 "and demonstrate over-quota borrowing when the cluster has "
                 "spare capacity.",
        "tasks": [
            "runai list departments",
            "runai create department research-org",
            "runai create project ml-research",
            "runai update project ml-research --gpu-quota 32 --gpu-guarantee 16",
            "runai list projects",
            "# Submit a small job inside the guarantee:",
            "runai submit small-train -p ml-research -g 8 -i pytorch:24.03 "
              "-- python train.py",
            "# Submit a larger job that uses over-quota borrowing:",
            "runai submit big-train -p ml-research -g 24 -i pytorch:24.03 "
              "-- python big_train.py",
            "runai top job",
            "runai list jobs -p ml-research",
        ],
    },
    {
        "title": "S33: NCP-AIO D2 — Slurm sacctmgr accounts, users, and QoS",
        "story": "Build the Slurm accounting hierarchy: cluster → account → "
                 "user → QoS. Then show fairshare and priority breakdown.",
        "tasks": [
            "sacctmgr list cluster",
            "sacctmgr list account",
            "sacctmgr list user",
            "sacctmgr list qos",
            "sshare                                    # raw vs effective usage",
            "sprio                                     # priority breakdown",
            "# Block a user from the GPU partition:",
            "scontrol update NodeName=node-007 State=DRAIN Reason=demo",
            "# Release a held job:",
            "scontrol show job 12471",
            "scancel 12473                             # cancel a pending job",
        ],
    },
    {
        "title": "S34: NCP-AIO D3 — NCCL multi-node test on Slurm",
        "story": "Submit a multi-node NCCL all-reduce benchmark and verify it "
                 "approaches NIC line-rate bandwidth. Maps to the exam's NCCL + "
                 "RoCEv2 / NVLink questions.",
        "tasks": [
            "module load slurm/slurm/23.02.8",
            "module load openmpi/gcc/64/4.1.5",
            "sinfo -N",
            "# Launch 4 nodes × 8 GPUs:",
            "srun -N 4 --gres=gpu:8 nccl all_reduce_perf",
            "# Or via sbatch with explicit GPU resource:",
            "sbatch --gres=gpu:8 --ntasks=4 nccl_test.sbatch",
            "squeue -u $USER",
            "# Inspect topology:",
            "nvidia-smi topo -m                        # NV18 = NVLink",
            "nvidia-smi nvlink --status                # all 18 links active",
            "nvidia-smi nvlink --errors                # error counters",
            "# RDMA fabric verification:",
            "ib_write_bw                               # ~196 Gb/s peak",
            "ib_read_bw",
        ],
    },
    {
        "title": "S35: NCP-AIO D3 — NGC: auth, container pull, model download",
        "story": "Authenticate to NGC, pull an inference container, and "
                 "download a pretrained model. Maps to the exam's NGC CLI and "
                 "'pull access denied' troubleshooting questions.",
        "tasks": [
            "ngc version",
            "ngc auth login --apikey $NGC_API_KEY      # CLI auth",
            "ngc config",
            "ngc orgs",
            "ngc registry image list                   # browse catalog",
            "ngc registry image info nvcr.io/nvidia/tritonserver:24.03",
            "# Docker-side auth for `docker pull` / kubectl image pulls:",
            "docker login nvcr.io -u '$oauthtoken' -p $NGC_API_KEY",
            "docker pull nvcr.io/nvidia/pytorch:24.03-py3",
            "# Download a model:",
            "ngc model list",
            "ngc model download-version nvidia/megatron-bert-345m:1.0",
        ],
    },
    {
        "title": "S36: NCP-AIO D3 — Kubernetes inference deployment with HPA",
        "story": "Deploy a Triton inference service as a stateless Kubernetes "
                 "Deployment, expose it via a LoadBalancer Service, and scale "
                 "with HPA on GPU utilization metrics from DCGM Exporter.",
        "tasks": [
            "kubectl create namespace inference",
            "kubectl apply -f triton-deployment.yaml",
            "kubectl get pods -n inference -o wide",
            "kubectl scale deployment triton --replicas=3 -n inference",
            "kubectl rollout status deployment/triton -n inference",
            "# Verify GPU device plugin / GPU operator:",
            "kubectl get clusterpolicy",
            "kubectl describe node node-001 | grep nvidia.com/gpu",
            "# Resource quota per namespace:",
            "kubectl create -f gpu-resourcequota.yaml",
            "# Confirm DCGM Exporter is feeding HPA:",
            "kubectl get pods -n gpu-operator | grep dcgm-exporter",
        ],
    },
    {
        "title": "S37: NCP-AIO D4 — Troubleshooting NCCL connectivity failures",
        "story": "A multi-node training job fails with "
                 "'NCCL WARN Connect to <ip> failed'. Triage from the network "
                 "interfaces all the way through to the RDMA stack.",
        "tasks": [
            "# 1. Check basic IP reachability between nodes:",
            "kubectl get nodes -o wide",
            "kubectl describe node node-002 | grep InternalIP",
            "# 2. Network interface in use by NCCL:",
            "cat /etc/nccl.conf                        # NCCL_SOCKET_IFNAME",
            "# 3. Switch port + RoCEv2 health:",
            "cmsh -c 'monitoring alerts'",
            "mlxlink                                   # FEC + BER",
            "# 4. RDMA datapath:",
            "ib_write_bw                               # peak ~196 Gb/s",
            "perftest                                  # available tools",
            "# 5. Container-side network (if running in pods):",
            "docker network inspect bridge",
            "# 6. Re-run NCCL with debug:",
            "srun -N 2 --gres=gpu:8 nccl all_reduce_perf",
        ],
    },
    {
        "title": "S38: NCP-AIO D4 — Hardware diagnostics drill (XID + PSU + thermal + fabric)",
        "story": "An H100 node is throwing XID 79 and reports 92°C junction "
                 "temp. Triage thermal, PSU, ECC, and NVLink in one sitting.",
        "tasks": [
            "# Thermal + ECC trend:",
            "nvidia-smi                                 # eyeball temps",
            "nvidia-smi dmon                            # 5-line live view",
            "nvidia-smi --query-gpu=temperature.gpu,power.draw --format=csv -l 2",
            "dmesg | grep xid                           # Xid 79 detected",
            "# Per-GPU diagnostic:",
            "dcgmi diag -r 3 -i 0",
            "dcgmi health",
            "nvidia-bug-report.sh                       # for NVIDIA support",
            "# PSU + BMC sensors:",
            "ipmitool sdr | grep -i 12v",
            "ipmitool sel                               # event log",
            "# NVLink + fabric manager:",
            "nvidia-smi nvlink --errors",
            "nvidia-smi topo -m",
            "systemctl status nvidia-fabricmanager",
            "cat /var/log/fabricmanager.log             # (sim returns OK)",
            "# Drain + reset:",
            "scontrol update NodeName=node-007 State=DRAIN Reason=XID79",
            "kubectl cordon node-007",
            "nvidia-smi --gpu-reset                     # if no active processes",
        ],
    },
    # =====================================================================
    # NCP-AIO Exam-style situational lab scenarios (S39-S44).
    # Each one frames a real exam-question scenario as a triage walk-through.
    # =====================================================================
    {
        "title": "S39: EXAM LAB — GPU pod stuck Pending after node drain",
        "story": "You drained node-007 to swap a PSU. An inference pod that "
                 "requests `nvidia.com/gpu: 1` is now stuck in Pending and won't "
                 "schedule on the remaining nodes. Triage from kubectl down to "
                 "the GPU-operator labels, then recover.",
        "tasks": [
            "# 1. Confirm the symptom:",
            "kubectl get pods -A -o wide | grep -i pending",
            "kubectl describe pod gpu-pod -n default     # look for FailedScheduling",
            "# 2. Why aren't the other GPU nodes accepting it?",
            "kubectl describe node node-001 | grep -i taint",
            "kubectl get nodes -o wide",
            "kubectl get nodes -L nvidia.com/gpu.product",
            "# 3. Check Run:ai scheduler if relevant:",
            "kubectl get pods -n runai",
            "journalctl -u runai-scheduler -n 50",
            "# 4. Resolve — uncordon node-007 once PSU is fixed:",
            "kubectl uncordon node-007",
            "scontrol update NodeName=node-007 State=RESUME",
            "# 5. Verify the pod is now scheduled and serving:",
            "kubectl get pods -A -o wide | grep gpu-pod",
            "kubectl rollout status deployment/triton -n inference",
            "kubectl logs -f gpu-pod -n default          # follow startup logs",
        ],
    },
    {
        "title": "S40: EXAM LAB — Slurm node DRAIN with reason 'NHC: failure'",
        "story": "`sinfo` shows node-007 in `drain*` with reason "
                 "'NHC: GPU ECC threshold'. Investigate via journalctl + "
                 "nvidia-smi -q, then resume.",
        "tasks": [
            "# 1. See which nodes are drained and why:",
            "sinfo                                        # state column",
            "scontrol show node node-007                 # full node detail",
            "# 2. Look at the failure source:",
            "journalctl -u slurmd -n 200                  # NHC: failure...",
            "journalctl -u slurmctld -n 100",
            "# 3. ECC + thermal verification on the GPUs:",
            "nvidia-smi -q -d ECC,TEMPERATURE",
            "nvidia-smi --query-gpu=ecc.errors.uncorrected.aggregate.total --format=csv",
            "dcgmi diag -r 2 -i 0",
            "# 4. Once cleared (or hardware swapped), restore the node:",
            "scontrol update NodeName=node-007 State=RESUME",
            "sinfo -N | grep node-007                    # back to idle",
        ],
    },
    {
        "title": "S41: EXAM LAB — Rolling update of Triton inference deployment",
        "story": "Push a new image (tritonserver:24.06) to the production "
                 "inference Deployment with zero downtime, monitor the rollout, "
                 "and roll back if startup fails.",
        "tasks": [
            "# 1. Authenticate and verify the new image is in NGC:",
            "ngc auth login --apikey $NGC_API_KEY",
            "ngc registry image info nvcr.io/nvidia/tritonserver:24.06",
            "# 2. Make sure pods can pull from nvcr.io:",
            "kubectl create secret docker-registry ngc-secret \\",
            "  --docker-server=nvcr.io \\",
            "  --docker-username='$oauthtoken' \\",
            "  --docker-password=$NGC_API_KEY -n inference",
            "# 3. Push the new image into the deployment:",
            "kubectl set image deployment/triton triton=nvcr.io/nvidia/tritonserver:24.06 -n inference",
            "kubectl rollout status deployment/triton -n inference",
            "kubectl rollout history deployment/triton -n inference",
            "kubectl logs -f deployment/triton -n inference",
            "# 4. If pods crashloop, roll back:",
            "kubectl rollout undo deployment/triton -n inference",
            "kubectl get pods -n inference -o wide",
        ],
    },
    {
        "title": "S42: EXAM LAB — BCM software-image rollout hung on 2 of 20 nodes",
        "story": "A cmsh `softwareimage commit` finished but two nodes are "
                 "still showing the old image. Use cmsh to inspect device "
                 "status, force a re-sync via imageupdate, and use SOL to "
                 "watch the boot if needed.",
        "tasks": [
            "cmsh -c 'device list'                        # which nodes report old image?",
            "cmsh -c 'device status'",
            "cmsh -c 'softwareimage list'                 # confirm new image present",
            "# Per-node sync force:",
            "imageupdate -n node010 -w",
            "imageupdate -n node011 -w",
            "# Watch the BMC serial console while the node reboots:",
            "ipmitool sol info",
            "ipmitool sol activate                        # type ~. to disconnect",
            "# Verify after reboot:",
            "rshell node010",
            "ls /                                          # new image content visible?",
            "exit",
            "cmsh -c 'device list'                        # all UP again",
        ],
    },
    {
        "title": "S43: EXAM LAB — Run:ai job preempted by training-priority class",
        "story": "A researcher's interactive Jupyter session was killed when a "
                 "high-priority training job started. Inspect Run:ai project "
                 "quotas, scheduler logs, and re-launch with adjusted quota.",
        "tasks": [
            "# 1. Confirm the preemption:",
            "runai list jobs -p ml-research",
            "runai describe job notebook-bob",
            "# 2. Check the project's guaranteed quota and current usage:",
            "runai list projects",
            "kubectl get pods -n runai",
            "journalctl -u runai-scheduler -n 50",
            "# 3. Adjust the quota or guarantee for the affected project:",
            "runai update project ml-research --gpu-quota 64 --gpu-guarantee 32",
            "# 4. Re-submit the interactive session:",
            "runai submit notebook-bob -p ml-research -g 1 --interactive \\",
            "  -i nvcr.io/nvidia/pytorch:24.03-py3",
            "runai resume notebook-bob",
            "runai list jobs -p ml-research",
        ],
    },
    {
        "title": "S44: EXAM LAB — Slurm cluster-level submit gating during maintenance",
        "story": "You need to put the Slurm cluster into a state where new "
                 "jobs CANNOT be submitted while you upgrade slurmctld, but "
                 "existing running jobs continue. After the upgrade, restore "
                 "submission and verify everything is healthy.",
        "tasks": [
            "# 1. Drain inbound submissions cluster-wide:",
            "scontrol update SubmitEnabled=no",
            "squeue                                       # running jobs still there",
            "# 2. Drain a single node for hardware work alongside:",
            "scontrol update NodeName=node-005 State=DRAIN Reason=upgrade",
            "sinfo -N | grep node-005",
            "# 3. Stop slurmctld for the upgrade:",
            "systemctl stop slurmctld",
            "journalctl -u slurmctld -n 50",
            "# 4. Bring everything back:",
            "systemctl start slurmctld",
            "systemctl status slurmctld",
            "scontrol update NodeName=node-005 State=RESUME",
            "scontrol update SubmitEnabled=yes",
            "sinfo                                         # all UP",
            "sbatch --array=0-9 --gres=gpu:1 train.sbatch  # array job test",
            "squeue",
        ],
    },
]


def show_scenarios() -> int:
    info("Practice scenarios:\n")
    for i, sc in enumerate(SCENARIOS, 1):
        info(f"  {i}. {sc['title']}")
    info("\nType 'scenario <n>' to start one (e.g. 'scenario 2').")
    return 0


def run_scenario(n: int) -> int:
    if n < 1 or n > len(SCENARIOS):
        return err(f"no such scenario {n}")
    sc = SCENARIOS[n - 1]
    info("\n" + "=" * 64)
    info(sc["title"]); info("-" * len(sc["title"]))
    info(textwrap.fill(sc["story"], width=70))
    info("\nSuggested commands to try (in order):")
    for t in sc["tasks"]:
        info(f"  {t}")
    info("=" * 64 + "\n")
    return 0


# ===========================================================================
# Dispatch
# ===========================================================================
COMMANDS = {
    # Run:ai
    "runai": cmd_runai,
    # Slurm core
    "sinfo": cmd_sinfo,
    "squeue": cmd_squeue,
    "sbatch": cmd_sbatch,
    "scancel": cmd_scancel,
    "scontrol": cmd_scontrol,
    "sacct": cmd_sacct,
    # Slurm extensions
    "srun": cmd_srun,
    "salloc": cmd_salloc,
    "sshare": cmd_sshare,
    "sprio": cmd_sprio,
    "seff": cmd_seff,
    "sreport": cmd_sreport,
    "sacctmgr": cmd_sacctmgr,
    # Kubernetes
    "kubectl": cmd_kubectl,
    "ksim":    cmd_kubectl,
    "k":       cmd_kubectl,
    "helm":    cmd_helm,
    # BCM
    "cmsh": cmd_cmsh,
    "cm-info": cmd_cm_info,
    "cm-version": cmd_cm_version,
    "cmha": cmd_cmha_status,            # 'cmha status' or just 'cmha'
    "mhcheck": cmd_mhcheck,
    "ndlist": cmd_ndlist,
    "node-installer-status": cmd_node_installer_status,
    # GPU / Linux tools
    "dcgmi": cmd_dcgmi_v2,                 # wraps cmd_dcgmi + extensions
    "nvidia-smi": cmd_nvidia_smi,
    "nvidia-bug-report.sh": cmd_nvidia_bug_report,
    "dmesg": cmd_dmesg,
    "ipmitool": cmd_ipmitool,
    "mlxlink": cmd_mlxlink,
    "mlxfwmanager": cmd_mlxfwmanager,
    "mst": cmd_mst,
    # Lab-mode commands (new)
    "module": cmd_module,
    "cm-wlm-setup": cmd_cm_wlm_setup,
    "cm-kubernetes-setup": cmd_cm_kubernetes_setup,
    "cat": cmd_cat,
    "su": cmd_su,
    "ssh": cmd_ssh,
    # Lab-2: Container Toolkit, Docker, PyTorch, Driver, BMC
    "nvidia-smi": cmd_nvidia_smi_v2,        # overrides earlier mapping
    "curl":       cmd_curl,
    "apt":        cmd_apt_get,
    "apt-get":    cmd_apt_get,
    "dpkg":       cmd_dpkg,
    "nvidia-ctk": cmd_nvidia_ctk,
    "systemctl":  cmd_systemctl,
    "lspci":      cmd_lspci,
    "uname":      cmd_uname,
    "mkdir":      cmd_mkdir,
    "docker":     cmd_docker,
    "python":     cmd_python,
    "python3":    cmd_python,
    "bmc":        cmd_bmc,
    # BCM Admin Lab (Practices 0-5)
    "./setup_lab.sh":   cmd_setup_lab,
    "setup_lab.sh":     cmd_setup_lab,
    "./readmac.sh":     cmd_readmac,
    "readmac.sh":       cmd_readmac,
    "cm-chroot-sw-img": cmd_cm_chroot_sw_img,
    "touch":            cmd_touch,
    "ls":               cmd_ls,
    "lsmod":            cmd_lsmod,
    "rshell":           cmd_rshell,
    "imageupdate":      cmd_imageupdate,
    "reboot":           cmd_reboot_node,
    "bcmuser":          cmd_bcmuser,
    "bcmgroup":         cmd_bcmgroup,
    "monitoring":       cmd_monitoring,
    # NCP-AIO exam additions
    "nvidia-smi":       cmd_nvidia_smi_v3,   # final overrides — MIG/nvlink/dmon
    "docker":           cmd_docker_v2,       # adds login + network inspect
    "ngc":              cmd_ngc,
    "ib_write_bw":      cmd_ib_write_bw,
    "ib_read_bw":       cmd_ib_read_bw,
    "ib_send_bw":       cmd_ib_send_bw,
    "perftest":         cmd_perftest,
    # Final exam-style overrides
    "nvidia-smi":       cmd_nvidia_smi_v4,    # adds -q -d ECC,TEMPERATURE,POWER,CLOCK
    "ipmitool":         cmd_ipmitool_v2,      # adds sol activate/info/deactivate
    "kubectl":          cmd_kubectl_v2,       # adds set/create-v2/annotate/patch/force delete
    "ksim":             cmd_kubectl_v2,
    "k":                cmd_kubectl_v2,
    "runai":            cmd_runai_v2,         # adds suspend/resume + list nodepools
    "sbatch":           cmd_sbatch_v2,        # tolerates --array/--constraint/etc.
    "kubeadm":          cmd_kubeadm,
    "journalctl":       cmd_journalctl,
    # Mock exam runner — 30 MCQ + 3 labs, weighted to NCP-AIOL blueprint
    "mock-exam":        cmd_mock_exam,
    "mock_exam":        cmd_mock_exam,
}


MANPAGES = {
    "kubectl": ("kubectl - control the Kubernetes cluster manager",
                "kubectl [command] [TYPE] [NAME] [flags]",
                "Common commands: get, describe, create, apply, delete, exec, logs, "
                "top, scale, rollout, set, drain, cordon, uncordon, label, taint, "
                "patch, annotate, port-forward, run, edit, explain.\n\n"
                "Examples:\n"
                "  kubectl get pods --all-namespaces\n"
                "  kubectl describe node node-001\n"
                "  kubectl set image deployment/triton triton=nvcr.io/.../tritonserver:24.06\n"
                "  kubectl rollout undo deployment/triton\n"
                "  kubectl create secret docker-registry ngc-secret --docker-server=nvcr.io ..."),
    "cmsh":    ("cmsh - Bright/NVIDIA Base Command Manager shell",
                "cmsh                                interactive shell\n"
                "cmsh -c \"<cmd> [; <cmd> ...]\"      one-shot mode",
                "Modes: device, category, softwareimage, kernelmodules, monitoring, "
                "wlm, user, group, main.\n\n"
                "Examples:\n"
                "  cmsh -c 'device list'\n"
                "  cmsh -c 'softwareimage clone default-image gpu; commit'\n"
                "  cmsh -c 'category use Lite; set softwareimage default-image-backup; commit'\n"
                "  cmsh -c 'device set node003 category Lite; commit'\n"
                "  cmsh -c 'monitoring alerts'"),
    "sbatch":  ("sbatch - submit a batch script to Slurm",
                "sbatch [OPTIONS] <script.sh>",
                "Common options: -N <n>, --gres=gpu:<n>, --ntasks=<n>, --time=HH:MM:SS, "
                "--partition=<p>, --array=<spec>, --exclusive, --constraint=<list>, "
                "--dependency=afterok:<jobid>, -J <name>.\n\n"
                "Examples:\n"
                "  sbatch -N 4 --gres=gpu:8 -t 02:00:00 train.sbatch\n"
                "  sbatch --array=0-99%10 array_task.sbatch\n"
                "  sbatch --dependency=afterok:12471 post.sbatch"),
    "srun":    ("srun - run a parallel job interactively under Slurm",
                "srun [OPTIONS] <cmd>",
                "Common options: -N <n>, --gres=gpu:<n>, --pty, --kill-on-bad-exit, "
                "--wait-all-nodes=1.\n\n"
                "Examples:\n"
                "  srun -N1 --gres=gpu:1 --pty bash\n"
                "  srun -N 4 --gres=gpu:8 nccl all_reduce_perf"),
    "scontrol":("scontrol - administer Slurm jobs, nodes, partitions",
                "scontrol [show|update|create|release|reconfigure|...] <args>",
                "Examples:\n"
                "  scontrol show node node-007\n"
                "  scontrol update NodeName=node-007 State=DRAIN Reason='PSU swap'\n"
                "  scontrol update NodeName=node-007 State=RESUME\n"
                "  scontrol update SubmitEnabled=no                  # cluster-wide gate\n"
                "  scontrol release 12471"),
    "sacctmgr":("sacctmgr - manage Slurm accounting database (slurmdbd)",
                "sacctmgr [list|add|modify|delete] {account|user|qos|cluster} [flags]",
                "Examples:\n"
                "  sacctmgr list account\n"
                "  sacctmgr list qos\n"
                "  sacctmgr modify account research set GrpTRES=gres/gpu=16\n"
                "  sacctmgr modify user alice set qos+=high"),
    "runai":   ("runai - Run:ai workload + project CLI",
                "runai [SUBCOMMAND] [flags]",
                "Sub-commands: list, submit, submit-mpi, describe, logs, delete, "
                "create, update, exec, suspend, resume, top, config, whoami, cluster, "
                "list workspaces, list nodepools.\n\n"
                "Examples:\n"
                "  runai list jobs -p ml-research\n"
                "  runai submit my-train -p ml-research -g 16 -i pytorch:24.03 -- python train.py\n"
                "  runai submit-mpi distrib -p ml-research -g 8 --workers 4 -- mpirun python train.py\n"
                "  runai update project ml-research --gpu-quota 64 --gpu-guarantee 32"),
    "nvidia-smi":("nvidia-smi - NVIDIA System Management Interface",
                  "nvidia-smi [OPTIONS]",
                  "Useful flags: -q [-d ECC,TEMPERATURE,POWER,CLOCK], "
                  "--query-gpu=<fields> --format=csv [-l <sec>], topo -m, "
                  "nvlink --status|--errors, dmon, pmon, --gpu-reset, "
                  "-mig 1|0, mig -lgip|-cgi|-cci|-lgi|-lci|-dgi|-dci.\n\n"
                  "Examples:\n"
                  "  nvidia-smi\n"
                  "  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 2\n"
                  "  nvidia-smi -q -d ECC,TEMPERATURE\n"
                  "  nvidia-smi -mig 1; nvidia-smi mig -cgi 19,19,19,19,19,19,19"),
    "dcgmi":   ("dcgmi - NVIDIA Data Center GPU Management CLI",
                "dcgmi [SUBCOMMAND] [flags]",
                "Sub-commands: discovery, group, config, diag, health, profile, stats.\n\n"
                "Examples:\n"
                "  dcgmi discovery -l\n"
                "  dcgmi group -c student_group\n"
                "  dcgmi group -g 9 -a 0,1,2,nvswitch:12\n"
                "  dcgmi config -g 9 --get\n"
                "  dcgmi diag -r 3 -i 0\n"
                "  dcgmi health"),
    "ipmitool":("ipmitool - utility for IPMI / BMC management",
                "ipmitool [-H bmc-host -U user -P pass] {sdr|sel|chassis|sol|...}",
                "Examples:\n"
                "  ipmitool sdr | grep -i 12v\n"
                "  ipmitool sel list | grep -i psu\n"
                "  ipmitool chassis status\n"
                "  ipmitool sol info\n"
                "  ipmitool sol activate"),
    "ngc":     ("ngc - NVIDIA NGC CLI for catalog access",
                "ngc [SUBCOMMAND] [flags]",
                "Sub-commands: auth, config, registry, model, user, orgs, version.\n\n"
                "Examples:\n"
                "  ngc auth login --apikey <KEY>\n"
                "  ngc registry image list\n"
                "  ngc model download-version nvidia/megatron-bert-345m:1.0"),
    "docker":  ("docker - manage Docker engine objects",
                "docker {pull|push|run|ps|images|login|exec|network|logs|...} ...",
                "Examples:\n"
                "  docker pull nvcr.io/nvidia/pytorch:24.03-py3\n"
                "  docker login nvcr.io -u '$oauthtoken' -p $NGC_API_KEY\n"
                "  docker run --gpus all --rm -it nvcr.io/nvidia/pytorch:24.03-py3"),
    "helm":    ("helm - the package manager for Kubernetes",
                "helm {list|install|upgrade|status|repo|history|rollback|uninstall} [flags]",
                "Examples:\n"
                "  helm list -A\n"
                "  helm install gpu-operator nvidia/gpu-operator -n gpu-operator\n"
                "  helm rollback gpu-operator 4"),
    "journalctl":("journalctl - query the systemd journal",
                  "journalctl [-u UNIT] [-f] [-n N] [--since=<duration>]",
                  "Examples:\n"
                  "  journalctl -u slurmctld -n 100\n"
                  "  journalctl -u kubelet -f\n"
                  "  journalctl -u nvidia-fabricmanager"),
    "kubeadm": ("kubeadm - the kubernetes cluster bootstrap tool",
                "kubeadm {init|token|join|reset|version|upgrade} [flags]",
                "Examples:\n"
                "  kubeadm init\n"
                "  kubeadm token create --print-join-command\n"
                "  kubeadm join 10.141.0.1:6443 --token <T> --discovery-token-ca-cert-hash sha256:<H>"),
    "mock-exam":("mock-exam - 120-minute NCP-AIOL practice exam (30 MCQ + 3 labs)",
                 "mock-exam [BANK] [start]",
                 "BANK is 1..6 — six independent question banks. Each bank uses the "
                 "official 31/23/23/23 domain weighting.\n\n"
                 "Examples:\n"
                 "  mock-exam              # interactive, bank 1\n"
                 "  mock-exam 4            # interactive, bank 4\n"
                 "  mock-exam 2 start      # bank 2, skip the 'start' confirmation"),
}


def _print_manpage(cmd: str) -> int:
    if cmd not in MANPAGES:
        info(f"No manual entry for {cmd}")
        info(f"(Try `help` for the simulator's command list.)")
        return 1
    name, syn, body = MANPAGES[cmd]
    print(f"\nNAME")
    print(f"    {name}")
    print(f"\nSYNOPSIS")
    for line in syn.splitlines():
        print(f"    {line}")
    print(f"\nDESCRIPTION")
    for line in body.splitlines():
        print(f"    {line}")
    print()
    return 0


def dispatch(tokens: list[str], state: dict) -> int:
    if not tokens:
        return 0
    # Strip leading `sudo` — students copy/paste lab commands verbatim
    if tokens[0] == "sudo":
        tokens = tokens[1:]
        if not tokens: return 0
    # OS-style help: `man <cmd>` or `<cmd> --help` (for known commands)
    if tokens[0] == "man" and len(tokens) >= 2:
        return _print_manpage(tokens[1])
    if (len(tokens) >= 2 and tokens[1] in ("--help", "-h")
            and tokens[0] in MANPAGES):
        return _print_manpage(tokens[0])
    # `exit` while inside a docker container leaves the container shell
    if tokens[0] in ("exit", "quit") and state.get("in_container"):
        state["in_container"] = False
        state["container_id"] = ""
        state["container_image"] = ""
        save_state(state)
        info("(exited container)")
        return 0
    # `exit` from cm-chroot-sw-img returns to the head node prompt
    if tokens[0] in ("exit", "quit") and state.get("chroot_active"):
        state["chroot_active"] = False
        state["chroot_image"] = ""
        save_state(state)
        info("exit")
        return 0
    # `exit` / `logout` from rshell <node> returns to cmsh device mode
    if tokens[0] in ("exit", "quit", "logout") and state.get("rshell_node"):
        state["rshell_node"] = ""
        save_state(state)
        info("logout")
        return 0
    head = tokens[0]
    if head in ("help", "?"):
        print(TOP_HELP); return 0
    if head == "scenarios":
        return show_scenarios()
    if head == "scenario":
        if len(tokens) < 2 or not tokens[1].isdigit():
            return show_scenarios()
        return run_scenario(int(tokens[1]))
    if head == "reset":
        STATE_FILE.unlink(missing_ok=True)
        info("State wiped. Restarting with a fresh cluster.")
        return 0
    if head == "state":
        info(json.dumps(state, indent=2)[:4000]); return 0
    if head in ("quit", "exit"):
        sys.exit(0)
    fn = COMMANDS.get(head)
    if not fn:
        return err(f"unknown command '{head}'.  type 'help'.")
    return fn(tokens[1:], state)


# ===========================================================================
# REPL
# ===========================================================================
BANNER = """\
============================================================
  GPU Cluster CLI Simulator   (NCP-AIIOL practice)
  16 nodes  ·  128 H100 GPUs  ·  Run:ai 2.18 · Slurm 23.11 · k8s 1.28
============================================================
Type 'help' for commands, 'scenarios' for practice tasks,
'reset' to start over, 'quit' to exit.
"""


def repl() -> None:
    state = load_state()
    print(BANNER)
    try:
        import readline  # nicer line editing if available
    except Exception:
        pass
    while True:
        try:
            # Lab-style prompt: user@host:~# (root) or :~$ (other)
            # Special prompt modes (in priority order):
            #   1. docker container:  root@<cid>:/workspace#
            #   2. cm-chroot-sw-img:  root@<image>:/#
            #   3. rshell <node>:     root@<node>:~#
            #   4. default:           user@host:~{# or $}{ (project)}
            if state.get("in_container"):
                cid = state.get("container_id", "abc123")[:12]
                raw = input(f"root@{cid}:/workspace# ").strip()
            elif state.get("chroot_active"):
                img = state.get("chroot_image", "default-image")
                raw = input(f"root@{img}:/# ").strip()
            elif state.get("rshell_node"):
                node = state["rshell_node"]
                raw = input(f"root@{node}:~# ").strip()
            else:
                user = state.get("user", "student")
                host = state.get("host", "bcm")
                sigil = "#" if state.get("is_root", False) else "$"
                proj_tag = f" ({state['current_project']})" if state.get("current_project") else ""
                raw = input(f"{user}@{host}:~{sigil}{proj_tag} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not raw:
            continue
        try:
            tokens = shlex.split(raw)
        except ValueError as e:
            err(f"parse error: {e}"); continue
        try:
            dispatch(tokens, state)
        except SystemExit:
            raise
        except Exception as e:
            err(f"internal error: {e}")
        # reload state because handlers persisted any changes
        state = load_state()


def main() -> None:
    if len(sys.argv) > 1:
        state = load_state()
        rc = dispatch(sys.argv[1:], state)
        sys.exit(rc)
    repl()


if __name__ == "__main__":
    main()
