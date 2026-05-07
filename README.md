# GPU CLI Simulator — NCP-AIOL Exam Prep

A hands-on study tool for the **NVIDIA-Certified Professional: AI
Operations** exam (NCP-AIOL). Type real Slurm, Kubernetes, BCM,
Run:ai, NVIDIA, and Mellanox commands and get realistic output back —
against a persistent simulated cluster of 16 nodes × 8 H100 GPUs.

> Built because no good NCP-AIOL prep tool existed. Sharing in case it
> helps other candidates. Single Python file, no PyPI dependencies
> beyond stdlib, runs on macOS / Linux / WSL.

5.png

---

## Quick start

```bash
git clone https://github.com/<username>/gpu-cli-simulator.git
cd gpu-cli-simulator
python3 gpu_cli_simulator.py
```

You'll land in a REPL prompt:

```
============================================================
  GPU Cluster CLI Simulator   (NCP-AIOL practice)
  16 nodes  ·  128 H100 GPUs  ·  Run:ai 2.18 · Slurm 23.11 · k8s 1.28
============================================================
Type 'help' for commands, 'scenarios' for practice tasks,
'mock-exam' for a 120-minute timed exam, 'quit' to exit.
(ml-research) >
```

---

## What's in it

### 50+ simulated commands across the full NCP-AIOL stack

```
BCM        cmsh (interactive + -c form), cm-info, cm-version, cmha,
           mhcheck, ndlist, cm-wlm-setup, cm-kubernetes-setup,
           cm-chroot-sw-img, softwareimage, kernelmodules,
           category, device, monitoring, wlm

Slurm      sinfo, squeue, sbatch, srun, salloc, scancel, scontrol,
           sacct, sshare, sprio, seff, sreport, sacctmgr

Kubernetes kubectl get/describe/exec/logs/top/cordon/drain/uncordon,
           scale, rollout (status/restart/history/undo), set image,
           run, create (namespace/secret/deployment/...), label, taint,
           edit, explain, port-forward, patch, annotate, helm,
           kubeadm

Run:ai     list/submit/submit-mpi/describe/logs/delete/create/update,
           workspaces, departments, projects, suspend, resume,
           top job, list nodepools, exec, port-forward

NVIDIA     nvidia-smi (full incl. -mig, mig -lgip/-cgi/-cci, nvlink
           --status/--errors, dmon, pmon, --gpu-reset, -q -d),
           dcgmi (diag, health, profile, group lifecycle, config,
           discovery -l), nvidia-bug-report.sh, nvidia-ctk

Linux      docker (pull, run, login, network), ngc (auth, registry,
           model), apt-get/dpkg, journalctl, systemctl, ipmitool
           (sdr, sel, sol, chassis), mlxlink, mlxfwmanager, mst,
           lspci, uname, dmesg, lsmod, ssh, su

RDMA       ib_write_bw, ib_read_bw, ib_send_bw, perftest
```

### 44 scenario walk-throughs

Step-by-step practice tasks framed as exam-style problems. Examples:

- **S31** Configure MIG on H100 as 7×1g.10gb instances
- **S32** Run:ai quotas — department + project + over-quota borrowing
- **S38** Hardware diagnostics drill: XID 79 + PSU + thermal + fabric
- **S41** Rolling update of Triton inference deployment with kubectl set image
- **S42** BCM software-image rollout hung on 2 of 20 nodes
- **S44** Slurm cluster-level submit gating during slurmctld upgrade

```bash
(ml-research) > scenarios          # list all 44
(ml-research) > scenario 41        # walk through one
```

### 6 timed mock exams

Each one mirrors the official NCP-AIOL blueprint:

- 30 multiple-choice questions + 3 hands-on lab exercises
- 120-minute timer
- Domain weighting **31% Installation · 23% Administration · 23% Workload Management · 23% Troubleshooting**
- Six independent question banks (240+ questions, 36 lab tasks)
- Immediate per-question feedback (✓/✗ + explanation)
- Per-step lab feedback (which expected commands matched, which missed)
- Length-balanced, position-shuffled options to remove bias

```bash
(ml-research) > mock-exam              # bank 1
(ml-research) > mock-exam 4            # bank 4
(ml-research) > mock-exam 2 start      # skip the confirmation
```

### Real-shape OS-style help

```bash
(ml-research) > man kubectl
(ml-research) > kubectl --help
(ml-research) > nvidia-smi --help
```

You get a proper NAME / SYNOPSIS / DESCRIPTION layout for the 16 most
commonly tested commands. Help works inside the lab phase of
mock-exams too — without counting against your grade.

[ADD SCREENSHOT HERE — `man kubectl` output]

### Persistent simulated cluster state

Your changes survive across REPL sessions. Drain a node, the next
session shows it drained. Submit a Run:ai job, it appears in `runai list jobs` next time. Configure MIG, the layout persists. State
lives in `gpu_cli_state.json` next to the script.

```bash
(ml-research) > reset            # wipe state and reseed
```

---

## Why I built this

I'm prepping for NCP-AIOL. The official study guide is good, but
nothing on the market lets you actually *type* `cmsh -c "device list"`
or `nvidia-smi mig -cgi 19,19,19,19,19,19,19` and see realistic
output. The exam itself has a 3-lab hands-on component that's hard to
prep for without a real cluster — which most candidates don't have.

So I built one. The simulator's 44 scenarios + 6 mock exams cover the
official blueprint domains with the same weighting. By taking
mock-exam through banks 1-6 and walking the matching scenarios, you'll
build the muscle memory you need.

If you find a missing command or scenario, open an issue or PR. I
update it whenever I notice a gap.

---

## Architecture

```
gpu_cli_simulator.py        ─── single Python file, ~8000 lines, stdlib only
   ├─ Command dispatch table         (50+ commands)
   ├─ Persistent state (JSON file)   (16 nodes × 8 H100 cluster)
   ├─ Per-tool handlers              (cmsh, kubectl, runai, nvidia-smi, ...)
   ├─ Interactive cmsh sub-REPL      (with [bcm->mode]% prompt nesting)
   ├─ Scenario walkthroughs          (44 numbered guides)
   ├─ Mock exam runner               (6 banks, 120-min timer, lab grading)
   ├─ Man-page system                (NAME/SYNOPSIS/DESCRIPTION format)
   └─ State migration on load        (backwards-compat across versions)

gpu_cli_state.json          ─── auto-generated, persistent cluster state
```

No external services. No telemetry. No PyPI install required for the
core simulator. (`mock-exam` uses only stdlib too.)

---

## Roadmap

- More NCCL multi-node debugging scenarios
- Run:ai inference workload state machine
- DCGM Exporter Prometheus output simulation
- Multi-cluster federation simulation
- Web-UI mode (FastAPI bridge)
- DOCA on BlueField-3 DPU walkthrough

---

## Author

Built by **Baljeet Bhasin** while preparing for NCP-AIOL. NCP-AII and
NCA-AIIO certified, ex-Oracle OCI Compute TPM, transitioning into
AI infrastructure roles. Connect on LinkedIn:
[https://www.linkedin.com/in/baljeetbhasin](https://www.linkedin.com/in/baljeet-bhasin-pmp-968959/)

If you're hiring for AI infrastructure roles (TPM, Solutions Architect,
Customer Engineer, AI Operations Lead) at NVIDIA, a GPU cloud, or a
federal SI — get in touch. [bbhasin@gmail.com](mailto:bbhasin@gmail.com).

---

## License

MIT — see `LICENSE`. Use it, fork it, build on it, share it. If it
helps you pass NCP-AIOL, drop me a note. That'd make my week.
