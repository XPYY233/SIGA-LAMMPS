# SIGA-LAMMPS

A small, faithful engineering reproduction of **SIGA** (*Self-Evolving Coding-Agent
Adapters for Scientific Simulation*) for the **LAMMPS** molecular-dynamics
simulator, built on the open-source **DeepSeekHarness** coding agent.

> Reference paper: `docs/paper/2606.09774v1.pdf` —
> *SIGA: Self-Evolving Coding-Agent Adapters for Scientific Simulation*
> (Ho, Liu, Chen, Wang, Qin — UC San Diego).

## What this project is

SIGA is **not** a new scientific agent. It is a thin *Simulator-Interface
Grounding Adapter* bolted onto an existing coding agent, supplying the
simulator's executable contract — its vocabulary, structural constraints,
validation rules, and termination conditions — at exactly the three points where
an agent fails: before generation, during self-correction, and at termination.

This repository reproduces the paper's LAMMPS transfer study and extends it with
**real execution**: tasks are not merely authored but validated, uploaded to an
HPC cluster over SSH, submitted through SLURM, executed, and their results
returned to a web interface.

The four adapter components (the paper's M/R/X/S) are reproduced literally:

| | Component | Mechanism here |
|---|---|---|
| **M** | Procedural memory | An always-on LAMMPS primer injected as a harness *prompt section* — never a tool the agent must remember to call |
| **R** | Retrieval | A BM25-ranked LAMMPS knowledge base (docs / example scripts / command reference) behind an agent-callable `search_lammps` tool |
| **X** | Validator | A deterministic, rule-based `validate_lammps_input` tool the agent may call at will |
| **S** | Stop hook | An **externally enforced** termination gate: turn completion is blocked and the agent steered back to work while validation fails |

R uses BM25, not a dense vector store. ChromaDB was tried first and abandoned on
measurement: its ONNX model download stalled at roughly 20 KB/s, and the local
hashing vectoriser that replaced it collapsed ~431k features into 1024 dimensions,
about 421 per bucket. BM25 indexes the same 6,972 documents in 2.3 seconds and
needs no model. The measured consequence is recorded in `docs/TODO.md`: BM25
answers *command vocabulary* well but not *paraphrase*, which is the case the
paper frames R as existing for. R's measured contribution is therefore a lower
bound, and two strict `xfail` tests pin the gap.

X and S are deliberately distinct mechanisms, and the code keeps them apart:

- **X** is *agent-managed* validation — the agent chooses when to check its work.
- **S** is *externally enforced* validation — the harness refuses to let the turn
  end. It is not something the agent can skip, forget, or argue with.

## Scope

**In scope (v1):** five LAMMPS benchmark classes — Lennard-Jones melting, NVT
equilibration, mean-square-displacement diffusion, uniaxial tension, and
nanoindentation — plus a four-configuration ablation
(vanilla / +M / +M+R / +M+R+X+S) on identical tasks.

**Deliberately out of scope:** self-evolution, multi-agent systems, autonomous
hypothesis generation, materials inverse design, Bayesian optimization, ML
potential training, DFT workflows, automatic paper writing, and long-running
autonomous scientific campaigns.

## Layer separation

The system is six strictly separated layers. Nothing may reach across a boundary
it does not own:

1. **LLM** — DeepSeek models.
2. **Agent harness** — DeepSeekHarness: planning, context management, tool
   calling, file editing, the execution loop, error feedback.
3. **SIGA-style LAMMPS adapter** — M, R, X, S. The core of this project.
4. **Scientific simulator** — LAMMPS.
5. **HPC execution** — SSH/SFTP + SLURM, constrained to a configured remote
   workspace.
6. **Human interface** — a local web UI.

A guiding constraint: **adapter validation and LAMMPS runtime execution are
separate concerns.** The stop hook `S` never inspects whether LAMMPS ran; it
checks only the structural validity of the authored input. Runtime outcome is a
downstream fact reported to the human, not a termination condition.

## Running it

**Double-click `启动 SIGA-LAMMPS.command`.** That is the entry point. It checks
its preconditions, starts both services, waits until they actually answer, and
opens the console in your browser. Ctrl-C in that window stops everything.

The same thing from a terminal:

```bash
.venv/bin/python start.py                 # 默认配置 mrsx
.venv/bin/python start.py --replace       # 先清理上次残留的进程
.venv/bin/python start.py --configuration mr
```

| Port | What |
|---|---|
| **3080** | Your DeepSeekHarness GUI. **This project never touches it.** |
| **3081** | The harness instance this project starts, with the SIGA overlay |
| **8090** | The SIGA console — this is the page you work in |

First start takes 40–60 seconds; the harness has to boot and load the plugin tree.

### If it will not start

`start.py` fails before starting anything, and every message names the next
action rather than the symptom:

- **"端口被占用"** — a leftover process from a previous session. `--replace`
  clears it, but only for processes positively identified as this project's own:
  the harness is matched on *both* its entry script and this repo's overlay path,
  so another checkout's harness is never mistaken for ours. Anything else on the
  port is reported with its PID and left alone.
- **Already running** — double-clicking again does not error. It detects the live
  console, says so, and opens the browser.
- **Missing credentials** — set `DEEPSEEK_API_KEY`, or put it in
  `~/.dsh/.credentials.yaml`.

### A run that has finished

A completed cluster run leaves its outputs on the cluster. The console's
**超算上的文件** panel lists the remote directory without downloading it;
**查看** reads a file over the connection, **取回** copies one file back and
renders it if it is a trajectory. A trajectory can be hundreds of megabytes, so
nothing is transferred until you ask for it.

## Status

Built and verified end to end, module by module, in the order
`M → R → X → S → local LAMMPS → HPC executor → benchmark → web backend → web frontend`.

What is verified, rather than merely written:

- The four configurations are isolated as intended — vanilla 25 tools / no primer,
  +M 25 / primer, +M+R 33 tools, +M+R+X+S the same plus an active stop gate.
- One full ablation cell has run: `lj_melt × mrsx`, 310 s, 39 tool calls, 1 stop-gate
  interception, 51k in / 54k out tokens, compliance and sanity both 1.0.
- A real cluster round trip: submitted, polled, log read, results fetched back, a
  trajectory rendered, and observables computed (500 atoms; `fcc_atoms` 500 → 0 and
  first RDF peak 6.41 → 2.50 between the solid and liquid frames — the melting
  evidence the task is for).

The full 4×5 ablation matrix is built but deliberately **not** run: 20 cells at
roughly 100k tokens each is about 2M tokens of quota. `docs/TODO.md` records this
and the other open items, including the R paraphrase gap above.

Two honest limits: **level-4 checks never guess** — anything needing a structural
judgement, such as whether a crystal actually melted, is reported as requiring
human review rather than decided; and the web console is a demonstration
interface, not scientific validation.

## Safety

This project talks to a real HPC cluster. Accordingly: credentials live only in
an untracked `.env`; the agent is **never** granted an unrestricted remote shell;
every remote operation is confined beneath a configured workspace root; SLURM
resources are capped by configuration the agent cannot raise; every submission is
logged; and jobs can be cancelled by the human at any time.

## License

MIT — see `LICENSE`.
