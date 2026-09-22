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
| **R** | Retrieval | ChromaDB-backed LAMMPS knowledge base (docs / example scripts / command reference) behind an agent-callable `search_lammps` tool |
| **X** | Validator | A deterministic, rule-based `validate_lammps_input` tool the agent may call at will |
| **S** | Stop hook | An **externally enforced** termination gate: turn completion is blocked and the agent steered back to work while validation fails |

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

## Status

🚧 **Pre-implementation.** The harness capability audit is complete; the MVP
architecture is being settled before module code is written. Implementation
proceeds incrementally, one tested module at a time:

`M → R → X → S → local LAMMPS test → HPC executor → benchmark → web backend → web frontend → integration test`

## Safety

This project talks to a real HPC cluster. Accordingly: credentials live only in
an untracked `.env`; the agent is **never** granted an unrestricted remote shell;
every remote operation is confined beneath a configured workspace root; SLURM
resources are capped by configuration the agent cannot raise; every submission is
logged; and jobs can be cancelled by the human at any time.

## License

MIT — see `LICENSE`.
