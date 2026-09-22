# 02 — MVP architecture (Option B + MCP), for sign-off

Companion to `01-harness-audit.md`. This document is the contract we build
against; module code starts only after it is agreed.

## Decisions taken

| # | Decision | Rationale |
|---|---|---|
| D1 | **Agent-facing tools are exposed over MCP** from a Python stdio server | Clean process boundary between layer 2 and layer 3; adapter stays language-neutral and reusable by other harnesses |
| D2 | **The Web UI is a separate Python app** (FastAPI + a small frontend) driving the harness over its HTTP RPC + SSE | Full control of Areas A/B/C; harness stays a dependency, not a host |
| D3 | **A thin TypeScript plugin remains** for `M` and `S` only | Prompt sections and turn listeners are in-process harness concepts with no out-of-process equivalent (see audit §2) |
| D4 | **All adapter logic lives in Python**, behind one CLI | ChromaDB/paramiko are Python; and `X` and `S` must provably run *the same checks*, which one shared implementation guarantees |
| D5 | **SIGA installs into its own profile** (`siga`), never the live `web` profile | This session runs on `web`; mounting a stop hook there would gate the user's own conversations |
| D6 | **Four agent presets** encode the ablation | `packages/preset/agent-presets` isolates tool schemas and prompt sections per session by construction (audit §2b) |

## System diagram

```
        ┌──────────────────────────────────────────────────────────────┐
 L6     │  SIGA Web App  (Python / FastAPI + frontend)                 │
        │                                                              │
        │  Area A  request + file upload ──┐                           │
        │  Area B  agent activity  ◄───────┼── GET /api/events.mux (SSE)│
        │  Area C  job + logs      ◄───────┘                           │
        └───────────────┬──────────────────────────────────────────────┘
                        │  POST /api/session.create { agentPreset, cwd }
                        │  POST /api/session.prompt
                        │  POST /api/session.cancel
                        ▼
        ┌──────────────────────────────────────────────────────────────┐
 L2     │  DeepSeekHarness   `dsh web --profile siga`  (its own port)  │
        │                                                              │
        │  presets:  vanilla │ m │ mr │ mrsx     ← the ablation        │
        │                                                              │
        │  ┌─ siga-plugin (TS, ~150 lines) ─────────────────────────┐  │
        │  │   M : ctx.systemPrompt.section()                       │  │
        │  │   S : ctx.on('agent/turn-stopping') → steer()          │  │
        │  └────────────────────────────────────────────────────────┘  │
        │  ┌─ dsh-mcp-client ──► stdio ──► Python MCP server ──────┐  │
        │  └────────────────────────────────────────────────────────┘  │
        └───────────────┬──────────────────────────────────────────────┘
                        │  MCP (JSON-RPC over stdio)
        ┌───────────────▼──────────────────────────────────────────────┐
 L3     │  SIGA LAMMPS Adapter  (Python)                               │
        │   memory/      R: retrieval/       X: validator/             │
        │   lammps_memory.md  chroma index      deterministic rules    │
        │                                                              │
        │   adapter/cli.py   ← the single shared entry point           │
        │   adapter/mcp_server.py ← the 7 agent-facing tools           │
        └───────────────┬──────────────────────────────────────────────┘
                        │  same CLI, second entry point
        ┌───────────────▼──────────────────────────────────────────────┐
        │  L4 LAMMPS (local, validation-time)                          │
        │  L5 HPC: paramiko SSH/SFTP + SLURM, workspace-confined       │
        └──────────────────────────────────────────────────────────────┘
```

The important structural property: **`X` and `S` reach the validator through the
same `adapter/cli.py`.** X is a tool the agent calls; S is a listener the harness
calls. Same rules, different authority — which is precisely the distinction §3 of
the brief requires, and what the paper describes for its LAMMPS port.

## File layout

```
SIGA-LAMMPS/
├── adapter/                          # LAYER 3 — Python, no harness imports
│   ├── memory/lammps_memory.md       #   M: the primer TEXT
│   ├── retrieval/                    #   R: corpus build + ChromaDB query
│   │   ├── build_index.py
│   │   ├── corpora.py                #    3 collections: examples, docs, syntax
│   │   └── search.py
│   ├── validator/                    #   X: deterministic rules, zero LLM
│   │   ├── rules/                    #    one module per check family
│   │   ├── lammps_parse.py           #    script → ordered command list
│   │   └── engine.py                 #    → structured result
│   ├── tasks/                        #   5 benchmark specs + required-command tables
│   ├── cli.py                        #   THE shared entry point (JSON in/out)
│   └── mcp_server.py                 #   the 7 tools over stdio MCP
├── hpc/                              # LAYER 5 — Python
│   ├── ssh_client.py                 #   paramiko; workspace-confined
│   ├── slurm.py                      #   sbatch/squeue/sacct/scancel
│   ├── preflight.py                  #   auth vs. other failures (cert expiry!)
│   └── templates/job.slurm.j2
├── harness/                          # LAYER 2 integration — TypeScript
│   ├── siga-plugin/                  #   M prompt section + S stop gate
│   └── presets/                      #   vanilla | m | mr | mrsx  (agent.cordis.yml)
├── web/                              # LAYER 6 — Python
│   ├── backend/                      #   FastAPI: RPC proxy + SSE fan-out + runs
│   └── frontend/
├── benchmark/
│   ├── tasks/                        #   frozen natural-language specs
│   ├── ground_truth/                 #   reference LAMMPS inputs
│   └── evaluator/                    #   stage-1 structural + metric aggregation
├── config/
│   ├── config.yaml                   #   non-secret settings, resource ceilings
│   └── loader.py                     #   config.yaml + .env → typed config
├── workspace/                        #   generated runs (git-ignored)
├── tests/
└── docs/
```

## Layer specifications

### M — Procedural memory

`adapter/memory/lammps_memory.md`, injected by `harness/siga-plugin` as:

```ts
ctx.systemPrompt.section({ name: 'siga-lammps-memory', order: 150, text })
```

Always-on, never a tool. Content per the brief: command ordering, units
conventions, `atom_style`, lattice/region/create_box/create_atoms, read_data,
pair_style/pair_coeff, neighbor, velocity, fix nve/nvt/npt, compute, thermo,
timestep, run, plus the ordering errors, incompatible fixes, and pitfalls.

**Budget is a design constraint, not an afterthought** (audit G4: no token cap,
billed every request). Target ≤ 4k characters for v1; the primer is a *measured*
artifact with a size assertion in `tests/`. The paper's LAMMPS primer covered
ordering, unit conventions, and ten task-specific pitfalls — that is the bar.

Per-task pitfalls (the paper's list, kept as the seed): LJ lattice density
semantics, region-before-create_box ordering, unfix before switching
integrators, the SLLOD Couette pattern, MSD compute syntax.

### R — Retrieval

Three ChromaDB collections mirroring the paper: **example scripts**,
**documentation RST**, **command syntax extracted from the source tree**.

`build_index.py` ingests from a corpus root; the raw corpus is **not committed**
(it is large, licensed upstream, and fetchable) — `data/raw/` is git-ignored and
a documented `make corpus` step populates it. The index is rebuildable.

Tool: `search_lammps(query, k=5)` → `source`, `snippet`, `command`/`example`,
`metadata`. The agent never walks the documentation tree.

### X — Validator

Deterministic rules only; **no LLM in the validation path.** Checks the brief
requires, in order: required input files exist → command ordering → units →
atom_style → structure initialization → pair_style/pair_coeff → potential file
reference → fix/ensemble → timestep → run → task-specific required commands →
file references resolve → obvious conflicts.

Result contract (exactly as specified):

```json
{ "valid": false, "errors": [...], "warnings": [...], "suggestions": [...] }
```

Every finding carries a stable `code`, the offending `line`, and a machine-
readable `evidence`, so the benchmark can bucket failures by category and S can
format a repair instruction without re-parsing prose.

### S — Stop hook

`harness/siga-plugin`:

```
ctx.on('agent/turn-stopping', async ({ agent, turn }) => {
  const workspace = <resolve from session cwd>
  const result = await runValidatorCli(workspace, task)   // SAME cli as X
  if (result.valid) return
  if (blocksThisTurn >= MAX_BLOCKS) return   // ← see below
  agent.steer(createUserMessage({ content: repairInstruction(result),
                                  source: { kind: 'plugin', plugin: 'siga' } }))
})
```

**Two hazards the audit found, and how S handles them:**

1. **There is no continuation cap on `agent/turn-stopping`.** A validator that
   never passes would loop the agent indefinitely, burning tokens. S therefore
   keeps a per-`(session, turn)` block counter, caps it at `MAX_BLOCKS`
   (configurable, default 3), and on exhaustion **allows the turn to close** while
   recording a structured `siga/stop-gate-exhausted` fact. This is both a cost
   guard and a benchmark signal — "how often did S fail to converge" is a real
   result, not a hidden failure.
2. **Custom durable events must be `ignorable: true`** or the harness refuses to
   reload any session we touched (`session-persistence/coordinator.ts:1061`).

S inspects **only** structural validity. It never looks at whether LAMMPS ran.

### HPC execution

Tools (all over MCP, no `hpc_run_shell`): `hpc_upload_workspace`,
`hpc_submit_job`, `hpc_job_status`, `hpc_read_log`, `hpc_cancel_job`.

- Auth via the user's own `~/.ssh/config` alias + agent. Credentials are never
  read, stored, or transmitted by us.
- Every remote path is resolved and asserted to be under `SIGA_HPC_WORKSPACE`
  **after** normalization (guarding `..` and symlink escapes).
- SLURM resources are clamped to `config.yaml` ceilings; the agent may request
  less, never more.
- Every submission is appended to an audit log with the exact rendered script.
- `preflight.py` distinguishes **auth failure** from other failures. This is not
  hypothetical: the `sy_hl_login` certificate expired 2026-09-17, so a clear
  "certificate expired, renew it" message is the difference between a usable
  system and a mysterious hang.

### Benchmark

Frozen tasks (5 classes from the brief → the paper's 9-task set is the model):
LJ melting, NVT equilibration, MSD diffusion, uniaxial tension, nanoindentation.
Each has: NL specification, required files, expected key commands, reference
input, validation criteria.

Metrics per the brief: input completeness, deterministic validation pass rate,
LAMMPS initialization success, LAMMPS execution success, parameter correctness,
runtime, agent tool calls, LLM token usage, failure category.

Two evaluation stages as in the paper: **stage 1 structural** (deterministic,
11–15 per-task criteria) and **stage 2 value correctness**. The paper used an
LLM judge for stage 2; we additionally have something the paper did not — real
execution — so "did LAMMPS actually initialize and run" becomes a hard,
non-LLM signal.

The four configs are the four presets, driven through the same headless path with
identical tasks, model, and inference settings.

### Web app

FastAPI backend that is a **thin, honest proxy** plus run management:

| Endpoint | Purpose |
|---|---|
| `POST /api/runs` | create a run: allocate workspace, write uploads, create harness session with `agentPreset` + `cwd` |
| `POST /api/runs/{id}/message` | forward natural language to `session.prompt` (Area A: iterative modification) |
| `GET  /api/runs/{id}/events` | SSE, re-emitted from `/api/events.mux` (Area B) |
| `GET  /api/runs/{id}/job` | parsed job state + thermo + file list (Area C) |
| `POST /api/runs/{id}/cancel` | `session.cancel` and/or `scancel` |

**Area B shows tool calls, statuses, short action summaries, and validator
feedback — never hidden chain-of-thought.** We forward only what the session log
already makes durable: `tool/call`, `tool/result`, `agent/status`, and our own
`siga/validation` and `siga/stop-gate` facts. Nothing is synthesized.

## Build order (each step tested before the next)

| Step | Module | Test gate |
|---|---|---|
| 1 | `config/` + `.env` loader | secrets never logged; ceilings clamp correctly |
| 2 | `adapter/validator/` + `cli.py` | hand-written correct and broken LAMMPS scripts; table-driven expectations |
| 3 | `adapter/memory/` | size budget assertion; pitfalls present |
| 4 | `adapter/retrieval/` | corpus build; known-answer queries |
| 5 | `adapter/mcp_server.py` | all 7 tools callable over stdio; schema-valid |
| 6 | `harness/siga-plugin` + presets | S actually blocks a turn on a broken script, and releases on a fixed one |
| 7 | Local LAMMPS smoke test | a generated script runs to completion with `lmp` |
| 8 | `hpc/` | preflight + dry-run submission (needs renewed cert) |
| 9 | `benchmark/` | one task × 4 configs end-to-end |
| 10 | `web/` | run created, events streamed, job displayed |
| 11 | Integration | the brief's full flow, with the iterative-modification loop |

## Open questions for you

- **HPC certificate**: needs renewal before step 8; steps 1–7 are unblocked.
- **Corpus**: do you want me to fetch the LAMMPS source tree + docs locally for R
  (large, git-ignored), or point `config.yaml` at a path you already have?
- **Stage-2 judging**: the paper used an LLM judge for value correctness. Keep
  that, or rely on deterministic criteria + real execution results only?
