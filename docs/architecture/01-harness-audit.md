# 01 — DeepSeekHarness capability audit for the SIGA-LAMMPS architecture

**Steps 1–5 of the agreed process.** Status: audit complete, MVP architecture
proposed, awaiting sign-off before module code is written.

Every claim below was verified by reading the harness source at
`/Users/fanjunran/deepseek-harness/` (commit checked out locally on 2026-09-22).
Claims that are inferences rather than verified facts are labelled **[inference]**.

---

## Step 1 — What DeepSeekHarness actually is

DSH is **not** a monolithic agent. It is a **Cordis plugin tree**: there is no
privileged core to patch, and every part of the product — the model adapter, the
tool registry, the session log, *and the agent loop itself* — is a replaceable
plugin. Extending it means **mounting a plugin beside the others**.

Source: `docs/architecture.md`.

Consequences that matter for us:

- A running `dsh` is composed from ordered **layers**: bundles → profile
  `cordis.patch.yml` → home `cordis.patch.yml` → `--patch` overlay.
- Registrations are **reversible effects** that unwind when their plugin unloads.
- Profiles live at `$DSH_HOME/profiles/<name>` (here `DSH_HOME=/Users/fanjunran/.dsh`);
  out-of-tree plugins install with `dsh plugin --profile <name> add <package>`.
- A **bundle** is an npm package whose manifest declares
  `"dsh": { "bundle": { "patch": "./cordis.patch.yml" } }`.

Key core packages and their context keys (`docs/architecture.md`):

| Package | Owns | `ctx` key |
|---|---|---|
| `core/session` | append-only `SessionEvent` log + store | `ctx.sessions` |
| `core/system-prompt` | prompt-section and tool-schema assembly | `ctx.systemPrompt` |
| `core/tools` | scoped tool registry + guarded execution pipeline | `ctx.tools` |
| `core/agent` | `Agent` interface, live registry, `agent/*` events | `ctx.agents` |
| `core/agent-loop` | the default driver | `ctx.agentLoop` |
| `llm/llm` | message/stream vocabulary + adapter seam | `ctx.llm` |

The event model is organised into three domains, and **picking the right domain
is the first decision in most changes**:

- **Session events** — durable facts appended to the log, broadcast via
  `session/event`. Use when the fact must survive a reload.
- **Agent events** (`agent/*`) — carry a live `Agent`: inbox, step, status,
  request, validation, continuation. Use to observe or intercept work in flight.
- **Capability events** (`fs/*`, `tools/*`, `telemetry/*`) — attach policy and
  adapters to a seam without importing the loop.

Two invariants constrain our design:

1. **"Model-visible means logged."** Anything reaching a model request must be
   reconstructable from the session log, and a runtime invariant asserts it. A new
   model-visible input therefore requires a new session event.
2. **A step is one model request plus the tools it calls; a turn is zero or more
   steps.** The turn closes only once nothing is owed.

---

## Step 2 — Capabilities DSH already provides (mapped to SIGA's needs)

The headline result: **all six interfaces SIGA needs already exist as documented,
first-class extension points.** Nothing requires forking the harness or
re-implementing the agent loop.

| SIGA / project need | DSH mechanism | Verified at |
|---|---|---|
| **M** — always-on procedural memory | `ctx.systemPrompt.section({ name, order, text })` | `packages/core/system-prompt/src/index.ts:381` |
| Dynamic per-turn context | `ctx.systemPrompt.context(...)` | `packages/core/system-prompt/src/index.ts:398` |
| **R**/**X** — agent-callable tools | `ctx.tools.register(defineTool({...}))` | `docs/user/develop/basic/tool.md` |
| Tools from an external process | MCP client bridge, tools appear as `mcp__<server>__<tool>` | `packages/mcp/mcp-client/README.md` |
| **S** — enforced termination gate | `agent/turn-stopping` (serial) + `agent.steer()` | `packages/core/agent/src/runtime-types.ts:278`, `:133` |
| S — working precedent | Claude Code `Stop` hook → `turn-stopping` + `steer()` | `packages/hooks/hooks-claude-code/src/index.ts:270-275` |
| Intercept tool calls | `tools/pre-execute`, `tools/post-execute` (waterfalls) | `docs/architecture.md` |
| Deny / require approval | `tools/pre-execute` → `PreToolDecision.deny` / `.ask` | `packages/hooks/hooks-claude-code/README.md` |
| Background work (HPC polling) | `ctx.jobs` + `job_list`/`job_output`/`job_kill` | `docs/architecture.md`, `docs/tool-catalog.md:38` |
| Durable custom state | extend `SessionEventMap`, render from log | `docs/cookbook/adding-a-conversation-node.md` |
| Per-tool UI cards (Areas B/C) | `ToolDefinition.presentCall` / `presentResult` | `docs/subsystems/tools.md` |
| Custom chat view / panels | `ctx.slots.register({ name, ... }, Component)` | `packages/extensions/cordis-client-runner/src/client/slot-catalog.ts` |
| **Ablation configs** | **agent presets** — one `agent.cordis.yml` per configuration | `packages/preset/agent-presets/README.md` |
| Headless benchmark runs | Python SDK, `dsh --profile headless`, JSON-RPC | `docs/user/guide/python-sdk.md`, `packages/sdk` |
| Token + tool-call accounting | session JSONL logs assembled requests and tool calls | `docs/user/guide/python-sdk.md` |

### The two findings that decide the architecture

**(a) `S` is a real, first-class mechanism — not something we must simulate.**

`docs/subsystems/core.md:994` documents `agent/turn-stopping` verbatim:

> The turn is about to close: the model owes no response (no live tool calls, no
> fresh steering). **Awaited before the boundary commits** — a listener that
> objects steers (`agent.steer(...)`) and the machine re-reads its inbox: fresh
> steering runs another step, none closes the turn.

That is precisely SIGA's Stop Hook: the harness *refuses to let the turn end*.
The Claude Code bridge already implements exactly this to support CC's `Stop`
hook, and its README states the bridge exists only as a compatibility path,
adding: *"anything bespoke should be a native plugin on the same extension
points."* So SIGA's `S` should be a native `agent/turn-stopping` listener.

This matters because it is what makes **X and S structurally different**, as
required:

- `X` = a *tool* the agent may choose to call (`ctx.tools.register`).
- `S` = a *harness-invoked* listener the agent cannot skip, forget, or disable.

They can share one validator implementation while remaining different
mechanisms — which is exactly how the paper describes its LAMMPS port: *"X
replaced xmllint with an agent-callable LAMMPS script validator applying the
same checks as the hook."*

**(b) Agent presets give us the four-configuration ablation for free, correctly.**

`packages/preset/agent-presets/README.md`: a preset is a directory holding one
`agent.cordis.yml`; the roster mounts it once per process under a standing scope,
and each session that names it joins by having its agent scope key parented to
that mount. The preset **decides the tool schemas and prompt sections the model
sees**, and — critically — *"the mount's listeners are admitted for every agent
parented under it while a sibling preset's stay deaf."*

So four presets (`vanilla`, `m`, `mr`, `mrsx`) yield four configurations that are
provably isolated: identical LLM, identical harness, identical inference
settings, differing **only** in the adapter components mounted. The selected
preset is recorded as a durable `agent-preset/selected` session event, so every
result is reproducible after the fact. This satisfies experiment requirement §7
and constraint §10 by construction rather than by discipline.

---

## Step 3 — What is genuinely missing

Remarkably little. The gaps are packaging and plumbing, not capability.

| # | Gap | Severity | Note |
|---|---|---|---|
| G1 | **No LAMMPS domain content** — no primer, no corpus, no rules | Expected | This *is* the project. Not a harness gap. |
| G2 | **No SSH/SFTP remote-execution seam** | Medium | Nothing in the repo does SSH. The capability graph (`docs/architecture.md`) says to add a `ctx.fs` provider to move filesystem access remotely — powerful but a large build. A narrow, audited tool set is safer. |
| G3 | **Custom session events break resume by default** | **High (trap)** | `session-persistence/src/coordinator.ts:1061` refuses to interpret a log containing an event type absent from `KNOWN_SESSION_EVENT_TYPES` **unless `event.ignorable === true`**. Our own durable events must be marked ignorable, or sessions become unresumable. |
| G4 | **No prompt token budget** | Medium | `packages/llm/token-meter/src/estimate.ts:13` uses `CHARS_PER_TOKEN = 4`; compaction replays `header.system` verbatim, and there is no per-plugin prompt cap. An oversized M is billed on **every** request. M must be deliberately small and measured. |
| G5 | **No out-of-tree plugin template** | Low | Plugin authoring is documented (`docs/user/develop/basic/`) but there is no scaffolded starter. We write ours. |
| G6 | **No benchmark/evaluation harness for adapter ablations** | Low | The Python SDK runs tasks headlessly; the task matrix, metrics, and evaluator are ours to build. |

Worth stating plainly: **no gap requires modifying `packages/`.** We add plugins;
we do not patch the harness.

---

## Step 4 — Minimal retrofit plan

**Principle: one small TypeScript plugin for harness integration, a Python core
for the science.** Rationale for the split is in Step 5.

The harness-side retrofit is four registrations and one listener — no new
packages inside the harness, no fork:

| # | Retrofit | Mechanism | Rough size |
|---|---|---|---|
| 1 | Mount **M** as a preset-scoped prompt section | `ctx.systemPrompt.section({ name: 'siga-lammps-memory', order: 150, text })` | ~20 lines |
| 2 | Register **R** + **X** + HPC tools | `ctx.tools.register(defineTool({...}))` × ~7 | ~250 lines |
| 3 | Register **S** | `ctx.on('agent/turn-stopping', ...)` → validate → on failure `agent.steer(createUserMessage({ content, source }))` | ~60 lines |
| 4 | Mark our durable events `ignorable: true` | envelope contract, addresses G3 | ~5 lines |
| 5 | Compose four presets | `agent.cordis.yml` per config | 4 small YAML files |
| 6 | Three UI surfaces | `ctx.slots.register(...)` + `presentCall`/`presentResult` | ~200 lines |

Nothing in this list changes harness behaviour for anyone else: registrations are
reversible, and preset scoping means a sibling preset's listeners stay deaf.

### A safety requirement I want to flag before building anything

My own session **is** the live `web` profile — `DSH_SESSION_ID=session-0546c619…`
is running under `dsh web` on `127.0.0.1:3080` with `DSH_HOME=/Users/fanjunran/.dsh`.
If SIGA were mounted into that profile, its Stop Hook would become a termination
condition on *this* conversation and the tool registry would gain HPC tools in
every unrelated session.

**Therefore SIGA must be installed into its own profile** (e.g. `siga`), run as a
separate server on a different port, leaving the everyday `web` profile
untouched. This costs nothing and removes a genuine self-modification hazard.

---

## Step 5 — Proposed MVP architecture and interfaces

### The layer boundary, expressed as a file layout

`packages` map onto the six required layers with no mixing:

```
SIGA-LAMMPS/
├── adapter/                      # LAYER 3 — the SIGA adapter core (Python)
│   ├── memory/lammps_memory.md   #   M: the primer TEXT (data, not code)
│   ├── retrieval/                #   R: ChromaDB index + build scripts
│   ├── validator/                #   X: deterministic rules — no LLM anywhere
│   └── cli.py                    #   one JSON entry point: MCP / tools / S / benchmark
├── hpc/                          # LAYER 5 — SSH/SFTP + SLURM (Python, paramiko)
│   ├── ssh_client.py             #   confined to the configured workspace root
│   ├── slurm.py                  #   sbatch/squeue/sacct/scancel
│   └── templates/                #   job script templates
├── harness/siga-plugin/          # LAYER 2 integration — TypeScript, deliberately thin
│   ├── src/memory.ts             #   M: one prompt section
│   ├── src/validator-stop.ts     #   S: the turn-stopping gate
│   ├── src/tools.ts              #   R/X/HPC tool registrations
│   └── src/client/               #   LAYER 6 — slots for Areas A/B/C
├── benchmark/                    # tasks / ground_truth / evaluator
├── config/                       # .env + config.yaml loading, resource ceilings
└── docs/
```

**Why the logic lives in Python while the integration is TypeScript** — this is
the one real architectural judgement, so the reasoning is explicit:

- **ChromaDB is Python-native** (the paper used ChromaDB over three collections:
  example scripts, documentation RST, and command syntax). There is no
  maintained TypeScript ChromaDB client.
- **SSH/SFTP + SLURM + `.env` + resource ceilings** are natural in Python
  (`paramiko`), and `~/.ssh/config` certificate auth is handled by the system.
- **The deterministic validator must be shared.** The paper's X and S apply *the
  same checks*. One Python implementation exposed through one CLI — called by the
  tool *and* by the stop hook — makes that identity structural rather than a
  promise. It is also directly reusable by the benchmark evaluator.
- **TS is unavoidable anyway** because M (prompt section) and S (turn listener)
  are in-process harness concepts with no out-of-process equivalent.

### Interfaces

**The single adapter entry point** — every consumer goes through this, so X's
tool and S's gate can never drift:

```bash
python -m adapter.cli validate --workspace <dir> --task <benchmark-id> --json
# -> {"valid": false, "errors": [...], "warnings": [...], "suggestions": [...]}

python -m adapter.cli search --query "fix nvt" --k 5 --json
# -> [{"source": ..., "snippet": ..., "command": ..., "metadata": {...}}]
```

**Agent-facing tools** (names exactly as specified):

| Tool | Layer | Notes |
|---|---|---|
| `search_lammps(query, k?)` | R | ChromaDB over docs + examples + command reference |
| `validate_lammps_input(workspace, task?)` | X | deterministic rules; structured result |
| `hpc_upload_workspace(workspace)` | HPC | SFTP under the configured root only |
| `hpc_submit_job(workspace, resources)` | HPC | `sbatch`; resources clamped to config ceilings |
| `hpc_job_status(job_id)` | HPC | `squeue` + `sacct` |
| `hpc_read_log(job_id, which)` | HPC | `log.lammps` / `slurm-*.out` |
| `hpc_cancel_job(job_id)` | HPC | `scancel` |

There is deliberately **no** `hpc_run_shell`. The agent cannot express an
arbitrary remote command; every remote operation is a fixed function confined to
the configured workspace root.

**`S`'s contract**, stated precisely because it is the component most easily
implemented wrongly:

```
agent/turn-stopping
  → locate the session workspace
  → run the SAME validator X exposes
  → if valid:            return, turn closes
  → if invalid:          agent.steer(structured errors + repair instruction)
                         → machine re-reads inbox → another step runs
```

`S` inspects **only** the structural validity of the authored input. It never
inspects whether LAMMPS ran — keeping adapter validation and runtime execution
separate as required by §5.

### Two things I verified that will bite us if ignored

1. **`G3`** — our durable session events must set `ignorable: true`, or the
   harness refuses to reload any session we touched (`coordinator.ts:1061`).
2. **`G4`** — M's prompt text is billed on every request and is never trimmed.
   M must stay compact and be measured, not assumed cheap.

---

## Confirmed environment facts

| Fact | Value |
|---|---|
| Harness checkout | `/Users/fanjunran/deepseek-harness` (read-only for us) |
| Harness home / live profile | `/Users/fanjunran/.dsh`; `web` on `127.0.0.1:3080` — **this session's own profile** |
| Local LAMMPS | `/Users/fanjunran/.local/bin/lmp` — 22 Jul 2025 Update 4, Open MPI 5.0.9 ✅ |
| HPC alias | `sy_hl_login` → `sylogin.hpc.sjtu.edu.cn`, user `fjr200630-1`, cert auth |
| **HPC access right now** | ❌ **certificate expired 2026-09-17** (today: 2026-09-22) — needs renewal |

The HPC finding does not block M/R/X/S or local LAMMPS work. It does mean the HPC
layer must ship a preflight that distinguishes *auth failure* from other failures
and reports it actionably, rather than retrying blindly. Since `BatchMode=yes`
failed closed rather than hanging, that behaviour is already correct by default.

---

## Open decision

The one choice that changes everything downstream is **how the agent-facing
tools reach the harness**: registered natively in TypeScript (exact tool names,
native UI cards, one more subprocess hop) versus exposed over MCP from the Python
adapter (clean process boundary, but tools surface as `mcp__lammps__search_lammps`).
This is put to the user before any module code is written.
