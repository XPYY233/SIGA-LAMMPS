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
// The primer is read from disk verbatim — it is a document, not code.
ctx.systemPrompt.section({ name: 'siga-lammps-memory', order: 10, text: primer })

// Run-specific values are interpolated into it, so the primer stays static.
ctx.systemPrompt.variable('siga_task', () => activeTask(scope))
```

M stays a **static, reviewable Markdown file** — which is what a cheatsheet is —
and the few values that vary per run arrive through `{{siga_task}}`-style
variables registered with `systemPrompt.variable(name, provider)`
(`core/system-prompt/src/index.ts:446`). Names must match `[a-z][a-z0-9_]*`,
scoped values shadow globals, and rendering a section whose variable resolves to
`undefined` **fails loud** — so a missing variable surfaces immediately rather
than silently shipping a primer with a hole in it.

`order` follows the harness convention documented at
`core/system-prompt/src/index.ts:53-75`: `-100` is harness identity, `0` the
deployment persona, and `100–199` tool guidance. M is simulator grounding —
domain vocabulary and structural rules — not tool guidance, so it belongs just
after the persona and *before* the tool band. (An earlier draft of this document
used `150`, which would have slotted the primer into the tool-guidance band.)

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

Three collections mirroring the paper: **example scripts** (1,089), **documentation
RST** (4,977 chunks), **command syntax extracted from the source tree** (906).
Total 6,972 documents, built from an official LAMMPS checkout in ~2 seconds.

**Backend deviation — this is BM25, not ChromaDB.** The paper used ChromaDB with
dense embeddings, and this document originally specified the same. Two measured
findings forced the change, and the second is a limitation worth stating plainly
rather than burying:

1. ChromaDB's default embedder downloads an **83 MB** ONNX archive on first use.
   Here that transfer sustains ~20 KB/s and stalls; worse, ChromaDB re-downloads
   whenever the archive fails its SHA256 check, so every embedding call paid the
   failed download again. Indexing never completed a single batch, and the
   symptom reads as "embedding is slow" rather than "the model is absent".
2. A local hashing vectorizer was tried and rejected on measurement: ~431,000
   distinct features hashed into 1024 dimensions is **~421 features per bucket**.
   Rare discriminative terms drown in collisions — the literal phrase "mean
   square displacement" ranked behind a timing utility. No dimension that fits in
   memory repairs it, because the required width is the size of the feature space.

BM25 is the right structure for lexical matching: a sparse inverted index with
exact term statistics and no hashing, so there are no collisions to lose signal
in. It needs no network, is deterministic, and is auditable end to end.

**Measured quality, which bounds what R may claim in the ablation:**

| Query | Outcome |
|---|---|
| `compute msd`, `fix deform`, `fix nvt/sllod`, `pair_style lj/cut` | top-1 correct |
| `spherical indenter pressing into a surface` | `fix_indent` top-1 |
| `stretch a box along one axis` | `fix_deform` **absent** from top-5 |
| `thermostat to hold a constant temperature` | `nvt` **absent** from top-5 |

So R serves command-vocabulary queries well — an agent that knows it needs
`fix deform` and wants the syntax is served. It does **not** serve paraphrase, and
the paper frames R as existing precisely *"for when the agent does not know the
right simulator terms to search for"*. **This is a real shortfall against the
paper, not a neutral implementation choice**, and it must be reported as such in
any result: R's measured contribution here is a lower bound on the paper's R.
Both misses are recorded as strict `xfail`s in `tests/test_retrieval.py`, so if
either starts passing the suite fails and the limitation gets revisited.

Closing the gap needs dense embeddings from a model that can actually be fetched.

Corpus provenance is recorded per run. The corpus is an official LAMMPS checkout
(`github.com/lammps/lammps`); it is **not committed** (large, upstream-licensed),
so `data/` is git-ignored and the index is rebuilt with
`python -m adapter.cli build-index`.

Tool: `search_lammps(query, k=5)` → `source`, `collection`, `snippet`, `command`,
`score`, `metadata`. The agent never walks the documentation tree. The backend
is recorded with the index, because a retrieval result is not comparable across
backends.

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
   (configurable, default 3), and on exhaustion **allows the turn to close**.
   This is both a cost guard and a benchmark signal — "how often did S fail to
   converge" is a real result, not a hidden failure.
2. **The counter must live in plugin-owned memory, not the session log.** A
   `siga/stop-gate-exhausted` event would be the natural way to record it, and it
   is exactly what we may not do: an out-of-repo event type makes the session
   unresumable (audit G3). This costs us nothing — the counter is per-turn state,
   and turns do not survive a restart by definition. The *count* is reported to
   the benchmark layer, which owns its own storage.

S inspects **only** structural validity. It never looks at whether LAMMPS ran.

### How state is carried, given we may not add event types

This constraint touches S, Area B, and the benchmark, so it is stated once here.

| What we need | How it is carried | Event type it lands as |
|---|---|---|
| Repair instruction to the agent | `agent.steer(...)` | `user/message` |
| Durable plugin context | `agent.inject({ content, source: { kind: 'plugin', plugin: 'siga' } })` | `user/message` |
| Validator results | the `validate_lammps_input` tool's own return value | `tool/call` + `tool/result` |
| S block count within a turn | plugin-owned in-memory state | none (never durable) |
| S block counts across runs | the benchmark layer's own store | none (outside the harness) |

The third row is the pleasant consequence: **the validator's output is already
durable as a tool call**, so a custom `siga/validation` event would have been
redundant even if it were permitted. The agent's own validation history is
recoverable from the log without us inventing anything.

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
  hypothetical: certificates on this cluster are short-lived (the current one
  runs 2026-09-22 → 2026-10-22), so expiry is a routine monthly event rather than
  an exceptional fault. A clear "certificate expired, renew it" message is the
  difference between a usable system and a mysterious failure.
- **Every operation goes through `sy_hl_login`.** It is the only alias this
  certificate authenticates as, and it is the only submission target. The HPC
  layer hard-codes nothing else and adds no fallback host.

### Benchmark

Frozen tasks (5 classes from the brief → the paper's 9-task set is the model):
LJ melting, NVT equilibration, MSD diffusion, uniaxial tension, nanoindentation.
Each has an NL specification, required files, expected key commands, a reference
input, and validation criteria.

**Evaluation follows the mandated four-level model** (`docs/design-principles.md`
§7), *not* the paper's two-stage metric. The paper scored structural similarity
plus an LLM judge; the principles forbid treating reference similarity as the
primary metric, because functionally equivalent LAMMPS scripts legitimately
differ in IDs, naming, ordering, and numerical settings.

| Level | Question | Mechanism |
|---|---|---|
| 1 Static | Is the workspace structurally sound? | the X validator, unchanged — one implementation, no second copy for scoring |
| 2 Runtime | Does it actually run? | real LAMMPS + SLURM; non-LLM and hard |
| 3 Task compliance | Is it running the task that was asked for? | deterministic per-task checkers (ensemble, target T, duration, observable computed **and** output, structure/potential actually used) |
| 4 Physical sanity | Are results physically plausible? | bounded automatic checks; anything else is marked `human_review_required` |

**Level 4 is not a correctness claim.** An LLM may not assert physical validity
without a reliable basis; where the check is unreliable the run is marked for
human review rather than scored.

**Level 1 doubles as X.** Scoring does not reimplement the validator: the
benchmark calls the same `adapter/cli.py validate` that the agent's tool and the
S gate call. Divergence between "what the agent was told" and "what it was
scored on" would make the whole ablation uninterpretable.

**Ground truth is a reference, not the answer.** It is used for task
construction, expected commands and settings, compliance checking, debugging, and
controlled comparison. A script that differs textually but is runtime-successful,
task-compliant, and physically sane is **not** a failure.

**Failure taxonomy** (`docs/design-principles.md` §9), recorded per run rather
than collapsed into one score:

```
knowledge_error      syntax_error          command_order_error
missing_command      missing_file          bad_parameter
wrong_units          wrong_ensemble        invalid_reference
premature_termination runtime_error         physical_instability
task_noncompliance   unknown
```

**Audit record** per run (§11): task_id, run_id, model and settings, adapter
configuration, M version/hash, retrieval results, tool calls, validator results,
termination attempts, generated files, SLURM script, job ID, LAMMPS log, runtime
status, evaluation results (all four levels), token usage, wall-clock time,
failure category. Any failure must answer *what failed, and at which layer* —
not `score = 0`.

**Controlled comparison invariants** (§10): same model and version, same harness,
same specification, same benchmark, same available files, same HPC environment,
same resource limits, same inference settings, **same maximum trajectory
budget**, same evaluation pipeline. The four presets make the adapter
configuration the only intended variable.

Reported as *capability* versus *reliability floor*, not as a single aggregate:
the core question is whether DeepSeekHarness already authors LAMMPS capably while
failing unreliably, and which failure modes each of M, R, X, S reduces.

### Web app

FastAPI backend that is a **thin, honest proxy** plus run management:

| Endpoint | Purpose |
|---|---|
| `POST /api/runs` | create a run: allocate workspace, write uploads, create harness session with `agentPreset` + `cwd` |
| `POST /api/runs/{id}/message` | forward natural language to `session.prompt` (Area A: iterative modification) |
| `GET  /api/runs/{id}/events` | SSE, re-emitted from `/api/events.mux` (Area B) |
| `GET  /api/runs/{id}/job` | parsed job state + thermo + file list (Area C) |
| `POST /api/runs/{id}/cancel` | `session.cancel` and/or `scancel` |

### ⚠️ Our web app is the trust boundary — it must bind loopback

Verified against `packages/client/connection/src/index.ts:88-112`, which is worth
quoting because it settles a question we would otherwise have guessed at:

> CHOOSING one is not pinned, and `agentPreset.list` is not either. [...] The
> deeper reason is that the capability is not the preset's to grant: the
> deployment's own default already carries `bash` and the filesystem tools, so
> **any caller that may start a session at all can already run commands as this
> process.** Pinning the switch would be a fence beside an open gate.

Two consequences, one enabling and one requiring care:

- **Enabling:** `session.create` accepts `agentPreset` and is *not* loopback-pinned
  (`PRIVILEGED_METHODS` at `:90-107` lists only `agentPreset.read/copy/remove/
  openDocument`, `host.pickDirectory/openPath`, `settings.*`, `credentials.*`,
  `llm.discoverModels`). So D6's preset-driven ablation is reachable exactly as
  planned, with no privileged call.
- **Requiring care:** the `/api` fence is explicitly **a DNS-rebinding fence, not
  an authentication layer**. Anyone who can reach our FastAPI app effectively
  holds the ability to start sessions that run commands as this user. Therefore
  **the SIGA web app binds `127.0.0.1` only.** Exposing it on a LAN address
  without adding real authentication would hand out that capability, and the
  harness has already told us it will not stop us.

### Transport: the event stream is WebSocket, not SSE

**This section previously claimed the opposite. Recording the correction because
the mistake is instructive:** I read `packages/host/apiproxy/src/fetch/handler.ts`
in isolation — it genuinely does answer these paths with `sseResponse(...)` and
`content-type: text/event-stream` — without checking *which carrier mounts it for
a browser*. Reading a file is not the same as reading the wiring.

There are **two carriers**, and only one is reachable over the network:

| Carrier | Used by | Event transport |
|---|---|---|
| `toFetchHandler` + `InProcessApiClient` | in-process clients | SSE — never touches the network |
| `WebSocketDownlinks` via HTTP upgrade | the browser | **WebSocket** |

The decisive code is `packages/client/connection/src/index.ts:150-155`, which
intercepts a plain GET **before** it can reach the fetch handler:

```ts
if (request.method === 'GET' && (pathname === MUX_EVENTS_PATH || pathname === HOST_EVENTS_PATH)) {
  return new Response('upgrade required', {
    status: 426,
    headers: { connection: 'Upgrade', upgrade: 'websocket' },
  })
}
```

The upgrade routes are registered at `:193-194`, and `api-path.ts:11,14` names
them outright: *"Browser mux-frame **WebSocket** pathname"*. Frames arrive as
JSON `{ type: 'server-request', rpcId, method, payload }`
(`websocket-downlink.ts:16-24`).

**Practical effect — the split is per route, not per server:**

- `POST /api/<method>` RPC — plain HTTP, `application/json` (else 415). Unchanged,
  and `session.create`/`session.prompt` work exactly as this document assumes.
- `GET /api/events.mux` — **WebSocket upgrade only**; a plain GET returns 426.

So Area B needs a WebSocket client (`websockets` in Python), not an `httpx`
streaming GET. Building it the other way would have produced a 426 that reads
like an auth or routing failure rather than a protocol mismatch.

**Area B shows tool calls, statuses, short action summaries, and validator
feedback — never hidden chain-of-thought.** We forward only what the session log
already makes durable: `tool/call`, `tool/result`, `agent/status`, and
`user/message`. Nothing is synthesized, and nothing new is invented — validator
feedback reaches the UI through the `validate_lammps_input` tool call that
produced it, and through the `user/message` that S steers back to the agent when
it blocks a turn. Both are events the harness already writes.

## Implementation constraints confirmed against source

Concrete API facts the plugin must respect. Every entry was verified against the
harness source; several of them **correct** earlier assumptions in this project
rather than merely confirming them.

| Topic | Fact | Consequence for us |
|---|---|---|
| Exports | Harness plugins use **named exports, no default export** | `export const name`, `export const inject`, `export function apply(ctx)` |
| Lifecycle | **`ctx.on('ready')` and `ctx.on('dispose')` do not exist** | Cleanup must go through `ctx.effect(fn, label?)` disposers, which Cordis unwinds on unload |
| Plugin config | `Config` is a schemastery schema (`z.object({...})`) | Our plugin declares its validator path, block budget, and workspace root this way |
| Prompt service | `ctx.systemPrompt` exposes **five** effect-based methods: `section()` `:381`, `context()` `:398`, `suppressRuntimeContext()` `:418`, `tools()` `:430`, `variable()` `:446` | M uses `section()` + `variable()`. We do **not** need `tools()` — MCP already contributes our schemas |
| Prompt sections | No token budget or truncation inside `assemble()`; size pressure is `packages/compaction`'s job | M's size assertion in `tests/` is the only guard. Confirmed, not assumed. |
| Tool args | `tools/pre-execute` **cannot rewrite arguments** — only `allow`/`deny`/`ask`. Args are excluded from rewriting *because they are already logged and presented* | Good for us: what the agent asked for is exactly what the audit log shows |
| Tool results | `tools/post-execute` **can** replace content/value or `block` with feedback | Available if we ever need to redact a remote path from a result |
| Turn control | `ToolRunContext` exposes `deferContext()` and `concludeTurn()` | `concludeTurn()` is the *inverse* of S — a tool ending the turn early. S deliberately does not use it. |
| Approval | `ctx.approval.request({agent, toolName, callId?, reason?, signal?})`; policy `ask` \| `never`; fails **closed** to deny without an answerer | The right gate for HPC submission — a human click, not a silent upload |
| Tool visibility | `ctx.tools.restrict({ allow?, deny? })` — scoped contexts only, throws on a global one | Keeps HPC tools out of presets that should not have them |
| Whole-prompt transform | `system-prompt/assemble` waterfall exists | Not needed for M; `section()` is sufficient and less invasive |
| Durable events | 18 *in-repo* packages extend `SessionEventMap` by declaration merging — but `known-event-types.ts`:15-17 excludes out-of-repo plugins **by construction**, and `Session.append` offers no `ignorable` escape hatch | **We may not invent a `siga/*` event type.** All durable and model-visible state rides known types — see below |
| Remote execution | **`ssh`/`sftp`/`scp`/`slurm` do not exist anywhere in the harness** | Confirms layer 5 belongs in Python. Total grep over `packages/ apps/ docs/ examples/ python/` finds only `SSH_CONNECTION` env detection in a directory picker |
| Network sandbox | **None.** `sandbox/src/index.ts:25`: *"Network and process visibility are outside this vocabulary."* No egress allowlist | HPC confinement cannot come from an OS sandbox. It comes from having no `hpc_run_shell` at all — fixed-function tools are the enforcement |
| Env / config | `.env` loading is narrow: only `<cwd>/.env` and `$DSH_HOME/.env`, read-only, and it feeds the **credentials** domain | The harness will *not* read our `.env` for its own config. The MCP server receives settings through the `mcp-client` row's `env:` block (`!!js process.env.X`), or reads `.env` itself |
| Out-of-tree client UI | The `clientBundle()` tsdown preset is **not published**; a bundle-purity gate rejects cross-plugin value imports | **Validates Option B.** Building the three Areas as in-GUI client plugins would fight unpublished packaging |
| Tool card vocabulary | `ToolCallView`/`ToolResultView` are **closed unions** (`presentation.ts:41`, `:141`) — `card: 'job'` is not addable out-of-tree | **Validates Option B.** Area C's job card needs our own UI; the harness can only render `generic`/`terminal`/`diff`/`search`/`read`/`web` |
| Host HTTP routes | `ctx.webServer.register({ kind: 'exact'\|'prefix', path, handler })` (`webserver/src/index.ts:59`) is the sanctioned out-of-tree route | Available if a host-half route is ever needed; our FastAPI app is independent of it |
| S firing boundary | The gate runs only at the **natural stop boundary**: `agent-loop/src/agent.ts:294-299` gates on `turnEnds`, and a step that emitted tool calls leaves `turnEnds === null`, skipping it | Correct semantics for a Stop hook — S judges a *finished* attempt, not a mid-loop state. S never sees a half-written script. |
| S loop guard | `hooks-claude-code` hard-codes `stop_hook_active: false` and its README records `TODO(stop-loop-guard)`; `core/agent-loop/README.md:134` states *"No built-in turn budget"* | Confirms S **must** self-limit. Our `MAX_BLOCKS` budget is not defensive padding; without it an ungated validator loops forever. |
| Plugin-owned state | `ctx.storageDomain` + `defineDomain` give a plugin a KV domain, independent of the session log | The durable option for S's block counter if it ever needs to survive a restart. v1 keeps it in memory — per-turn state does not outlive a turn. |
| Benchmark driver | TS SDK is full-featured: `DeepSeekHarness.run()` with `onNotification` (`packages/sdk/client/src/api.ts:22`, `:98`). The **Python SDK is sessions-only** — no subagent or plugin API | Decides D2's benchmark path: drive headless via the harness CLI, or the TS SDK. Not the Python SDK. |
| **SDK cannot gate a turn** | Both SDKs expose only `initialize` / `session/prompt` / `shutdown` plus four notifications. They can **observe** a run but cannot **block** a turn's completion | **This is why D3 is forced, not merely convenient.** `S` has no out-of-process form: an external driver can detect a bad script only *after* the turn ended, which is post-hoc re-prompting, not the paper's termination condition. S must be an in-process plugin. |
| Cancellation, per path | HTTP RPC exposes `session.cancel` (and `session.attachment`, `session.history`). The SDK's stdio protocol exposes **no** cancellation and no client→server notifications | Option B's web app speaks HTTP RPC, so the brief's "the user can cancel a job" is satisfiable. Had we driven everything through the SDK, it would not be. |
| Headless CLI flags | `dsh --profile headless "task"` prints the final assistant text and exits 0/1. There is **no `--json` / `--print` flag** | Benchmark metrics come from the **session log**, not from CLI stdout. That is already how the evaluator is designed. |

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
