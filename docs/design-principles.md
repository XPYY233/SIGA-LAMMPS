# Design principles (binding)

Supplementary working specification for SIGA-LAMMPS. **These constrain design,
code, benchmark design, and Web work.** Where an implementation conflicts with
them, the conflict is raised rather than silently implemented.

Companion to `architecture/01-harness-audit.md` and `architecture/02-mvp-architecture.md`.

---

## 1. What SIGA is understood to be here

Not a new base model, not a new inference algorithm, not an autonomous scientist.

A mature coding agent already plans, navigates files, edits, calls tools, runs
commands, reads errors, and repairs. Its problem is not incapacity but the
absence of a **simulator-specific executable contract**, which shows up in long
trajectories as vocabulary errors, missing components, bad parameters,
non-validation, and premature termination.

SIGA-style adaptation grounds the agent at three interfaces **without rewriting
the agent loop**:

```
Base harness      H0 = (context, tools, termination)

M  → context        Procedural memory
R  → tools          Retrieval
X  → tools          Validator
S  → termination    Stop hook
```

If the harness already exposes the extension point, use it. Do not redesign the
core loop to implement SIGA.

## 2. M and R have different jobs — and this is non-negotiable

> Keep high-frequency procedural knowledge always on.
> Retrieve detailed domain knowledge on demand.

**M = always-on procedural knowledge.** Only:

- high-frequency rules and general command ordering
- unit conventions
- common simulation patterns
- common fatal mistakes and pitfalls
- tool usage guidance
- a completion checklist

**M is bounded: roughly 500–1000 tokens.** It is not extended indefinitely.

**Anything long, specific, narrow, or reliably retrievable goes to R.** The
tests enforce this, including a `MUST_NOT_APPEAR_IN_M` list, because the cheapest
way to break the boundary is to append "just one more useful thing".

> The goal is not to make the prompt longer.
> The goal is to ground the agent at the right interface.

## 3. Context budget is a first-class design constraint

Do not assume an unlimited window. **Record and control**: system prompt tokens,
M tokens, user task tokens, retrieved chunks, file contents, tool outputs,
trajectory history.

Retrieval returns **top-k chunks with small snippets and source metadata** —
never an entire manual, an entire example directory, or large unrelated files.

**Workspace is persistent external state; context is limited working state.**
Large LAMMPS input/data/log files live in the workspace and are read through
tools on demand, rather than residing permanently in context. Prefer any
compression, summarisation, tool-result truncation, or file-reference mechanism
the harness already provides over building one.

## 4. Prompt engineering and harness enforcement are not equivalent

These are different mechanisms, and only one is a guarantee:

| | |
|---|---|
| A. Prompt: *"Please validate before finishing"* | a **behavioural instruction** |
| B. Harness: validation failure **rejects termination** | **control-flow enforcement** |

Behaviour that must be guaranteed cannot rest on a prompt alone.

**S must be externally enforced termination control.** It may not be implemented
as a reminder in the system prompt.

**X and S stay distinct:**

| | X | S |
|---|---|---|
| when | in-trajectory | at termination |
| who drives | agent-managed | externally enforced |
| optional? | yes | no |

## 5. Do not over-complicate the adapter to look like the paper

Permitted and expected: simple Markdown procedural memory, simple vector
retrieval, a deterministic validator, a simple termination hook.

Optimise for **minimal, auditable, reproducible, testable** — not for
sophistication. Do not introduce multi-agent systems, planner hierarchies,
fine-tuning, RL, graph workflow engines, self-evolution, or large memory
frameworks unless the MVP has demonstrated the need.

The research question is narrow:

> Can a lightweight simulator-interface grounding layer improve the reliability
> of DeepSeekHarness for LAMMPS?

## 6. TreeSim is not our metric

The paper's TreeSim is a **reference-based structural similarity** measure.
TreeSim = 1 means high agreement with a reference deck under its tree
normalisation and matching rules. It is not physical correctness, not runtime
success, and not uniqueness of solution.

For LAMMPS, functionally equivalent scripts legitimately differ in variable
names, `fix`/`compute` IDs, file organisation, command sequence, and numerical
settings.

**Do not use string similarity to the ground-truth script as the primary
metric.** Never reward looking like the answer.

> Do not evaluate whether the generated script looks like the reference.
> Evaluate whether the agent actually accomplishes the requested simulation.

## 7. Four-level evaluation

**Level 1 — Static / structural.** Required files exist, referenced files exist,
required commands present, command ordering, units, `atom_style`,
`pair_style`/`pair_coeff` consistency, ensemble configuration, obvious conflicts,
requested outputs configured.
→ *Is this workspace structurally sound?*

**Level 2 — Runtime.** Real LAMMPS/HPC invocation: initialisation success,
`run 0` where applicable, no immediate fatal error, SLURM submission success, job
completion, no NaN or obvious crash, expected output files produced.
→ *Does it actually run?*

**Level 3 — Task compliance.** Does the run satisfy the specification — ensemble
actually NVT, target temperature actually 300 K, duration actually as requested,
observable actually computed **and** output, requested structure/potential
actually used.
→ *Is it running the task the user asked for?*

**Level 4 — Physical sanity**, only to the extent it can be judged
automatically: temperature behaviour, energy stability, atomic explosion,
impossible values, observable availability, ensemble behaviour.
→ *Are the results physically sane?*

**Level 4 is not a claim of scientific correctness.** Where a physical question
cannot be judged reliably, mark it `human review required`. An LLM must not
assert physical correctness without a reliable basis.

## 8. What ground truth is for

Reference scripts are for **benchmark construction, expected commands, expected
physical settings, task-compliance reference, debugging, and controlled
comparison** — not as the unique correct implementation.

A generated script that differs textually from the reference but is
runtime-successful, task-compliant, and physically reasonable is **not** a
failure. `fix ID = 1` versus `fix ID = thermostat` is not a scoring event if all
references agree and the physics is the same.

## 9. Capability versus reliability

Do not report a single aggregate score. Record failure categories:

```
knowledge_error      syntax_error          command_order_error
missing_command      missing_file          bad_parameter
wrong_units          wrong_ensemble        invalid_reference
premature_termination runtime_error        physical_instability
task_noncompliance   unknown
```

Core question: does DeepSeekHarness already have high LAMMPS authoring
**capability** but a weak **reliability floor**? And which failure modes do M, R,
X, S each reduce?

If SIGA reduces catastrophic failures, say **improved reliability / reduced
catastrophic failures / improved grounding / improved task compliance** — not
"the model got smarter".

## 10. Controlled comparison

Configurations — `Vanilla`, `M`, `M+R`, `M+R+X`, `M+R+X+S`, or the final
factorial design — must hold constant: same model and version, same harness,
same specification, same benchmark, same available files, same HPC environment,
same resource limits, same inference settings, same maximum trajectory budget,
same evaluation pipeline.

**The adapter configuration is the only intended variable.** Vanilla and full
SIGA must not run under materially different system prompts, or the comparison
cannot attribute the effect.

## 11. Full auditability

Every run records: `task_id`, `run_id`, model and settings, adapter
configuration, M version/hash, retrieval results, tool calls, validator results,
termination attempts, generated files, SLURM script, job ID, LAMMPS log, runtime
status, evaluation results, token usage, wall-clock time, failure category.

Any failure must answer: **what failed, and at which layer?** — not `score = 0`.

## 12. Web demo ≠ scientific validation

Two tracks sharing one backend, never conflated:

- **Product track** — Web UI, interaction, job submission, result display.
- **Research track** — benchmark, ablation, controlled experiments, failure
  analysis, metrics.

"The page works" is not "the framework is validated".

## 13. MVP definition

A researcher gives a natural-language LAMMPS request → DeepSeekHarness receives
it → M/R/X/S assist → a complete workspace is generated → validation runs → the
workspace is submitted to `sy_hl_login` → LAMMPS executes → runtime status and
basic results return → the researcher can inspect generated files and logs.

**Not required:** autonomous discovery, self-evolution, hypothesis generation,
arbitrary LAMMPS support, publication-quality physical interpretation.

**Required:** works end to end, is auditable, is reproducible.

## 14. Ask these four questions before adding anything

1. **Is this knowledge always needed?** Yes → M. No → R.
2. **Can this condition be checked deterministically?** Yes → validator/tool, not
   a prompt sentence.
3. **Must this condition always hold?** Yes → harness enforcement (S), not a
   reminder.
4. **Does this improve a measurable failure mode?** No evidence → do not add
   complexity yet.

## 15. Restrained narrative

Not supported: *"AI understands molecular dynamics"*, *"better than human
experts"*, *"autonomously performs scientific research"*.

Supported as a goal: *whether lightweight simulator-interface grounding improves
the reliability of a general coding agent for LAMMPS setup and execution.*

If the data support it: *procedural memory / retrieval / validation /
termination enforcement reduces specific classes of LAMMPS workflow failures.*

**Every conclusion must correspond to a specific experiment.**

## 16. Relationship to the original paper

This stage is **reproduction-oriented**: reproduce M/R/X/S, the
Vanilla-versus-adapter comparison, and failure-mode analysis.

The engineering difference is that the paper's LAMMPS transfer evaluates script
authoring while our environment permits real LAMMPS + HPC, so we add
**execution-grounded evaluation**. That is a *stricter verification method* at
this stage — not a new methodological contribution, and not to be framed as one
in the MVP.
