/**
 * SIGA-LAMMPS harness plugin — the harness-side half of the adapter.
 *
 * This plugin owns exactly two things, and deliberately nothing else:
 *
 *   M  the procedural-memory primer, as an always-on prompt section
 *   S  the termination gate (the stop hook)
 *
 * Everything else — retrieval, validation, HPC — reaches the model as MCP tools
 * from the Python adapter, so this file stays small and the science stays in one
 * language. See docs/architecture/02-mvp-architecture.md for why the split falls
 * here.
 *
 * ## Why S is a harness mechanism and not a prompt sentence
 *
 * "Please validate before finishing" is a behavioural instruction. This is
 * control-flow enforcement: the harness asks this listener before the turn
 * boundary commits, and a listener that objects steers another step. The agent
 * cannot skip it, forget it, or argue with it.
 *
 * That also makes S structurally different from X. X is a tool the agent may
 * choose to call; both reach the validator through the *same* `adapter/cli.py`,
 * which is what makes "the validator applies the same checks as the hook" a fact
 * about the code rather than a promise.
 */

import { execFile } from 'node:child_process'
import { existsSync, readFileSync, readdirSync } from 'node:fs'
import { dirname, isAbsolute, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { promisify } from 'node:util'

import type { Context } from '@deepseek-ai/cordis'
import { createUserMessage } from '@deepseek-ai/dsh-llm'
import z from '@deepseek-ai/schemastery'

const run = promisify(execFile)

export const name = 'siga-lammps'

/** Cordis waits for these services before calling `apply`. */
export const inject = ['systemPrompt']

/** Ordinal within the harness's prompt convention: after the persona, before tool guidance. */
const MEMORY_ORDER = 10

/** Name of the interpolated benchmark-task variable the primer references. */
const TASK_VARIABLE = 'siga_task'

/** Marks steered messages as plugin-authored, never mistaken for the user. */
const PLUGIN_SOURCE = { kind: 'plugin', plugin: 'siga-lammps' } as const

/** Exit codes of `adapter/cli.py validate`, by contract. */
const EXIT_INVALID = 1

export interface Config {
  /** Primer path. Defaults to `<repo>/adapter/memory/lammps_memory.md`. */
  memoryPath?: string
  /** Hard character budget for the primer. Defaults to 8000, matching config/config.yaml. */
  maxChars?: number
  /** Active benchmark task id, exposed to the primer as `{{siga_task}}`. */
  activeTask?: string
  /** Enable the S termination gate. Defaults to true. */
  stopGate?: boolean
  /** Python interpreter running the adapter CLI. Defaults to `<repo>/.venv/bin/python`. */
  pythonBin?: string
  /** Benchmark task id to validate against, if any. */
  taskId?: string
  /**
   * How many times S may block a single turn before letting it close.
   *
   * The harness imposes no continuation cap of its own — its own hook bridges
   * carry a `TODO(stop-loop-guard)` — so without this an unrelated validator that
   * never passes would loop the agent indefinitely, burning tokens with no way
   * out. This budget is load-bearing, not defensive padding.
   */
  maxBlocks?: number
  /** Wall-clock limit for one validator invocation. */
  validatorTimeoutMs?: number
}

export const Config: z<Config> = z.object({
  memoryPath: z.string(),
  maxChars: z.natural().default(8000),
  activeTask: z.string(),
  stopGate: z.boolean().default(true),
  pythonBin: z.string(),
  taskId: z.string(),
  maxBlocks: z.natural().default(3),
  validatorTimeoutMs: z.natural().default(60_000),
})

/** Resolve the repository root from this module's own location. */
function repoRoot(): string {
  return resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..')
}

/** Filenames LAMMPS input scripts conventionally use. */
const SCRIPT_PREFIXES = ['in.', 'input.']

/**
 * The LAMMPS input script in this workspace, if there is one.
 *
 * The gate must be inert everywhere else. Without this check S would attach
 * itself to every conversation in the profile and start blocking turns for
 * lacking a `units` command — turning a useful gate into a broken harness.
 * Ambiguity resolves to the shortest name, matching the canonical script in
 * every LAMMPS example set.
 */
function lammpsScriptIn(workspace: string): string | undefined {
  let entries: string[]
  try {
    entries = readdirSync(workspace)
  } catch {
    return undefined
  }
  const candidates = entries
    .filter((entry) => SCRIPT_PREFIXES.some((prefix) => entry.startsWith(prefix)))
    .sort((a, b) => a.length - b.length || a.localeCompare(b))
  return candidates[0]
}

interface ValidatorFinding {
  code: string
  message: string
  line?: number
}

interface ValidatorOutcome {
  kind: 'valid' | 'invalid' | 'error' | 'not-a-lammps-workspace'
  errors: ValidatorFinding[]
  warnings: ValidatorFinding[]
  detail?: string
}

/**
 * Run the shared validator.
 *
 * The outcomes are kept distinct on purpose. Collapsing `error` into `invalid`
 * would let a broken validator block every turn forever; collapsing it into
 * `valid` would silently disable the gate and quietly turn an M+R+X+S run into a
 * vanilla one. Neither failure is visible in a score.
 */
async function validate(workspace: string, config: Config): Promise<ValidatorOutcome> {
  if (lammpsScriptIn(workspace) === undefined) {
    return { kind: 'not-a-lammps-workspace', errors: [], warnings: [] }
  }

  const pythonBin = config.pythonBin
    ? isAbsolute(config.pythonBin)
      ? config.pythonBin
      : resolve(repoRoot(), config.pythonBin)
    : join(repoRoot(), '.venv', 'bin', 'python')
  if (!existsSync(pythonBin)) {
    return {
      kind: 'error',
      errors: [],
      warnings: [],
      detail: `adapter interpreter not found at ${pythonBin}`,
    }
  }

  const argv = [
    '-m',
    'adapter.cli',
    '--json',
    'validate',
    workspace,
    ...(config.taskId ? ['--task', config.taskId] : []),
  ]

  try {
    const { stdout } = await run(pythonBin, argv, {
      cwd: repoRoot(),
      timeout: config.validatorTimeoutMs ?? 60_000,
      maxBuffer: 4 * 1024 * 1024,
    })
    const payload = JSON.parse(stdout) as {
      valid: boolean
      errors?: ValidatorFinding[]
      warnings?: ValidatorFinding[]
    }
    return {
      kind: payload.valid ? 'valid' : 'invalid',
      errors: payload.errors ?? [],
      warnings: payload.warnings ?? [],
    }
  } catch (cause) {
    const failure = cause as { code?: number | string; stdout?: string }
    // Exit code 1 is a *finding*, not a failure: the validator ran and the script
    // is invalid. Every other shape means it could not run.
    if (failure.code === EXIT_INVALID && typeof failure.stdout === 'string') {
      try {
        const payload = JSON.parse(failure.stdout) as {
          errors?: ValidatorFinding[]
          warnings?: ValidatorFinding[]
        }
        return { kind: 'invalid', errors: payload.errors ?? [], warnings: payload.warnings ?? [] }
      } catch {
        return { kind: 'error', errors: [], warnings: [], detail: 'validator output was not JSON' }
      }
    }
    return {
      kind: 'error',
      errors: [],
      warnings: [],
      detail: cause instanceof Error ? cause.message : String(cause),
    }
  }
}

/** Compose the repair instruction S steers back to the agent. */
function repairInstruction(outcome: ValidatorOutcome, script: string, remaining: number): string {
  const lines: string[] = [
    'Stop-gate: your LAMMPS workspace does not pass validation, so this turn cannot end.',
    '',
    `Deterministic validation of \`${script}\` found ${outcome.errors.length} error(s):`,
    '',
  ]
  for (const finding of outcome.errors) {
    const where = finding.line !== undefined ? ` (line ${finding.line})` : ''
    lines.push(`  - [${finding.code}]${where} ${finding.message}`)
  }
  if (outcome.warnings.length > 0) {
    lines.push(
      '',
      `Also ${outcome.warnings.length} warning(s) worth checking, which do not block on their own:`,
    )
    for (const finding of outcome.warnings.slice(0, 5)) {
      const where = finding.line !== undefined ? ` (line ${finding.line})` : ''
      lines.push(`  - [${finding.code}]${where} ${finding.message}`)
    }
  }
  lines.push(
    '',
    'Fix the errors in the input script, then call mcp__lammps__validate_lammps_input to confirm.',
    'If you believe a finding is wrong, say which one and why rather than working around it.',
    `You have ${remaining} more gate check(s) this turn before it is allowed to close.`,
  )
  return lines.join('\n')
}

export function apply(ctx: Context, config: Config): void {
  const memoryPath = config.memoryPath
    ? isAbsolute(config.memoryPath)
      ? config.memoryPath
      : resolve(repoRoot(), config.memoryPath)
    : resolve(repoRoot(), 'adapter', 'memory', 'lammps_memory.md')

  let primer: string
  try {
    primer = readFileSync(memoryPath, 'utf8')
  } catch (cause) {
    throw new Error(
      `[siga-lammps] cannot read the M primer at ${memoryPath}. Refusing to load ` +
        'without procedural memory: a run that silently lacks M would be ' +
        'indistinguishable from a vanilla baseline while claiming to be M.',
      { cause },
    )
  }
  if (primer.trim() === '') {
    throw new Error(`[siga-lammps] the M primer at ${memoryPath} is empty`)
  }

  const maxChars = config.maxChars ?? 8000
  if (primer.length > maxChars) {
    throw new Error(
      `[siga-lammps] the M primer is ${primer.length} chars, over its ${maxChars} budget. ` +
        'The harness never truncates an always-on section, so this cost is paid on every ' +
        'request. Cut content, or raise memory.max_chars in config/config.yaml and mirror it here.',
    )
  }

  // ---- M: always-on procedural memory ------------------------------------
  //
  // `order` follows the harness convention: -100 harness identity, 0 persona,
  // 100-199 tool guidance. M is simulator grounding — domain vocabulary and
  // structural rules — so it sits after the persona and before the tool band.
  ctx.systemPrompt.section({
    name: 'siga-lammps-memory',
    order: MEMORY_ORDER,
    text: primer,
  })

  // A provider may return undefined, but rendering a section that references an
  // undefined value FAILS assembly, so this always resolves to a non-empty
  // string. The empty-string case is explicit because `??` does not catch it: a
  // patch setting `activeTask: ''` is neither null nor undefined, and
  // `'' ?? fallback` yields `''`, rendering the primer with a silent hole.
  const boundTask = config.activeTask?.trim() || undefined
  ctx.systemPrompt.variable(TASK_VARIABLE, () => boundTask ?? 'unspecified (not a benchmark run)')

  // ---- S: the termination gate -------------------------------------------
  if (config.stopGate === false) {
    ctx.logger?.info?.('[siga-lammps] M active; S stop gate disabled by config')
    return
  }

  const maxBlocks = config.maxBlocks ?? 3
  /** Blocks used this turn, keyed by session and turn. */
  const blocksThisTurn = new Map<string, number>()
  const reportedExhaustion = new Set<string>()

  ctx.on('agent/turn-stopping', async ({ agent, turn, signal }) => {
    if (signal.aborted) return

    const workspace = agent.session.header.cwd
    if (!workspace) return

    const key = `${agent.id}:${turn}`
    const used = blocksThisTurn.get(key) ?? 0

    if (used >= maxBlocks) {
      // Let the turn close. The harness has no cap of its own, so this is the
      // only thing standing between a never-passing validator and an infinite
      // loop. Reported once so the benchmark can count non-convergence instead
      // of having it vanish into a timeout.
      if (!reportedExhaustion.has(key)) {
        reportedExhaustion.add(key)
        ctx.logger?.warn?.(
          `[siga-lammps] S gate exhausted after ${maxBlocks} blocks on turn ${turn}; allowing ` +
            'the turn to close. This is a measured non-convergence, not a pass.',
        )
      }
      return
    }

    const outcome = await validate(workspace, config)
    if (outcome.kind === 'valid' || outcome.kind === 'not-a-lammps-workspace') return

    if (outcome.kind === 'error') {
      // Fail *open*, loudly. Blocking on a validator that cannot run would trap
      // the agent with no way to make progress, and the result would be
      // indistinguishable from a genuine failure in any downstream score.
      ctx.logger?.error?.(
        `[siga-lammps] S could not run the validator (${outcome.detail ?? 'unknown reason'}); ` +
          'allowing the turn to close. An unenforced gate is not a passing gate.',
      )
      return
    }

    blocksThisTurn.set(key, used + 1)
    const script = lammpsScriptIn(workspace) ?? 'the input script'
    ctx.logger?.info?.(
      `[siga-lammps] S blocked turn ${turn} (${used + 1}/${maxBlocks}): ` +
        `${outcome.errors.length} validation error(s)`,
    )
    agent.steer(
      createUserMessage({
        content: [
          {
            type: 'text',
            text: repairInstruction(outcome, script, maxBlocks - used - 1),
          },
        ],
        source: PLUGIN_SOURCE,
      }),
    )
  })

  ctx.effect(
    () => () => {
      blocksThisTurn.clear()
      reportedExhaustion.clear()
    },
    'siga-lammps: stop-gate state',
  )

  ctx.logger?.info?.(
    `[siga-lammps] M active: ${primer.length} chars ` +
      `(~${Math.ceil(primer.length / 4)} tokens/request); ` +
      `S gate active with a ${maxBlocks}-block budget per turn`,
  )
}
