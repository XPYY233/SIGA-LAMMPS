/**
 * SIGA-LAMMPS harness plugin — the harness-side half of the adapter.
 *
 * This plugin owns exactly two things, and deliberately nothing else:
 *
 *   M  the procedural-memory primer, as an always-on prompt section
 *   S  the stop hook (added in module S)
 *
 * Everything else — retrieval, validation, HPC — reaches the model as MCP tools
 * from the Python adapter, so this file stays small and the science stays in one
 * language. See docs/architecture/02-mvp-architecture.md for why the split
 * falls here.
 *
 * Two load-time failures are intentional and load-bearing:
 *
 *   1. A missing primer FAILS the plugin load rather than degrading to no
 *      memory. A silently absent M would make a benchmark run measure "vanilla"
 *      while reporting "M" — the worst possible bug in an ablation study.
 *   2. A primer over its size budget fails too. The harness never truncates an
 *      always-on section, so an unnoticed growth is billed on every request
 *      forever.
 */

import { readFileSync } from 'node:fs'
import { dirname, isAbsolute, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import type { Context } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'

export const name = 'siga-lammps'

/** Cordis waits for these services before calling `apply`. */
export const inject = ['systemPrompt']

/** Ordinal within the harness's prompt convention (see below). */
const MEMORY_ORDER = 10

/** Name of the interpolated benchmark-task variable the primer references. */
const TASK_VARIABLE = 'siga_task'

export interface Config {
  /** Primer path. Defaults to `<repo>/adapter/memory/lammps_memory.md`. */
  memoryPath?: string
  /** Hard character budget. Defaults to 8400, matching config/config.yaml. */
  maxChars?: number
  /**
   * Active benchmark task id, exposed to the primer as `{{siga_task}}`.
   * The benchmark driver sets this per run; interactive use leaves it unset.
   */
  activeTask?: string
}

export const Config: z<Config> = z.object({
  memoryPath: z.string(),
  maxChars: z.natural().default(8400),
  activeTask: z.string(),
})

/**
 * Resolve the repository root from this module's own location.
 *
 * `<repo>/harness/siga-plugin/src/index.ts` → `<repo>`. Deriving it here rather
 * than reading an environment variable means the plugin needs no configuration
 * to work in a fresh checkout.
 */
function repoRoot(): string {
  return resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..')
}

export function apply(ctx: Context, config: Config): void {
  const memoryPath = config.memoryPath
    ? (isAbsolute(config.memoryPath) ? config.memoryPath : resolve(repoRoot(), config.memoryPath))
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

  const maxChars = config.maxChars ?? 8400
  if (primer.length > maxChars) {
    throw new Error(
      `[siga-lammps] the M primer is ${primer.length} chars, over its ${maxChars} budget. ` +
        'The harness never truncates an always-on section, so this cost is paid on ' +
        'every request. Cut content, or raise memory.max_chars in config/config.yaml ' +
        'and mirror it here.',
    )
  }

  // ---- M: always-on procedural memory ------------------------------------
  //
  // `order` follows the harness convention: -100 is harness identity, 0 the
  // deployment persona, 100-199 tool guidance. M is simulator grounding —
  // domain vocabulary and structural rules — so it sits just after the persona
  // and before the tool band.
  ctx.systemPrompt.section({
    name: 'siga-lammps-memory',
    order: MEMORY_ORDER,
    text: primer,
  })

  // ---- the task variable the primer interpolates --------------------------
  //
  // A provider may return undefined, but rendering a section that references an
  // undefined value FAILS assembly. So this always resolves to a NON-EMPTY
  // string, and the unbound case says so in words rather than leaving a hole.
  //
  // The empty-string case is handled explicitly because `??` does not catch it:
  // a patch that sets `activeTask: ''` (which the generated overlay does) is
  // neither null nor undefined, so `boundTask ?? fallback` yields `''` and the
  // primer renders "Active task: " with nothing after it. Verified live — that
  // is exactly what the first headless run produced.
  const boundTask = config.activeTask?.trim() || undefined
  ctx.systemPrompt.variable(TASK_VARIABLE, () => boundTask ?? 'unspecified (not a benchmark run)')

  ctx.logger?.info?.(
    `[siga-lammps] M active: ${primer.length} chars (~${Math.ceil(primer.length / 4)} tokens/request) ` +
      `from ${memoryPath}`,
  )
}
