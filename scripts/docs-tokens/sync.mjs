#!/usr/bin/env node
/**
 * Sync design tokens from mono-nixtla (`packages/tokens/src`) into
 * `timegpt-docs/tokens/`, adapting them for Mintlify.
 *
 * Mintlify serves custom CSS raw to the browser (no Tailwind build), so the
 * Tailwind v4 at-rules in the source (`@theme`, `@theme inline`, `@utility`)
 * are inert there. This script keeps every plain CSS block verbatim and
 * materializes, as `:root` custom properties, only the `@theme` entries that
 * the browser needs to resolve:
 *
 *   theme.css       `--color-nixtla-*` bridge (consumed by semantic.css)
 *   semantic.css    `--spacing-space-*` (consumed by the `--section-*` tokens)
 *   typography.css  `--text-*--letter-spacing`, `--text-mono-*`, `--text-caption*`
 *                   (the bare `--text-{step}` sizes and `--line-height`s are
 *                   dropped: they alias `--fs-*`/`--lh-*` and would override
 *                   Mintlify's own Tailwind scale)
 *   motion.css      `--ease-*`
 *
 * Usage:
 *   node scripts/docs-tokens/sync.mjs            # reads ../mono-nixtla
 *   MONO_NIXTLA_PATH=/path/to/mono-nixtla node scripts/docs-tokens/sync.mjs
 */
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "..", "..");
const monoRoot = resolve(process.env.MONO_NIXTLA_PATH ?? join(repoRoot, "..", "mono-nixtla"));
const srcDir = join(monoRoot, "packages", "tokens", "src");
const outDir = join(repoRoot, "timegpt-docs", "tokens");
const lockPath = join(outDir, "tokens.lock.json");

if (!existsSync(srcDir)) {
  console.error(`✖ mono-nixtla tokens not found at ${srcDir}\n  Set MONO_NIXTLA_PATH to the mono-nixtla checkout.`);
  process.exit(1);
}

/* ---------------------------------------------------------------- parsing */

/** Split a stylesheet into top-level blocks: { prelude, body, raw }. Comments
 *  and whitespace between blocks are dropped (headers are regenerated). */
function topLevelBlocks(css) {
  const blocks = [];
  let i = 0;
  const n = css.length;
  while (i < n) {
    // skip whitespace and comments
    if (/\s/.test(css[i])) { i++; continue; }
    if (css.startsWith("/*", i)) { i = css.indexOf("*/", i + 2) + 2; continue; }
    const start = i;
    const open = css.indexOf("{", i);
    if (open === -1) break;
    let depth = 0, j = open;
    for (; j < n; j++) {
      if (css[j] === "{") depth++;
      else if (css[j] === "}") { depth--; if (depth === 0) break; }
    }
    blocks.push({
      prelude: css.slice(start, open).trim(),
      body: css.slice(open + 1, j),
      raw: css.slice(start, j + 1),
    });
    i = j + 1;
  }
  return blocks;
}

/** Declarations directly inside a body (ignores nested blocks such as @keyframes). */
function declarations(body) {
  const out = [];
  let depth = 0, buf = "";
  for (const ch of body) {
    if (ch === "{") depth++;
    if (ch === "}") { depth--; buf = ""; continue; }
    if (depth > 0) continue;
    if (ch === ";") { const d = buf.trim(); if (d.startsWith("--")) out.push(d); buf = ""; continue; }
    buf += ch;
  }
  return out;
}

const isTheme = (b) => /^@theme\b/.test(b.prelude);
const isUtility = (b) => /^@utility\b/.test(b.prelude);
const isDrop = (b) => isTheme(b) || isUtility(b);

function materialize(decls, { fallback } = {}) {
  if (decls.length === 0) return null;
  const lines = decls.map((d) => `  ${fallback ? fallback(d) : d};`);
  return `/* Materialized from @theme (Tailwind-only in the source). */\n:root {\n${lines.join("\n")}\n}`;
}

/* ------------------------------------------------------------ transforms */

const files = {
  "primitives.css": {
    note: "Verbatim copy: the seven OKLCH ramps.",
    transform: (css) => topLevelBlocks(css).map((b) => b.raw),
  },
  "theme.css": {
    note: "`@theme inline` → `:root`: the --color-nixtla-* aliases semantic.css resolves against.",
    transform: (css) =>
      topLevelBlocks(css).map((b) => (isTheme(b) ? materialize(declarations(b.body)) : b.raw)),
  },
  "semantic.css": {
    note: "Plain blocks verbatim (:root, media queries, .dark). From @theme inline only --spacing-space-* survives (the --section-* tokens read it); the Tailwind colour/container/radius bridge and @utility helpers are dropped.",
    transform: (css) =>
      topLevelBlocks(css).map((b) => {
        if (isUtility(b)) return null;
        if (!isTheme(b)) return b.raw;
        const spacing = declarations(b.body).filter((d) => d.startsWith("--spacing-space-"));
        return materialize(spacing, {
          // Tailwind's base unit; Mintlify defines it too, the fallback keeps us independent.
          fallback: (d) => d.replace("var(--spacing)", "var(--spacing, 0.25rem)"),
        });
      }),
  },
  "typography.css": {
    note: "Scale (:root --fs-* and --lh-* + responsive overrides) verbatim. From @theme inline: tracking, mono roles and caption. The bare --text-{step} size/line-height aliases are dropped (they duplicate --fs-* and --lh-* and would override Mintlify's own Tailwind scale). @utility eyebrow-* helpers are expressed in style.css via the mono-label recipe.",
    transform: (css) =>
      topLevelBlocks(css).map((b) => {
        if (isUtility(b)) return null;
        if (!isTheme(b)) return b.raw;
        const keep = declarations(b.body).filter(
          (d) => !/^--text-(xs|sm|base|lg|xl|[2-7]xl)(--line-height)?\s*:/.test(d),
        );
        return materialize(keep);
      }),
  },
  "motion.css": {
    note: "Durations and the reduced-motion override verbatim. From @theme only the --ease-* curves (no --animate-*, no keyframes). @utility paused dropped.",
    transform: (css) =>
      topLevelBlocks(css).map((b) => {
        if (isUtility(b)) return null;
        if (!isTheme(b)) return b.raw;
        return materialize(declarations(b.body).filter((d) => d.startsWith("--ease-")));
      }),
  },
};

/* ------------------------------------------------------------------- run */

let commit = "unknown";
try {
  const { execFileSync } = await import("node:child_process");
  commit = execFileSync("git", ["-C", monoRoot, "rev-parse", "--short=12", "HEAD"], { encoding: "utf8" }).trim();
} catch { /* not a git checkout */ }

const syncedAt = new Date().toISOString().slice(0, 10);
const sha256 = (s) => createHash("sha256").update(s).digest("hex");
const lock = { source: { repo: "mono-nixtla", path: "packages/tokens/src", commit, syncedAt }, files: {} };

mkdirSync(outDir, { recursive: true });

for (const [name, { note, transform }] of Object.entries(files)) {
  const src = readFileSync(join(srcDir, name), "utf8");
  if (note.includes("*/")) throw new Error(`note for ${name} contains "*/" and would close the header comment`);
  const header = [
    "/* GENERATED by scripts/docs-tokens/sync.mjs — DO NOT EDIT BY HAND.",
    `   Source: mono-nixtla/packages/tokens/src/${name} @ ${commit} (${syncedAt})`,
    `   ${note}`,
    "   Docs-only tokens and overrides live in tokens/docs.css. */",
  ].join("\n");
  const body = transform(src).filter(Boolean).join("\n\n");
  const out = `${header}\n\n${body}\n`;
  writeFileSync(join(outDir, name), out);
  lock.files[`tokens/${name}`] = { from: `packages/tokens/src/${name}`, sha256: sha256(out) };
  console.log(`✔ tokens/${name}`);
}

writeFileSync(lockPath, JSON.stringify(lock, null, 2) + "\n");
console.log(`✔ ${relative(repoRoot, lockPath)} (source @ ${commit})`);
