#!/usr/bin/env node
/**
 * Token contract for timegpt-docs.
 *
 *  1. Generated files match tokens/tokens.lock.json (no hand edits).
 *  2. Layering: style.css consumes tokens only — no colour / font-size
 *     literals, no primitives (--nixtla-*), no :root declarations.
 *     tokens/docs.css may reference primitives but holds no colour literals.
 *  3. Every var(--x) used in style.css / tokens/docs.css is declared in
 *     tokens/*.css or style.css itself, or is a known Mintlify host variable.
 *  4. style.css / docs.css never redeclare Mintlify's Tailwind theme names.
 *  5. (warn) docs-only tokens nobody reads.
 *
 * Usage: node scripts/docs-tokens/check.mjs   (exit 1 on errors)
 */
import { createHash } from "node:crypto";
import { readFileSync, readdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "..", "..");
const docsDir = join(repoRoot, "timegpt-docs");
const tokensDir = join(docsDir, "tokens");

const read = (p) => readFileSync(p, "utf8");
const stripComments = (css) => css.replace(/\/\*[\s\S]*?\*\//g, "");
const stripUrls = (css) => css.replace(/url\([^)]*\)/g, "url()");

const errors = [];
const warnings = [];
const err = (f, m) => errors.push(`${f}: ${m}`);
const warn = (f, m) => warnings.push(`${f}: ${m}`);

/* ---------------------------------------------------------- 1. lock */
const lock = JSON.parse(read(join(tokensDir, "tokens.lock.json")));
for (const [rel, { sha256 }] of Object.entries(lock.files)) {
  const actual = createHash("sha256").update(read(join(docsDir, rel))).digest("hex");
  if (actual !== sha256) err(rel, "differs from tokens.lock.json — regenerate with scripts/docs-tokens/sync.mjs instead of editing by hand");
}

/* ------------------------------------------------- collect declarations */
const tokenFiles = readdirSync(tokensDir).filter((f) => f.endsWith(".css")).map((f) => join("tokens", f));
const declaredIn = new Map(); // name → Set(files)
const declare = (file, css) => {
  for (const m of stripComments(css).matchAll(/(--[\w-]+)\s*:/g)) {
    if (!declaredIn.has(m[1])) declaredIn.set(m[1], new Set());
    declaredIn.get(m[1]).add(file);
  }
};
for (const f of tokenFiles) declare(f, read(join(docsDir, f)));
const styleCss = read(join(docsDir, "style.css"));
declare("style.css", styleCss);

/* Mintlify-owned variables we read or write deliberately. */
const HOST_VARS = new Set([
  "--shiki-dark", "--shiki-dark-bg", "--shiki-light",
  "--tw-ring-shadow", "--tw-ring-offset-shadow", "--tw-ring-color",
  "--spacing", "--sidebar-width", "--content-width",
]);

/* Mintlify's Tailwind v4 theme names: redeclaring them restyles Mintlify itself. */
const FORBIDDEN_DECL = /^--(spacing|text-(xs|sm|base|lg|xl|[2-5]xl)(--line-height)?|radius-xl|container-[\w-]+|primary-(light|dark)|background-(light|dark)|gray-\d+|color-primary|color-background-(light|dark))$/;

/* ----------------------------------- 2a. comments that close into text */
for (const f of [...tokenFiles, "style.css"]) {
  const raw = read(join(docsDir, f));
  const m = raw.match(/\*\/\S/);
  if (m) err(f, `"*/" glued to text (${m[0]}…) — a glob like --fs-*/--lh-* closes the comment early and breaks the stylesheet`);
}

/* --------------------------------------------- 2/3/4. per-file checks */
const COLOR_LITERAL = /(^|[\s:(,])(#[0-9a-fA-F]{3,8}\b|rgba?\(|hsla?\(|oklch\(|oklab\()|:\s*(white|black|red|blue|green|gray|grey)\s*(!important)?\s*;/;
const FONT_SIZE_LITERAL = /font-size\s*:\s*[0-9.]+(px|rem|em)/;

function checkFile(file, { allowPrimitives, allowRoot }) {
  const css = stripUrls(stripComments(read(join(docsDir, file))));
  css.split("\n").forEach((line, i) => {
    const at = `${file}:${i + 1}`;
    if (COLOR_LITERAL.test(line) && !/^\s*--nixtla-/.test(line)) err(at, `colour literal → use a token: ${line.trim()}`);
    if (FONT_SIZE_LITERAL.test(line)) err(at, `font-size literal → use --fs-* / --text-mono-*: ${line.trim()}`);
    if (!allowPrimitives && /var\(--nixtla-/.test(line)) err(at, `primitive used directly → go through a semantic token: ${line.trim()}`);
  });
  for (const m of css.matchAll(/(?:^|[{;\s])(--[\w-]+)\s*:/g)) {
    if (FORBIDDEN_DECL.test(m[1])) err(file, `redeclares Mintlify theme variable ${m[1]}`);
  }
  if (!allowRoot && /(^|\s|,):root\s*\{/.test(css)) err(file, "declares tokens in :root — move them to tokens/docs.css");
  for (const m of css.matchAll(/var\((--[\w-]+)/g)) {
    const name = m[1];
    if (!declaredIn.has(name) && !HOST_VARS.has(name)) err(file, `var(${name}) is not declared anywhere`);
  }
}

checkFile("style.css", { allowPrimitives: false, allowRoot: false });
checkFile("tokens/docs.css", { allowPrimitives: true, allowRoot: true });

/* --------------------------------------------------- 5. unused docs tokens */
const docsCss = stripComments(read(join(tokensDir, "docs.css")));
const used = new Set([...(styleCss + docsCss).matchAll(/var\((--[\w-]+)/g)].map((m) => m[1]));
for (const m of docsCss.matchAll(/^\s*(--[\w-]+)\s*:/gm)) {
  const name = m[1];
  const isHostOverride = (declaredIn.get(name)?.size ?? 0) > 1 || /^--font-weight-/.test(name);
  if (!used.has(name) && !isHostOverride) warn("tokens/docs.css", `${name} is declared but never read`);
}

/* ------------------------------------------------------------- report */
for (const w of warnings) console.warn(`⚠ ${w}`);
for (const e of errors) console.error(`✖ ${e}`);
if (errors.length) {
  console.error(`\n${errors.length} error(s), ${warnings.length} warning(s)`);
  process.exit(1);
}
console.log(`✔ tokens contract OK (${warnings.length} warning(s))`);
