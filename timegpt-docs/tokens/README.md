# Design tokens for the docs

The docs consume the Nixtla design system through CSS custom properties. Mintlify
loads every `.css` file in this content directory on every page, so each file
here is a layer; `style.css` (one level up) only writes component rules against
these tokens.

## Layers

| File | Owner | Contents |
| --- | --- | --- |
| `primitives.css` | **generated** | The seven OKLCH ramps, `--nixtla-{neutral,haze,water,aubergine,teal,red,orange}-{50…950}`. |
| `theme.css` | **generated** | `--color-nixtla-*` aliases. In mono-nixtla this is Tailwind's `@theme inline` bridge; here it is materialized as plain `:root` properties because `semantic.css` resolves against those names. |
| `semantic.css` | **generated** | The shadcn-named contract: `--background`, `--foreground`, `--primary`, `--secondary`, `--muted`, `--accent`, `--card`, `--popover`, `--border`, `--input`, `--ring`, `--highlight`, status colours, `--sidebar-*`, `--table-*`, `--blog-code-*`, `--chart-*`, layout spacing (`--spacing-space-*`, `--section-*`) … Light in `:root`, dark in `.dark`. |
| `typography.css` | **generated** | Type scale `--fs-{xs…7xl}` / `--lh-*` with responsive overrides at 48rem and 64rem, tracking `--text-*--letter-spacing`, and the mono roles `--text-mono-{nano,mini,regular}`, `--text-caption`. |
| `motion.css` | **generated** | `--duration-{instant,quick,base,slow,deliberate}`, `--ease-{standard,entrance,exit,overshoot}`, reduced-motion override. |
| `fonts.css` | docs | `@font-face` for the faces hosted in `/fonts`. |
| `docs.css` | docs | Tokens the DS has no slot for (grain, text-shift recipe, code-frame extras, `--space-*` component grid, layout rhythm) and overrides of synced tokens for this host (fonts, weights). |
| `tokens.lock.json` | generated | Source commit and SHA-256 of every generated file. |

Source of truth: `mono-nixtla/packages/tokens/src/` (the `@repo/tokens` package).
`nixtla_web` on `main` still uses the previous scale (`--display-*`, `--heading-*`)
and is **not** the reference.

## Sync and lint

```sh
# Re-pull tokens from a local mono-nixtla checkout (defaults to ../mono-nixtla)
node scripts/docs-tokens/sync.mjs
MONO_NIXTLA_PATH=~/code/mono-nixtla node scripts/docs-tokens/sync.mjs

# Contract check (runs in CI on every change under timegpt-docs/)
node scripts/docs-tokens/check.mjs
```

`sync.mjs` keeps every plain CSS block verbatim and drops the Tailwind-only
at-rules. Mintlify serves custom CSS raw (no Tailwind build), so `@theme`,
`@utility` and `@apply` would be inert; the script materializes the `@theme`
entries the browser actually needs (`--color-nixtla-*`, `--spacing-space-*`,
tracking and mono roles, `--ease-*`) and nothing else. The bare `--text-{step}`
aliases are intentionally not emitted: they duplicate `--fs-*`/`--lh-*` and
would override Mintlify's own Tailwind scale.

`check.mjs` fails the build when:

- a generated file no longer matches `tokens.lock.json` (hand edit);
- `style.css` contains a colour or `font-size` literal, reads a `--nixtla-*`
  primitive directly, or declares tokens in `:root`;
- `tokens/docs.css` contains a colour literal;
- a `var(--x)` has no declaration (other than the Mintlify host variables listed
  in the script);
- `style.css` or `docs.css` redeclare one of Mintlify's Tailwind theme variables
  (`--spacing`, `--text-xs…5xl`, `--radius-xl`, `--container-*`, `--gray-*`,
  `--primary-light/dark`, `--background-light/dark`).

## Rules for contributors

1. **Components read semantic tokens only.** In `style.css` use `--foreground`,
   `--card`, `--border`, `--fs-sm`… Never `--nixtla-water-300`, never a hex.
2. **New docs-only token → `docs.css`.** Its value must be a primitive, a
   semantic token, or a `color-mix()` of one. Put the dark value next to the
   light value in the same file.
3. **Overriding a synced token** (fonts, weights) is done in `docs.css` with
   `html:root` / `html.dark` so it wins regardless of file load order.
4. **Never edit a generated file.** Change `@repo/tokens` in mono-nixtla, then
   run the sync and commit the regenerated files plus the lock.
5. Need a new slot shared with the website? Add it to `@repo/tokens`, not here.

## Vocabulary map

| Intent | Token |
| --- | --- |
| Text, icons as ink | `--foreground`, secondary text `--muted-foreground` |
| Filled control (CTA, step badge, send button) | `--primary` + `--primary-foreground`; hover `color-mix(in oklch, var(--primary) 80%, transparent)` |
| Page | `--background`; sidebar `--sidebar` |
| Raised surfaces | cards/accordions/chat sheet `--card`; menus `--popover` + `--popover-foreground`, hover `--popover-accent` |
| Chips, blockquote, inline code | `--secondary` + `--secondary-foreground` |
| Hover surface | content (cards, summaries, nav) `--accent` / `--sidebar-accent`; outline controls `--muted` |
| Lines | hairline `--border`; TOC rail `--rule`; table lines `--table-border` (= `--table-ring` at 15 %, as in `typeset.css`) |
| Code frame (always dark) | `--blog-code-bg`, `--blog-code-header-bg`, `--blog-code-border`, `--blog-code-text`, `--blog-code-muted`; extras `--code-hover-overlay`, `--code-tab-hover` |
| Type | h1 `--fs-3xl`, h2 `--fs-2xl`, body `--fs-base`, tables/captions `--fs-sm`; labels `--text-mono-nano` / `--text-mono-mini` (eyebrow-xs / eyebrow-sm) |
| Motion | `--duration-quick` / `--duration-base` / `--duration-slow` with `--ease-standard` |

## Known host collisions (accepted)

Mintlify reads `--primary`, `--primary-light`, `--primary-dark` and `--gray-*` as
space-separated RGB triplets inside `rgb(var(--primary))`. Our `--primary` is an
OKLCH colour, so those Mintlify rules fall back to their defaults; the surfaces
that matter are restyled in `style.css`. This was already the case before the
tokenization and is a deliberate trade for keeping one vocabulary with the
website.

## Fonts

`@repo/tokens` sets `--font-mono`/`--font-code` to Berkeley Mono and
`--font-heading` to Tiempos. Both are commercial faces and this repository is
public, so they are not hosted here. `docs.css` repoints the mono stack at
Supply and headings at the sans until the files can be served from a Nixtla
origin; the token names are already the final ones, so the switch is a one-line
change in `docs.css` plus the `@font-face` rules.
