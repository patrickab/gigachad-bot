# Product

## Register

product

## Users

Solo developer-power-user running this on a personal machine (Linux, local-first). Day-to-day context: long chat sessions with many parallel tabs, deep research queries, OCR work, side-by-side model comparison. Keyboard-first; defines all personal shortcuts himself. Optimizes the tool for sustained deep-focus use, not for demo or sharing.

## Product Purpose

A unified local chat interface over 2100+ LLMs (Ollama, vLLM, Gemini, OpenAI, etc.) with four modes: chat, deep research, web search, OCR. Exists to consolidate workflows that would otherwise live in 4+ separate tools. Success = "I haven't opened ChatGPT in a month."

## Brand Personality

Three words: **focused, restrained, technical.**

- Flow state first. The user is here to immerse in work, learning, or exploration. Every element either supports deep focus or gets out of the way.
- Subtle dopamine kicks. Micro-interactions reinforce progress (copy checkmark, smooth toggles) without demanding attention. Never celebratory: no pulse, no flash, no sound.
- Spatial consistency. Elements live in predictable zones (header, bottom-right, sidebar). Clean surfaces, subtle glows, smooth transitions.
- Monochromatic with red reserved for destruction. No accent hue at all — active/selected/interactive states use ink/subtle surface shifts, not a colored highlight. Red is used only for destructive actions. No per-feature color coding. No 2nd or 3rd accent.
- Smooth and subtle. All UI motion is understated.

## Anti-references

- **Not SaaS-cream / Linear-esque off-white.** Avoid the 2024-2026 AI-tool warm-neutral body bg. The "yellowish broken white" light mode should still feel distinct from paper/parchment stock.
- **Not pure black/white (Material/Apple stark).** Dark mode descends through warm-tinted greys, not neutral-grey all the way down. Light mode starts below pure white.
- **No accent hue.** No cyan/amber/yellow/green anywhere. The UI is strictly warm-tinted monochromatic: all interactive states are expressed through surface elevation shifts (focus: `border-divider-strong`, hover: `bg-hover`, active: `bg-surface-elevated`). Red is reserved for destructive actions only and must never be used decoratively.
- **Not shadcn-default surface stack.** Surfaces (page / surface / elevated / overlay) should feel hand-tuned via `color-mix` interpolation, not a default ramp off zinc-50/100/200.

## Design Principles

1. **Containers dynamically adapt their color scheme via `color-mix`.** A surface inherits its parent's color and blends toward the appropriate elevation target. No hardcoded `bg-zinc-X` per component — only declared intents like "elevated one step" that interpolate locally. This is the single source of truth for the surface stack.
2. **Strictly monochromatic, warm-tinted, with red reserved for destruction.** The `--ink` role token carries a warm hue (~85° in OKLCH) at tiny chroma (0.005–0.020), tinting all text and surfaces with a subtle warmth. No accent hue exists. Active/selected/interactive states are conveyed through surface elevation shifts (`bg-surface-elevated`, `border-divider-strong`, `bg-hover`), not chroma. Red (`--danger`) is used for destructive actions only.
3. **Themes are flips of the same scale, not separate scales.** Dark and light modes share the same role-based token names (`--ink`, `--paper`, `--surface`, `--ink-muted`, etc.) and the same elevation logic. The only thing that changes is the polarity and warmth of the base.
4. **Backgrounds and foregrounds must both carry context.** Dividers, hover states, and active states are color-mix outputs (e.g. `color-mix(in oklab, var(--ink) 8%, transparent)`), not hardcoded zinc values that disappear against a new background.
5. **Restraint wins over cleverness.** The hierarchy is: a small set of base role tokens (`--ink`, `--paper`, `--surface`, `--danger`) → `color-mix` derivations (`--divider`, `--hover-overlay`, `--danger-soft`) → Tailwind v4 `@theme` mapping (`--color-ink: var(--ink)`) → component classes (`text-ink`, `bg-paper`). Components never see raw hex or raw zinc values.

## Accessibility & Inclusion

Best-effort, no formal WCAG target. User defines all personal keyboard mappings himself, so keyboard-naming and shortcut-naming conventions are his concern, not the design system's. The product still aims for: readable body text on every surface, visible focus indicators, and dividers that actually separate (i.e. visible against the surfaces they divide). Reduced-motion respected.
