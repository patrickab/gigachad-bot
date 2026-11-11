# draft style guide

This deck is a draft. The goal is to settle what the talk says and in what order, together with the user. Visual polish comes later in the `presentation` theme, so spend no effort on layout. Start a new draft from `skeleton.tex`, it also shows every macro below.

## Your job

- Brainstorm with the user: propose what belongs on the slides, what is missing, what to cut or merge, and where the order breaks the story
- Keep the storyline readable from the frame titles alone: read in order, they tell the talk
- Mark open points in the frame with `\question{...}`, never fill a gap with guessed content
- Change the structure freely: move, split and merge frames and chapters whenever the story gets clearer

## Outline

- `\section{}` and `\subsection{}` carry the structure: the agenda frame (`\tableofcontents`) lists them and every `\section` opens its chapter slide on its own, so the structure is visible while drafting
- The outline is the frame titles under their sections, keep it in sync with the story when frames move

## Frames

- Plain beamer only: title, short `itemize`, `\note{}`. No columns, overlays, blocks, tables or charts
- A slide that needs a visual says what it shows and what it proves: `\visual{Chart: error against grid resolution, ours lowest at every size}`. Name the kind (chart, table, figure, number, steps, timeline, quote) first
- Figures the user already has may go in with `\includegraphics[width=\linewidth]{imgs/...}`
- Sources go right under what they back: `\source{Author Year, Fig. n}`

## Polish

Switching the root to `\usetheme[progress]{presentation}` ends the draft. The polish follows `.templates/presentation/STYLE.md`, every `\question` is settled before it. Title, agenda and sections stay as they are, the new theme only redraws them.
