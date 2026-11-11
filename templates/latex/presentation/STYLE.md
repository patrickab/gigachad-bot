# presentation style guide

A calm, minimalist Beamer theme on warm paper. Build every slide from a primitive in `primitives/` and copy its frame from `example.tex` (or the primitive file itself). Start a new deck from `skeleton.tex`.

## Philosophy

- **Calm**: warm paper background, soft cards, no heavy borders, no decoration without a function.
- **Content first**: the claim and its evidence get the space, chrome stays small and quiet.
- **One accent**: the accent color has one job, marking the single thing in focus.
- **One message per slide**: if a slide needs two claims, it is two slides.
- **Consistent**: one corner radius, one gap, one margin, one arrow style. New slides copy their neighbours.
- **Shared primitives**: reuse a primitive before writing custom markup. If none fits, extend one in the sty instead of a one-off frame.

## Structure

- The talk is told in chapters: one `\section{Chapter}` per chapter, one `\subsection{Step}` per step inside it. Nothing else marks structure.
- Agenda: `\begin{frame}{Agenda}{Optional italic line}\tableofcontents\end{frame}`. The theme sets the chapters as columns with their subsections, beamer's toc options such as `[currentsection]` still apply. Never build an agenda by hand.
- Every `\section` opens its chapter slide on its own: number, title, its subsections, all chapters on a faint track. Never add divider frames yourself.
- Content slides show where you are in a small muted note in the footer, before the page number (`Chapter · Step`, the step dropped when too long). With `\usetheme[progress]{...}` the footer adds a quiet bar with one segment per chapter, filled up to the current slide.
- Three to five chapters, at most six. Two to four subsections per chapter, a chapter may have none.
- Rhythm: each chapter gets at least one slide where a single visual (hero number, chart, full-bleed image, figure, steps) carries the claim. A `\statement` slide may close a chapter.
- Chapter titles: one to three words, a noun phrase (`Method`, `Results`). Subsection titles: one to three words naming the step (`Key figures`), short enough for one line in an agenda column. No sentences, no punctuation.
- `\section[Short]{Title}` and `\subsection[Short]{Title}`: the footer location uses the short form, agenda and chapter slides show the full title.
- Frames before the first `\section` (title, agenda) and `[plain]` frames carry no location and no progress bar.
- Build with `latexmk -pdf`: agenda, chapter slides and progress bar read the previous run, latexmk reruns until they are current.

## Slide rules

- Title is the claim as a full sentence: `\begin{frame}{Claim as a sentence}{Short context}`.
- The subtitle is a visual anchor, not a sentence to read: two to five words naming what the slide shows (scope, unit, source), e.g. `Error in mm, lower is better`. Never a second claim, never an instruction. Drop it when it adds nothing.
- Bullets state cause and effect, one grammar per list. Three or four short items, plain `itemize` sets them at slide size.
- `\hl{}` at most one phrase per slide.
- Cards: two to four, one heading each, two or three short points as `itemize` below it, never a paragraph.
- One big number: `\hero{18\texttimes}{what it measures}` in a `[c]` frame. Two or three: `keyfigures`. `\keyfigure[0.49]{49\%}{label}` adds a bar against the full track, give the reference in one `\subtle` line.
- One equation: `keyequation`, it earns a slide only when every term is named with `\term`. Two or three related ones: `equations` with one `\eq{name}{math}` row each. `\eqhl` marks the one term in focus.
- Tables: `datatable{X r r}` with booktabs, an `X` column takes the free width. Rows are options sorted best first, columns are metrics, the subtitle says which direction is better. `\bestrow` tints the winner row, `\best{}` marks one value in accent. Ratings are `\harvey{0}` to `\harvey{4}` cells, a decision table ends in a `Pick it for` row.
- Slides with one visual or one short list use `\begin{frame}[c]`, so it sits centred under the title. Longer text slides stay top aligned.
- Figures: same view, same scale, one caption each. Overlays step through the same figure: one `\roundimage`, changing notes in `overprint`.
- Images of any aspect ratio fit: `\roundimage` caps them at `\linewidth` wide and `\gcfigheight` tall. `\fullbleed[source]{file}{caption}` in a `[plain]` frame covers the page, the caption sits on a paper band.
- Evidence: `\claim{headline}{text}` in a narrow column beside the figure, table or chart in the wide one. Before and after is the same frame with two equal columns, an `\eyebrow{Before}` and `\eyebrow{After}` over each, the changed element marked with `\mk` on both sides. With a label above and a line below, cap the images at `\roundimage[height=0.5\textheight]{file}`.
- Charts: pgfplots with `\begin{axis}[gc chart]`. Label lines at their end (`node[gc label]{name}`), never a legend, `[gc focus]` gives the one series in focus the accent. Unit and direction go in the subtitle. `ymode=log` works.
- Bars: `\begin{axis}[gc bars,symbolic y coords={...}]` for a few named values, coordinates are `(value,name)`, sorted best first. The value stands at the bar's end, `\addplot[gc focus]` is the one bar in accent. `xmode=log` with `xmin=1` works.
- Sources: whatever is taken from elsewhere (figure, data, quote) carries `\source{Author Year, Fig. n}` right under it. The `references` frame before the closing statement lists each one in full with `\reference{authors (year)}{title. venue.}`, up to six per slide.
- Steps read left to right on one rail and advance one click per step, `\pause` before them is fine. `[2]` freezes step 2 as current, `[0]` is a plain pipeline. Up to six steps, short titles.
- Timelines are Gantt charts: `gantt[now]{units}`, one row per `\task{first}{last}{name}` or `\milestone{unit}{name}`, in order of start. The subtitle names the time unit, `now` is the time elapsed and marks today. Up to seven rows.
- A thesis, a quote or the closing line gets its own `[plain,c]` frame with `\statement[source]{sentence}`, the source is optional.
- Text helpers: `\muted{}` for supporting sentences, `\subtle{}` for caveats, `\source{}` for sources, `\eyebrow{}` for small labels.
- Beamer's `block`, `exampleblock` and `alertblock` render as cards and take overlays (`\begin{block}<2->{Title}`), `alertblock` carries the accent border.
- Speaker notes: every content frame ends with `\note{...}`, two to four plain sentences. Notes carry the talk, slides carry the evidence: say why it matters, never read the slide aloud. Notes become hidden PDF annotations for the presenter view and never print.

## Color levels

| Level | Colors | Use |
|---|---|---|
| Ink (text) | `ink`, `inkmuted`, `inksubtle`, `inkfaint` | body, supporting text, captions, sources and data marks (at least 4.5:1 on paper), decoration only (`inkfaint`, never data or text to read) |
| Paper (backgrounds) | `paper`, `surface`, `surfaceelevated` | slide background (darkest), level 1 cards, level 2 cards (brightest) |
| Accent | `accent` | the one thing in focus: highlighted phrase, best value, winner row, focus series, current step, today mark |

Two levels on a slide: paper and surfaceelevated. Use surface only when nesting needs a third. Never introduce new colors in a deck, the accent is set once in the sty.

## Primitives

| Id | Use when |
|---|---|
| `text/bullets` | a few cause and effect points |
| `text/cards` | two to four parallel ideas |
| `text/statement` | a thesis or a quote on its own slide |
| `text/references` | the full entry for every source named on the slides |
| `numbers/hero` | one big number carries the slide |
| `numbers/keyfigures` | two or three numbers, optional bars against a reference |
| `numbers/keyequation` | one equation with every term named |
| `numbers/equations` | two or three related equations as named rows |
| `tables/datatable` | options against metrics, the winner row tinted |
| `tables/decision` | options rated against criteria, ending in a decision |
| `figures/fullbleed` | one image covers the slide, a short claim on it |
| `figures/evidence` | a claim beside the figure, table or chart that proves it, also before and after |
| `figures/annotated` | numbered markers on a figure with notes |
| `figures/chart` | a pgfplots chart with direct labels and one series in focus |
| `figures/bars` | a few named values as horizontal bars, one in focus |
| `figures/figrow` | equal images side by side |
| `process/steps` | a sequence or pipeline on a numbered rail |
| `process/gantt` | a timeline of tasks and milestones as a Gantt chart |

Title, agenda and closing frames live in `skeleton.tex`.

## Theme interface

One convention for every macro in the sty, so a frame reads the same to a human and an agent:

- **Primitive**: an environment, one item macro per entry inside it: `keyfigures`/`\keyfigure`, `equations`/`\eq`, `figrow`/`\fig`, `steps`/`\step`, `gantt`/`\task` and `\milestone`, `annotated`/`\annot`, `references`/`\reference`. Each item macro takes its parts as arguments, text never goes into a comma list. Tables are plain `tabular` rows inside `datatable`, charts plain pgfplots inside `axis`.
- **Layouts**: plain beamer, `\begin{columns}`, `overprint`, `block`. The theme adds no wrapper around them.
- **Inline helpers**: `\hl`, `\mk`, `\muted`, `\subtle`, `\source`, `\eyebrow`, `\claim`, `\hd`, `\best`, `\bestrow`, `\harvey`, `\term`, `\eqhl`, `\roundimage`, `\fullbleed`, `\hero`, `\statement`, all with an optional first argument where one exists. pgfplots styles `gc chart`, `gc bars`, `gc focus`, `gc label`.
- **Tunable**: lengths `\gcmargin`, `\gcgap`, `\gcarc`, `\gcfigheight`, `\gccaptionwidth`, `\gccolgap`, `\gcprogresswidth`, `\gcganttlabel`, `\gcganttrow`, sizes as beamer fonts (`\setbeamerfont{key figure}{size=...}`, also `hero`, `statement`, `lead`, `claim`, `equation`, `equation row`, `card title`, `card text`, `eyebrow`, `itemize/enumerate body`), colors as listed above.
- **Internal**: everything else starts with `\gc@` and is not used in frames.
