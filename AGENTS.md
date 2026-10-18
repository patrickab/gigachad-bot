## Never start the dev server autonomously

The backend (`uvicorn`, port 8001) and frontend (`next dev`, port 2999) are
started via `run.sh` or manually by the user, in a terminal they control.

- Never run `run.sh`, `uvicorn`, `npm run dev`, or equivalent yourself —
  even in the background, even to verify a change.
- If you need to see the app running (UI check, manual test, screenshot),
  ask the user to start it and confirm it's up, then use their already-running
  instance. Do not launch your own copy alongside theirs.

## UI hints

No hover tooltips, except on rendered markdown.

## UI style

Clean, calm UI that lets the user focus on the content (chat, document, canvas).

- **Content first**: the work surface gets the space. Chrome stays small and quiet.
- **Minimal**: no decoration without a function. Remove an element before adding one.
- **Low noise**: muted colors, no heavy borders, no badges or animation for their own sake.
- **Progressive disclosure**: secondary controls appear on hover or focus, and common actions stay one click away.
- **Consistent**: new controls copy the shape, size, colors and states of their neighbors.
- **Stable**: adding a control must not shift or resize existing ones.
- **Shared primitives**: reuse the components in `src/frontend/components/ui/` before writing custom markup. Extend a primitive if it falls short, rather than forking a one-off.
- **Accessible**: icon-only buttons have an `aria-label`. Keyboard focus is always visible.

