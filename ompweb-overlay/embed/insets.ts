/**
 * The shell lays its top bar and composer over the frame, the transcript scrolls beneath them like a gigachad chat.
 *
 * - `--cf-inset-top` / `--cf-inset-bottom`: shell chrome in px (`insets` message)
 * - `--cf-composer-area`: height of what is left of omp-web's composer area (queue, status, Tasks/Subagents), which
 *   `theme/embed.css` floats above the shell composer, so the transcript pads past it too
 */

/** omp-web's composer area: the transcript column's sibling holding the hidden input box (`components/ChatWindow.tsx`). */
const COMPOSER_AREA = ":has(> [role=log]) ~ div:has(.chat-input-shell)";
const AREA_POLL_MS = 500;

function setPixels(name: string, pixels: number) {
  document.documentElement.style.setProperty(name, `${Math.max(0, Math.round(pixels))}px`);
}

export function applyShellInsets({ top, bottom }: { top: number; bottom: number }) {
  setPixels("--cf-inset-top", top);
  setPixels("--cf-inset-bottom", bottom);
}

/** Keep `--cf-composer-area` at the composer area's height, re-finding it when omp-web remounts the chat. Returns the stop. */
export function trackComposerArea(): () => void {
  let area: Element | null = null;
  const observer = new ResizeObserver(([entry]) => setPixels("--cf-composer-area", entry?.borderBoxSize[0]?.blockSize ?? 0));
  const refind = () => {
    const next = document.querySelector(COMPOSER_AREA);
    if (next === area) return;
    if (area) observer.unobserve(area);
    area = next;
    if (area) observer.observe(area);
    else setPixels("--cf-composer-area", 0);
  };
  refind();
  const poll = window.setInterval(refind, AREA_POLL_MS);
  return () => {
    window.clearInterval(poll);
    observer.disconnect();
  };
}
