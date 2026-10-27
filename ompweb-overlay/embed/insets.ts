/**
 * The shell lays its top bar and composer over the frame, the transcript scrolls beneath them like a gigachad chat.
 *
 * - `--cf-inset-top` / `--cf-inset-bottom`: shell chrome in px (`insets` message)
 * - `--cf-composer-area`: height of what is left of omp-web's composer area (queue, status, Tasks/Subagents), which
 *   `theme/embed.css` floats above the shell composer, so the transcript pads past it too
 * - A session opens at its newest message, whatever settles in after upstream's first scroll
 */

/** omp-web's composer area: the transcript column's sibling holding the hidden input box (`components/ChatWindow.tsx`). */
const COMPOSER_AREA = ":has(> [role=log]) ~ div:has(.chat-input-shell)";
/** omp-web's conversation scroller (`components/ChatWindow.tsx`). */
const TRANSCRIPT = '[data-selection-scope="chat"][role=log]';
/** Reader input in the transcript: from then on upstream's own follow logic decides where it stays. */
const READER_INPUT_EVENTS = ["wheel", "touchstart", "pointerdown", "keydown"] as const;
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

function openSessionId(): string | null {
  return new URLSearchParams(window.location.search).get("session");
}

/**
 * Hold each opened session's transcript at its end until the reader scrolls, clicks or types in it. Returns the stop.
 *
 * - Upstream scrolls to the end once on open. Layout that lands later (shell insets, the composer area, Markdown,
 *   diagrams, the Tasks/Subagents cards) would leave the newest message below the shell composer
 * - Starts over for a new transcript element or another `?session=`
 */
export function settleTranscriptAtBottom(): () => void {
  let transcript: HTMLElement | null = null;
  let sessionId: string | null = null;
  let settling = false;
  const holdAtEnd = () => {
    if (settling && transcript) transcript.scrollTop = transcript.scrollHeight;
  };
  const stopSettling = () => {
    settling = false;
  };
  const observer = new ResizeObserver(holdAtEnd);
  const release = () => {
    observer.disconnect();
    READER_INPUT_EVENTS.forEach((type) => transcript?.removeEventListener(type, stopSettling));
  };
  const refind = () => {
    const next = document.querySelector<HTMLElement>(TRANSCRIPT);
    if (next === transcript && openSessionId() === sessionId) return;
    release();
    transcript = next;
    sessionId = openSessionId();
    settling = transcript !== null;
    if (!transcript) return;
    READER_INPUT_EVENTS.forEach((type) => transcript?.addEventListener(type, stopSettling, { passive: true }));
    observer.observe(transcript);
    if (transcript.firstElementChild) observer.observe(transcript.firstElementChild);
    holdAtEnd();
  };
  refind();
  const poll = window.setInterval(refind, AREA_POLL_MS);
  return () => {
    window.clearInterval(poll);
    release();
  };
}
