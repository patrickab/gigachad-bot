/** Class on <html> while the app runs inside a frame, the hook for `custom/theme/embed.css`. */
export const EMBED_CLASS = "cf-embed";

function isFramed(): boolean {
  try {
    return window.self !== window.top;
  } catch {
    // A cross-origin parent denies access to window.top: framed.
    return true;
  }
}

// Marked when the overlay loads: app/client.tsx imports it before the first render, so the hidden chrome never flashes.
// A rendered <script> would not run, React inserts it after parsing.
if (typeof document !== "undefined" && isFramed()) document.documentElement.classList.add(EMBED_CLASS);

/** True inside a frame, i.e. inside gigachad-bot's shell. */
export function isEmbedded(): boolean {
  return typeof document !== "undefined" && document.documentElement.classList.contains(EMBED_CLASS);
}
