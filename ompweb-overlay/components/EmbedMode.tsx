/** Class on <html> while the app runs inside a frame, the hook for `custom/theme/embed.css`. */
export const EMBED_CLASS = "cf-embed";

const MARK_FRAMED_SCRIPT = `(function(){var framed;try{framed=window.self!==window.top}catch(e){framed=true}if(framed)document.documentElement.classList.add("${EMBED_CLASS}")})();`;

/** True inside a frame, i.e. inside gigachad-bot's shell. */
export function isEmbedded(): boolean {
  return typeof document !== "undefined" && document.documentElement.classList.contains(EMBED_CLASS);
}

/** Mark <html> as embedded during parsing, before hydration, so the hidden chrome never flashes. */
export function EmbedMode() {
  return <script dangerouslySetInnerHTML={{ __html: MARK_FRAMED_SCRIPT }} />;
}
