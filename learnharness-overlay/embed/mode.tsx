import { THEME_STORAGE_KEY } from "./theme";

/** Class on <html> while LearnHarness runs inside a frame, the hook for every embed-only rule in `custom/theme/`. */
export const EMBED_CLASS = "cf-embed";

const MARK_FRAMED_SCRIPT = `(function(){var framed;try{framed=window.self!==window.top}catch(e){framed=true}if(!framed)return;var root=document.documentElement;root.classList.add("${EMBED_CLASS}");try{if(localStorage.getItem("${THEME_STORAGE_KEY}")==="light")root.classList.remove("dark")}catch(e){}})();`;

/** True inside a frame, i.e. inside gigachad-bot's shell. */
export function isEmbedded(): boolean {
  return typeof document !== "undefined" && document.documentElement.classList.contains(EMBED_CLASS);
}

/** Mark <html> as embedded during parsing, before the page content and hydration, so upstream's look never flashes. */
export function EmbedMode() {
  return <script dangerouslySetInnerHTML={{ __html: MARK_FRAMED_SCRIPT }} />;
}
