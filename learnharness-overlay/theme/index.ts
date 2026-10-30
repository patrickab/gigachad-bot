// Overlay theme, imported once by root.tsx.
//
// - Colours apply only while embedded (`html.cf-embed`), standalone LearnHarness keeps upstream's look
// - palette.css: role colours from the shell's palette, tokens.css: LearnHarness's colour tokens from them
// - fonts.css, type.css, radius.css, prose.css: gigachad's fonts, type and radius ladders, markdown and chat look
// - chrome.css: upstream's sidebar and title bar hidden while the shell drives (`html.cf-shell-drive`)
// - settings.css: where LearnHarness takes provider keys (Providers section, setup key stage), hidden embedded or not
// - The overlay's own CSS is imported from here, never via CSS @import: Turbopack does not track @import'ed files, edits
//   would not rebuild under `next dev`. Package stylesheets (`@gigachad/chat-ui/fonts.css` in fonts.css,
//   `markdown.css` in prose.css) may be @import'ed: they change only with a new pack, which the launcher's build stamp covers
import "@gigachad/chat-ui/palette.css";
import "./palette.css";
import "./tokens.css";
import "./fonts.css";
import "./type.css";
import "./radius.css";
import "./prose.css";
import "./chrome.css";
import "./settings.css";
