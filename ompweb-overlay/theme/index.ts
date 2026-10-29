// Overlay theme, imported once by root.tsx, which app/layout.tsx imports after globals.css.
//
// - Every rule applies only while embedded (`html.cf-embed`), standalone omp-web keeps upstream's look
// - palette.css: role colors, tokens.css: upstream tokens from them, then one file per concern
// - Import CSS from here, never via CSS @import: Turbopack does not track @import'ed files, edits would not rebuild
// - `@gigachad/chat-ui` is gigachad-bot's shared UI: Q&A cards, prose, code and diagrams, click hints, and the `--gc-*` role colors palette.css maps
import "katex/dist/katex.min.css";
import "@gigachad/chat-ui/markdown.css";
import "@gigachad/chat-ui/code.css";
import "@gigachad/chat-ui/turn.css";
import "@gigachad/chat-ui/hints.css";
import "@gigachad/chat-ui/palette.css";
import "./fonts.css";
import "./palette.css";
import "./tokens.css";
import "./base.css";
import "./chat-ui.css";
import "./embed.css";
