import assert from "node:assert/strict";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { createJiti } from "jiti";
import { parseHTML } from "linkedom";
import React from "react";
import { createRoot } from "react-dom/client";
import { renderToStaticMarkup } from "react-dom/server";

/** The LearnHarness checkout this overlay lands in as `custom/`. */
const PROJECT_ROOT = fileURLToPath(new URL("..", import.meta.url));
const THEME = fileURLToPath(new URL("./theme/", import.meta.url));
const readUpstream = (path) => readFileSync(join(PROJECT_ROOT, path), "utf8");

// A DOM for the client render, `<html class="cf-embed">` as `embed/mode.tsx` marks a framed page.
const { window, document } = parseHTML('<!doctype html><html class="cf-embed"><body></body></html>');
Object.assign(globalThis, { window, document, IS_REACT_ACT_ENVIRONMENT: true });

const jiti = createJiti(import.meta.url, { jsx: { runtime: "automatic" }, tsconfigPaths: true });
const upstream = await jiti.import("../components/ai-elements/message.tsx");
const overlay = await jiti.import("./overrides/components/ai-elements/message.tsx");

/** Upstream source files whose Tailwind classes the frame shows. */
function sourceFiles(dir) {
  return readdirSync(join(PROJECT_ROOT, dir), { recursive: true, encoding: "utf8" })
    .filter((file) => /\.(tsx?|jsx?)$/.test(file))
    .map((file) => join(dir, file));
}

const SIZE_RUNG = /^text-(xs|sm|base|lg|[2-9]?xl|\[\d+(\.\d+)?px\])$/;
const RADIUS_RUNG = /^rounded(-(t|r|b|l|s|e|tl|tr|br|bl|ss|se|es|ee))?(-(xs|sm|md|lg|[2-9]?xl|\[\d+(\.\d+)?px\]))?$/;
// Variants as specific as a plain class (responsive, pseudo-elements): the base rung's mapping would beat them, so each needs its own.
const TYING_VARIANT = /^(sm|md|lg|xl|2xl|file|before|after|placeholder|marker|selection|first-letter|first-line|backdrop)$/;

/** Size and radius classes in upstream's sources: the rungs, and the variants that need their own mapping. */
function usedClasses() {
  const rungs = new Set();
  const tying = new Set();
  for (const file of [...sourceFiles("components"), ...sourceFiles("app")]) {
    for (const token of readUpstream(file).split(/[\s"'`{}(),]+/)) {
      const candidate = token.replace(/^!/, "");
      const colon = candidate.lastIndexOf(":");
      const base = candidate.slice(colon + 1);
      if (!SIZE_RUNG.test(base) && !RADIUS_RUNG.test(base)) continue;
      rungs.add(base);
      const variants = colon < 0 ? [] : candidate.slice(0, colon).split(":");
      if (variants.length > 0 && variants.every((variant) => TYING_VARIANT.test(variant))) tying.add(candidate);
    }
  }
  return { rungs, tying };
}

/** Class names a stylesheet has a selector for, unescaped (`.text-\[10px\]` → `text-[10px]`). */
function mappedClasses(...files) {
  const css = files.map((file) => readFileSync(join(THEME, file), "utf8").replace(/\/\*[\s\S]*?\*\//g, "")).join("\n");
  return new Set([...css.matchAll(/\.((?:\\.|[\w-])+)/g)].map(([, name]) => name.replace(/\\(.)/g, "$1")));
}

test("every size and radius class upstream uses has gigachad's value in type.css or radius.css", () => {
  const { rungs, tying } = usedClasses();
  const mapped = mappedClasses("type.css", "radius.css");
  assert.ok(rungs.has("text-sm") && rungs.has("rounded-md"), "the scan finds upstream's classes");
  assert.deepEqual([...rungs].filter((name) => !mapped.has(name)), [], "rungs without a mapping");
  assert.deepEqual([...tying].filter((name) => !mapped.has(name)), [], "responsive or pseudo-element variants without a mapping");
});

test("the upstream knobs the theme re-points still exist", () => {
  const layout = readUpstream("app/layout.tsx");
  assert.match(layout, /JetBrains_Mono\(\{[^}]*variable: "--font-jetbrains"/, "fonts.css names the JetBrains Mono face for code");
  assert.match(layout, /variable: "--font-sans3"/);
  const globals = readUpstream("app/globals.css");
  assert.match(globals, /\.lh-prose \{[^}]*font-family: var\(--font-prose\);[^}]*font-size: var\(--lh-prose-size\);[^}]*line-height: var\(--lh-prose-leading\)/);
  assert.match(globals, /--font-ui: var\(--font-jetbrains\)/);
  assert.match(globals, /--font-title: /);
  assert.match(globals, /\[data-streamdown="code-block"\]/);
  assert.match(readUpstream("components/chat.tsx"), /data-testid="user-message"/);
});

test("the package stylesheets the theme imports resolve, with the font files fonts.css points at", () => {
  const require = createRequire(join(PROJECT_ROOT, "package.json"));
  for (const [file, specifier] of [["fonts.css", "@gigachad/chat-ui/fonts.css"], ["prose.css", "@gigachad/chat-ui/markdown.css"]]) {
    assert.match(readFileSync(join(THEME, file), "utf8"), new RegExp(`@import "${specifier}";`));
    const path = require.resolve(specifier);
    if (file !== "fonts.css") continue;
    for (const [, url] of readFileSync(path, "utf8").matchAll(/url\("([^"]+)"\)/g)) assert.ok(existsSync(join(dirname(path), url)), url);
  }
});

test("the markdown override keeps every upstream export", () => {
  assert.deepEqual(Object.keys(upstream).filter((name) => !(name in overlay)), []);
});

test("standalone, markdown renders exactly as upstream's", () => {
  for (const name of ["MessageResponse", "ToolContentResponse"]) {
    const render = (module) => renderToStaticMarkup(React.createElement(module[name], { mode: "static" }, "**Result** $x^2$\n\n- one"));
    assert.equal(render(overlay), render(upstream), name);
  }
});

test("embedded, every markdown response sits in gigachad's markdown root without changing the caller's layout", async () => {
  for (const name of ["MessageResponse", "ToolContentResponse"]) {
    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);
    await React.act(async () => root.render(React.createElement(overlay[name], { mode: "static", className: "inline" }, "**Result** \\(x^2\\)")));
    const wrapper = container.firstElementChild;
    assert.equal(wrapper.className, "gc-markdown", name);
    assert.equal(wrapper.getAttribute("data-markdown-root"), "", name);
    assert.match(wrapper.getAttribute("style"), /display:\s*contents/, name);
    const response = wrapper.firstElementChild;
    assert.match(response.className, /\blh-prose\b.*\binline\b/, `${name}: upstream's Streamdown root with the caller's class`);
    assert.equal(response.querySelector("strong").textContent, "Result");
    assert.ok(response.querySelector(".katex"), `${name}: upstream's math pipeline`);
    await React.act(async () => root.unmount());
    container.remove();
  }
});
