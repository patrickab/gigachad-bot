import assert from "node:assert/strict";
import "../tests/setup-dom.mjs";
import test, { afterEach } from "node:test";
import React from "react";
import { cleanup, fireEvent, render } from "@testing-library/react/pure.js";
import { createJiti } from "jiti";

// jsdom has no matchMedia, upstream's `useTheme` (the code blocks' color scheme) reads the OS scheme from it.
window.matchMedia ??= () => ({ matches: false, addEventListener() {}, removeEventListener() {} });

const jiti = createJiti(import.meta.url, { jsx: { runtime: "automatic" }, tsconfigPaths: true });
const { EmbeddedMarkdown } = await jiti.import("./components/EmbeddedMarkdown.tsx");
const { AgentLinkContext } = await jiti.import("../lib/agent-links.ts");
afterEach(cleanup);

test("shared renderer renders prose and LaTex while preserving source code", () => {
  const view = render(React.createElement(EmbeddedMarkdown, null, "**Result** $x^2$\n\n\\[y = 2\\]\n\n`$not_math$`"));
  assert.equal(view.container.querySelector("strong").textContent, "Result");
  assert.equal(view.container.querySelectorAll(".katex").length, 2);
  assert.equal(view.container.querySelector("code").textContent, "$not_math$");
});

test("shared renderer preserves OMP file and agent link actions without executing hostile URLs", () => {
  const openedFiles = [];
  const openedAgents = [];
  const view = render(React.createElement(AgentLinkContext.Provider, { value: (ids) => openedAgents.push(ids) },
    React.createElement(EmbeddedMarkdown, { cwd: "/tmp/project", onOpenFile: (path) => openedFiles.push(path) },
      "[File](./example.ts) and [Agent](agent://Scout) and [unsafe](javascript:alert%281%29)")));
  fireEvent.click(view.getByRole("link", { name: "File" }));
  assert.deepEqual(openedFiles, ["/tmp/project/example.ts"]);
  fireEvent.click(view.getByRole("link", { name: "Agent" }));
  assert.deepEqual(openedAgents, [["Scout"]]);
  assert.equal(view.container.querySelector('a[href^="javascript:"]'), null);
});

test("linked images never produce nested interactive lightbox buttons inside anchors", () => {
  const view = render(React.createElement(EmbeddedMarkdown, null, "[**![Preview](https://example.com/image.png)**](https://example.com)"));
  assert.equal(view.container.querySelector("a button"), null);
  assert.ok(view.getByRole("img", { name: "Preview" }));
});

test("a link still streaming in renders as text, not as a navigable anchor", () => {
  const view = render(React.createElement(EmbeddedMarkdown, { isStreaming: true }, "See [the docs](https://exa"));
  assert.equal(view.container.querySelector("a"), null);
  assert.match(view.container.textContent, /the docs/);
});

test("tables keep the package's element and gain upstream's chart only with autoCharts", () => {
  const table = "| Model | Score |\n|---|---|\n| A | 1 |\n| B | 3 |";
  const plain = render(React.createElement(EmbeddedMarkdown, null, table));
  assert.ok(plain.container.querySelector(".gc-markdown-table-wrap > table"));
  assert.equal(plain.container.querySelector("[data-table-chart]"), null);
  cleanup();
  const charted = render(React.createElement(EmbeddedMarkdown, { autoCharts: true }, table));
  assert.ok(charted.container.querySelector("[data-table-chart]"));
  assert.equal(charted.container.querySelector("table").getAttribute("node"), null);
});

test("inline code keeps hast metadata off the DOM", () => {
  const view = render(React.createElement(EmbeddedMarkdown, null, "Run `npm test` now"));
  const code = view.container.querySelector("code");
  assert.equal(code.textContent, "npm test");
  assert.equal(code.getAttribute("node"), null);
});

test("fenced code is the package's code block in omp-web's color scheme", () => {
  const view = render(React.createElement(EmbeddedMarkdown, null, "```ts\nconst x = 1\n```"));
  const block = view.container.querySelector(".gc-code");
  assert.ok(block);
  assert.equal(block.getAttribute("data-code-language"), "ts");
  assert.match(block.getAttribute("data-theme"), /^(dark|light)$/);
  assert.ok(view.getByRole("button", { name: "Copy code" }));
});

test("mermaid fences render as the package's diagram, not upstream's source and Preview toggle", () => {
  const view = render(React.createElement(EmbeddedMarkdown, null, "```mermaid\ngraph TD\n  A --> B\n```"));
  assert.ok(view.container.querySelector(".gc-diagram-pending, .gc-diagram"));
  assert.equal(view.container.querySelector(".gc-code"), null);
  assert.equal(view.queryByRole("button", { name: /preview/i }), null);
});
