import assert from "node:assert/strict";
import "../tests/setup-dom.mjs";
import test, { afterEach } from "node:test";
import React from "react";
import { cleanup, render, screen } from "@testing-library/react/pure.js";
import { createJiti } from "jiti";

// `findTranscriptEnd` checks `instanceof HTMLDivElement`, which the shared DOM setup does not expose.
globalThis.HTMLDivElement = window.HTMLDivElement;
const jiti = createJiti(import.meta.url, { jsx: { runtime: "automatic" }, tsconfigPaths: true });
const { ComposerPanels } = await jiti.import("./overrides/components/ComposerPanels.tsx");
const { EMBED_CLASS } = await jiti.import("./components/EmbedMode.tsx");

afterEach(() => {
  cleanup();
  document.body.replaceChildren();
  document.documentElement.classList.remove(EMBED_CLASS);
});

/** Chat column with upstream's bare end-of-transcript marker, plus the mount point for the panels. */
function chatColumn() {
  const column = document.createElement("div");
  column.innerHTML = '<div role="log"><div><div><p>message</p><div></div></div></div></div>';
  const mount = column.appendChild(document.createElement("div"));
  document.body.append(column);
  return mount;
}

const panels = (status) => React.createElement(ComposerPanels, {
  todoPhases: [],
  subagents: [{ id: "a1", agent: "task", status, task: "Complete assignment thoroughly", index: 0 }],
  onSelectSubagent: () => {},
  defaultExpanded: true,
});

test("embedded subagent chips keep their label but lose the native tooltip, also after a status change", async () => {
  document.documentElement.classList.add(EMBED_CLASS);
  const view = render(panels("started"), { container: chatColumn() });
  const chip = screen.getByRole("button", { name: /Complete assignment thoroughly/ });
  assert.ok(chip.closest(".cf-session-panels"));
  assert.equal(document.querySelector(".cf-session-panels [title]"), null);
  view.rerender(panels("completed"));
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(document.querySelector(".cf-session-panels [title]"), null);
});

test("standalone subagent chips keep upstream's tooltip", () => {
  const view = render(panels("started"), { container: chatColumn() });
  assert.match(view.getByRole("button", { name: /Complete assignment thoroughly/ }).title, /Complete assignment thoroughly/);
});
