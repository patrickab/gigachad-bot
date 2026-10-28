import assert from "node:assert/strict";
import "../tests/setup-dom.mjs";
import test, { afterEach } from "node:test";
import React from "react";
import { cleanup, fireEvent, render } from "@testing-library/react/pure.js";
import { createJiti } from "jiti";

const jiti = createJiti(import.meta.url, { jsx: { runtime: "automatic" }, tsconfigPaths: true });
const { QaTurn } = await jiti.import("./components/QaTurn.tsx");
const { EMBED_CLASS } = await jiti.import("./components/EmbedMode.tsx");
const { summarizeTurn } = await jiti.import("./components/turn-summary.ts");
const { MessageView } = await jiti.import("../components/MessageView.tsx");
const { planTranscriptRows } = await jiti.import("./overrides/lib/chat-transcript-plan.ts");
const { renderTranscriptGroupRow } = await jiti.import("./overrides/components/ChatWindow-transcript-group-row.tsx");

const setEmbedded = (embedded) => document.documentElement.classList.toggle(EMBED_CLASS, embedded);
afterEach(() => {
  cleanup();
  setEmbedded(false);
});

const messages = [
  { role: "user", content: "Inspect **this** file", timestamp: 1000 },
  { role: "assistant", content: [
    { type: "thinking", thinking: "Private reasoning" },
    { type: "text", text: "First **finding**." },
    { type: "toolCall", toolName: "read", toolCallId: "read-1", input: { path: "example.ts" } },
  ] },
  { role: "toolResult", toolCallId: "read-1", content: [{ type: "text", text: "Raw tool output" }] },
  { role: "assistant", content: [
    { type: "toolCall", toolName: "read", toolCallId: "read-2", input: { path: "other.ts" } },
    { type: "text", text: "Final answer: $x^2$." },
  ] },
  { role: "user", content: "Next question", timestamp: 2000 },
];
const plain = [messages[0], { role: "assistant", content: [{ type: "text", text: "Answer" }] }, messages[4]];

function card(overrides = {}) {
  return React.createElement(QaTurn, {
    messages, userIndex: 0, endIndex: 4, live: false,
    question: React.createElement(MessageView, { message: messages[0] }),
    ...overrides,
  }, React.createElement(MessageView, { message: messages[3] }));
}

function groupContext(transcript, live = false) {
  const renderMessage = (idx, options = {}) => React.createElement("div", { key: `${options.keyPrefix ?? "message"}-${idx}`, "data-rendered": idx }, `message ${idx}`);
  return { messages: transcript, renderMessage, live, messageRefs: { current: [] }, visibleRefIndexByMessage: new Map() };
}

test("embedded, plain question/answer turns stay indivisible at history page boundaries", () => {
  setEmbedded(true);
  const rows = planTranscriptRows(plain);
  assert.deepEqual(rows.map(({ kind, userIndex, endIndex }) => ({ kind, userIndex, endIndex })), [
    { kind: "group", userIndex: 0, endIndex: 2 },
    { kind: "group", userIndex: 2, endIndex: 3 },
  ]);
  assert.deepEqual(rows[0].segments.map(({ kind, index }) => ({ kind, index })), [{ kind: "text", index: 1 }]);
});

test("standalone, plain turns keep upstream's message rows", () => {
  assert.deepEqual(planTranscriptRows(plain), [0, 1, 2].map((index) => ({ kind: "message", index })));
});

test("embedded, a typed prompt's turn renders as one card around upstream's nodes", () => {
  setEmbedded(true);
  const [row] = planTranscriptRows(plain);
  const nodes = renderTranscriptGroupRow(row, groupContext(plain));
  assert.equal(nodes.length, 1);
  const view = render(React.createElement(React.Fragment, null, nodes));
  const turn = view.container.querySelector(".gc-turn");
  assert.equal(turn.dataset.expanded, "false");
  assert.deepEqual([...view.container.querySelectorAll("[data-message-index]")].map((el) => el.dataset.messageIndex), ["0", "1"]);
  fireEvent.click(view.getByRole("button", { name: /Inspect/ }));
  assert.deepEqual([...view.container.querySelectorAll("[data-rendered]")].map((el) => el.dataset.rendered), ["0", "1"]);
});

test("embedded, a skill invocation's turn is a card asking what the user typed, without the skill body", () => {
  setEmbedded(true);
  const skillTurn = [
    { role: "custom", customType: "skill-prompt", display: true, content: "SKILL BODY instructions", details: { name: "teach", args: "diffusion" }, timestamp: 3000 },
    ...messages.slice(1, 4),
  ];
  const [row] = planTranscriptRows(skillTurn);
  assert.equal(row.kind, "group");
  const nodes = renderTranscriptGroupRow(row, groupContext(skillTurn));
  assert.equal(nodes.length, 1);
  const view = render(React.createElement(React.Fragment, null, nodes));
  assert.equal(view.container.querySelector(".gc-turn").dataset.expanded, "true");
  assert.ok(view.getAllByText("/skill:teach diffusion").length > 0);
  assert.equal(view.queryByText(/SKILL BODY/), null);
  assert.equal(summarizeTurn(skillTurn, 0, 4).question, "/skill:teach diffusion");
});

test("embedded, a skill invoked after a turn with tools starts its own turn instead of folding into it", () => {
  setEmbedded(true);
  const skill = { role: "custom", customType: "skill-prompt", display: true, content: "SKILL BODY", details: { name: "teach" }, timestamp: 3000 };
  const transcript = [...messages.slice(0, 4), skill, ...messages.slice(1, 4)];
  const turns = planTranscriptRows(transcript).map(({ kind, userIndex, endIndex }) => ({ kind, userIndex, endIndex }));
  assert.deepEqual(turns, [{ kind: "group", userIndex: 0, endIndex: 4 }, { kind: "group", userIndex: 4, endIndex: 8 }]);
});

test("standalone or a non-prompt anchor, the group row renders exactly as upstream", () => {
  const [row] = planTranscriptRows(messages);
  const standalone = renderTranscriptGroupRow(row, groupContext(messages));
  assert.ok(standalone.length > 1);
  assert.equal(standalone[0].key, "message-0");
  setEmbedded(true);
  const summary = [{ ...messages[0], branchSummary: true }, ...messages.slice(1)];
  assert.equal(renderTranscriptGroupRow(row, groupContext(summary))[0].key, "message-0");
});

test("turn copying excludes reasoning and tool results, retaining answer order", () => {
  assert.deepEqual(summarizeTurn(messages, 0, 4), {
    question: "Inspect **this** file", label: "Inspect **this** file",
    answer: "First **finding**.\n\nFinal answer: $x^2$.",
    endsWithText: true,
  });
});

test("embedded, a compaction anchors a card in context order: what it saved asks, the kept tail and the continuation answer", () => {
  setEmbedded(true);
  const compaction = { role: "custom", customType: "compaction", display: true, timestamp: 5000, details: { tokensBefore: 85000, tokensAfter: 44000 },
    content: "## Goal\n\nDark mode\n\n<read-files>\na.ts\n</read-files>" };
  const kept = { role: "assistant", content: [{ type: "text", text: "Kept before compaction" }], timestamp: 4000 };
  const continued = { role: "assistant", content: [{ type: "text", text: "Continued after" }], timestamp: 6000 };
  const transcript = [compaction, kept, continued, messages[4]];
  const [row] = planTranscriptRows(transcript);
  assert.deepEqual({ kind: row.kind, userIndex: row.userIndex, endIndex: row.endIndex }, { kind: "group", userIndex: 0, endIndex: 3 });
  assert.deepEqual(summarizeTurn(transcript, 0, 3), {
    question: "## Goal\n\nDark mode", label: "Conversation compacted · 85k → 44k tokens",
    answer: "Kept before compaction\n\nContinued after", endsWithText: true,
  });
});

test("embedded, a collapsed turn previews upstream's answer nodes, tools folded into its Process details row", () => {
  setEmbedded(true);
  window.history.replaceState(null, "", "/?session=qa-preview");
  const [row] = planTranscriptRows(messages);
  const renderMessage = (idx, options = {}) => React.createElement(MessageView, { key: `${options.keyPrefix ?? "message"}-${idx}`, message: options.messageOverride ?? messages[idx] });
  const view = render(React.createElement(React.Fragment, null, renderTranscriptGroupRow(row, { ...groupContext(messages), renderMessage })));
  assert.equal(view.container.querySelector(".gc-turn").dataset.expanded, "false");
  const preview = view.getByRole("region", { name: "Answer preview" });
  assert.match(preview.querySelector(".process-details-toggle").textContent, /Process details/);
  assert.equal(view.queryByRole("list", { name: "Tools used" }), null);
});

test("preview interactions do not expand a turn, and explicit expansion survives remounting per session", () => {
  window.history.replaceState(null, "", "/?session=qa-regression");
  let view = render(card());
  const preview = view.getByRole("region", { name: "Answer preview" });
  fireEvent.click(preview);
  fireEvent.scroll(preview, { target: { scrollTop: 40 } });
  assert.equal(view.container.querySelector(".gc-turn").dataset.expanded, "false");
  assert.ok(view.getByText(/Final answer/));
  fireEvent.click(view.getByRole("button", { name: /Inspect/ }));
  assert.equal(view.container.querySelector(".gc-turn").dataset.expanded, "true");
  assert.ok(view.getByRole("button", { name: "Copy question as markdown" }));
  assert.ok(view.getByRole("button", { name: "Copy answer as markdown" }));
  view.unmount();
  view = render(card());
  assert.equal(view.container.querySelector(".gc-turn").dataset.expanded, "true");
  view.unmount();
  window.history.replaceState(null, "", "/?session=other-session");
  view = render(card());
  assert.equal(view.container.querySelector(".gc-turn").dataset.expanded, "false");
  view.unmount();
  window.history.replaceState(null, "", "/?session=qa-regression");
  view = render(card());
  fireEvent.click(view.getAllByRole("button", { name: "Collapse question and answer" })[0]);
  assert.equal(view.container.querySelector(".gc-turn").dataset.expanded, "false");
});

test("a live turn cannot be collapsed and has no card actions even when previously collapsed", () => {
  const view = render(card({ live: true }));
  assert.equal(view.container.querySelector(".gc-turn").dataset.expanded, "true");
  assert.deepEqual(view.queryAllByRole("button", { name: "Collapse question and answer" }), []);
  assert.equal(view.queryByRole("button", { name: /Copy .* as markdown/ }), null);
});
