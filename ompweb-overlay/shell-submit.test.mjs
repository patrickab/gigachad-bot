import assert from "node:assert/strict";
import "../tests/setup-dom.mjs";
import test from "node:test";
import { createJiti } from "jiti";

const jiti = createJiti(import.meta.url, { jsx: { runtime: "automatic" }, tsconfigPaths: true });
const { submitShellMessage } = await jiti.import("./embed/submit.ts");

const image = { data: "aGk=", mimeType: "image/png", previewUrl: "data:image/png;base64,aGk=" };

/** Composer props recording every call; `overrides` replace the defaults. */
function composer(overrides = {}) {
  const calls = [];
  const record = (name, value) => (...args) => {
    calls.push([name, ...args]);
    return Promise.resolve(value);
  };
  return {
    calls,
    props: {
      isStreaming: false,
      onSend: record("send", { status: "accepted" }),
      onSteer: record("steer", true),
      onFollowUp: record("followUp", true),
      onBuiltinCommand: record("builtin", { handled: false }),
      ...overrides,
    },
  };
}

test("idle, a prompt is sent and kept only when omp-web rejected it", async () => {
  const sent = composer();
  assert.equal(await submitShellMessage(sent.props, " hello ", []), "cleared");
  assert.deepEqual(sent.calls, [["send", "hello", undefined]]);

  const rejected = composer({ onSend: async () => ({ status: "rejected" }) });
  assert.equal(await submitShellMessage(rejected.props, "hello", [image]), "kept");
  assert.equal(await submitShellMessage(composer({ onSend: async () => ({ status: "superseded" }) }).props, "hello", []), "cleared");
});

test("during a run a message is queued as a follow-up, and a refused queue keeps it", async () => {
  const busy = composer({ isStreaming: true });
  assert.equal(await submitShellMessage(busy.props, "next", [image]), "cleared");
  assert.deepEqual(busy.calls, [["followUp", "next", [image]]]);

  const refused = composer({ isStreaming: true, onFollowUp: async () => false });
  assert.equal(await submitShellMessage(refused.props, "next", []), "kept");
});

test("during a run a !command runs at once instead of being queued", async () => {
  const busy = composer({ isStreaming: true });
  assert.equal(await submitShellMessage(busy.props, "!ls", []), "cleared");
  assert.deepEqual(busy.calls, [["send", "!ls", undefined]]);
});

test("a handled builtin is cleared unless it failed or asked to keep the input, and is never sent as a prompt", async () => {
  for (const [result, outcome] of [[{ handled: true }, "cleared"], [{ handled: true, error: "nope" }, "kept"], [{ handled: true, retainInput: true }, "kept"]]) {
    const builtin = composer({ onBuiltinCommand: async () => result });
    assert.equal(await submitShellMessage(builtin.props, "/compact", []), outcome);
    assert.deepEqual(builtin.calls, []);
  }
  const unknown = composer();
  assert.equal(await submitShellMessage(unknown.props, "/unknown", []), "cleared");
  assert.deepEqual(unknown.calls.map(([name]) => name), ["builtin", "send"]);
});

test("a /btw with images and a blocked composer keep the message without sending", async () => {
  const btw = composer();
  assert.equal(await submitShellMessage(btw.props, "/btw why", [image]), "kept");
  const blocked = composer({ submissionBlocked: true });
  assert.equal(await submitShellMessage(blocked.props, "hello", []), "kept");
  assert.deepEqual([...btw.calls, ...blocked.calls], []);
});
