import assert from "node:assert/strict";
import test from "node:test";
import { createJiti } from "jiti";

const jiti = createJiti(import.meta.url, { tsconfigPaths: true, jsx: true });
const { applyModelPick, applyThinkingPick, modelState, servedModels } = await jiti.import("./embed/model.tsx");

const FLASH = { provider: "custom", id: "gemini/gemini-3.5-flash", supportedLevels: ["off", "low", "high"], contextWindow: 1_000_000 };
const SONNET = { provider: "custom", id: "litellm_proxy/anthropic/claude-sonnet-4-5", supportedLevels: ["off", "medium", "high"], contextWindow: 200_000 };
const CATALOG = { models: [FLASH, SONNET] };
const ref = (entry) => ({ provider: entry.provider, id: entry.id });
const MODELS = {
  strong: { model: ref(FLASH), thinkingLevel: "low" },
  cheap: { model: ref(SONNET) },
  vision: { model: ref(FLASH), thinkingLevel: "off" },
};

/** LearnHarness's API for one test: GET *routes* (path -> body, Error = unreachable), every PUT answered *putStatus*. */
function serve(t, routes, putStatus = 200) {
  const original = globalThis.fetch;
  const originalWindow = globalThis.window;
  const puts = [];
  const signals = [];
  globalThis.window = new EventTarget();
  globalThis.window.addEventListener("lh:settings-saved", (event) => signals.push(event.detail.section));
  globalThis.fetch = async (path, init = {}) => {
    if (init.method === "PUT") {
      puts.push({ path, body: JSON.parse(init.body) });
      return { ok: putStatus < 400, status: putStatus, json: async () => ({}) };
    }
    const body = routes[path];
    if (body === undefined) return { ok: false, status: 404, json: async () => ({}) };
    if (body instanceof Error) throw body;
    return { ok: true, status: 200, json: async () => structuredClone(body) };
  };
  t.after(() => {
    globalThis.fetch = original;
    globalThis.window = originalWindow;
  });
  return { puts, signals };
}

const available = async () => false;
const unavailable = async () => true;

test("a draft shows the strong tier, unlocked, with its model's levels", () => {
  assert.deepEqual(modelState(null, MODELS, undefined, CATALOG), {
    type: "model",
    sessionId: null,
    model: { provider: "custom", modelId: FLASH.id },
    thinkingLevel: "low",
    thinkingLevels: ["off", "low", "high"],
    locked: false,
    unavailable: false,
  });
  const unpicked = modelState(null, undefined, undefined, CATALOG);
  assert.equal(unpicked.model, null);
  assert.equal(unpicked.thinkingLevel, null);
});

test("a session shows its own stamped model, locked while it is served", () => {
  const session = { settings: { model: ref(SONNET) } };
  const state = modelState("s1", MODELS, session, CATALOG);
  assert.deepEqual(state.model, { provider: "custom", modelId: SONNET.id });
  // No stored level on a stamped model: the strong tier's level rides, upstream's display rule.
  assert.equal(state.thinkingLevel, "low");
  assert.equal(state.locked, true);
  assert.equal(state.unavailable, false);
  assert.equal(modelState("s1", MODELS, undefined, CATALOG), null, "nothing is posted before the session record is read");
});

test("a legacy unstamped session runs on the strong tier, an unserved model unlocks the session", () => {
  assert.deepEqual(modelState("s1", MODELS, {}, CATALOG).model, { provider: "custom", modelId: FLASH.id });
  const gone = modelState("s1", MODELS, { settings: { model: { provider: "custom", id: "ollama/gone" } } }, CATALOG);
  assert.equal(gone.unavailable, true);
  assert.equal(gone.locked, false);
  assert.deepEqual(gone.thinkingLevels, []);
  // An unread catalog is not proof of a gone model (upstream's send guard defers too).
  assert.equal(modelState("s1", MODELS, { settings: { model: { provider: "custom", id: "ollama/gone" } } }, undefined).locked, true);
});

test("the served list is the catalog's provider and id", () => {
  assert.deepEqual(servedModels(CATALOG), [
    { provider: "custom", modelId: FLASH.id },
    { provider: "custom", modelId: SONNET.id },
  ]);
});

test("a draft pick writes the strong tier upstream's way: siblings kept, the stored level rides", async (t) => {
  const { puts, signals } = serve(t, { "/api/models": CATALOG, "/api/settings": { settings: { models: MODELS } } });
  const result = await applyModelPick({ type: "set-model", sessionId: null, provider: "custom", modelId: SONNET.id }, available);
  assert.equal(result, "written");
  assert.deepEqual(puts, [
    { path: "/api/settings/models", body: { ...MODELS, strong: { model: ref(SONNET), thinkingLevel: "low" } } },
  ]);
  assert.deepEqual(signals, ["models"]);
});

test("a draft pick without a stored level populates the model's default order level", async (t) => {
  const { puts } = serve(t, { "/api/models": CATALOG, "/api/settings": { settings: null } });
  await applyModelPick({ type: "set-model", sessionId: null, provider: "custom", modelId: SONNET.id }, available);
  assert.equal(puts[0].body.strong.thinkingLevel, "medium");
});

test("a pick of a model the proxy does not serve is dropped", async (t) => {
  const { puts } = serve(t, { "/api/models": CATALOG, "/api/settings": { settings: { models: MODELS } } });
  const result = await applyModelPick({ type: "set-model", sessionId: null, provider: "custom", modelId: "ollama/gone" }, available);
  assert.equal(result, "ignored");
  assert.deepEqual(puts, []);
});

test("a session pick is dropped while the session's model is served", async (t) => {
  const { puts } = serve(t, { "/api/models": CATALOG, "/api/settings": { settings: { models: MODELS } }, "/api/chats/s1": { settings: { model: ref(FLASH) } } });
  const checked = [];
  const result = await applyModelPick(
    { type: "set-model", sessionId: "s1", provider: "custom", modelId: SONNET.id },
    async (sessionId) => (checked.push(sessionId), false),
  );
  assert.equal(result, "ignored");
  assert.deepEqual(checked, ["s1"]);
  assert.deepEqual(puts, []);
});

test("a session whose model is gone takes a served pick through upstream's session write", async (t) => {
  const { puts, signals } = serve(t, {
    "/api/models": CATALOG,
    "/api/settings": { settings: { models: MODELS } },
    "/api/chats/s%201": { settings: { model: { provider: "custom", id: "ollama/gone" }, thinkingLevel: "high" } },
  });
  const result = await applyModelPick({ type: "set-model", sessionId: "s 1", provider: "custom", modelId: SONNET.id }, unavailable);
  assert.equal(result, "written");
  assert.deepEqual(puts, [{ path: "/api/chats/s%201/settings", body: { model: ref(SONNET), thinkingLevel: "high" } }]);
  assert.deepEqual(signals, ["models"], "the hidden control and the send guard re-read the session");
});

test("a refused or unreachable write reports a failure and signals nothing", async (t) => {
  const { signals } = serve(t, { "/api/models": CATALOG, "/api/settings": { settings: { models: MODELS } } }, 400);
  assert.equal(await applyModelPick({ type: "set-model", sessionId: null, provider: "custom", modelId: SONNET.id }, available), "failed");
  assert.deepEqual(signals, []);

  serve(t, { "/api/models": new Error("offline") });
  assert.equal(await applyModelPick({ type: "set-model", sessionId: null, provider: "custom", modelId: SONNET.id }, available), "failed");
});

test("a draft thinking pick sets the strong tier's level, one its model supports", async (t) => {
  const { puts, signals } = serve(t, { "/api/models": CATALOG, "/api/settings": { settings: { models: MODELS } } });
  assert.equal(await applyThinkingPick({ type: "set-thinking", sessionId: null, level: "high" }), "written");
  assert.deepEqual(puts, [{ path: "/api/settings/models", body: { ...MODELS, strong: { model: ref(FLASH), thinkingLevel: "high" } } }]);
  assert.deepEqual(signals, ["models"]);

  assert.equal(await applyThinkingPick({ type: "set-thinking", sessionId: null, level: "medium" }), "ignored");
  assert.equal(await applyThinkingPick({ type: "set-thinking", sessionId: "s1", level: "high" }), "ignored");
  assert.equal(puts.length, 1);
});

test("the model-picker override keeps every upstream export", async () => {
  const upstream = Object.keys(await jiti.import("../components/model-picker.tsx"));
  const override = Object.keys(await jiti.import("./overrides/components/model-picker.tsx"));
  assert.deepEqual(upstream.filter((name) => !override.includes(name)), []);
});
