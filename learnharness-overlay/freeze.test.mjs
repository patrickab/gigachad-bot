import assert from "node:assert/strict";
import test from "node:test";
import { createJiti } from "jiti";

const jiti = createJiti(import.meta.url, { tsconfigPaths: true });
const { freezeSessionPicker, sessionModelUnavailable, SESSION_PICKER_EVENTS } = await jiti.import("./embed/freeze.ts");

const available = async () => false;
const removed = async () => true;
/** Lets the freeze's model check and its dispatch run. */
const settle = () => new Promise((resolve) => setTimeout(resolve, 0));

/** A window dispatching events at itself like a browser at the target: capture listeners first, then the others. */
function fakeWindow(pathname) {
  const listeners = [];
  const win = {
    location: { pathname },
    addEventListener: (type, listener, options) => listeners.push({ type, listener, capture: Boolean(options?.capture) }),
    removeEventListener: (type, listener, options) => {
      const index = listeners.findIndex((entry) => entry.type === type && entry.listener === listener && entry.capture === Boolean(options?.capture));
      if (index >= 0) listeners.splice(index, 1);
    },
    dispatch(type) {
      win.dispatchEvent(new CustomEvent(type));
    },
    dispatchEvent(event) {
      let stopped = false;
      event.stopImmediatePropagation = () => (stopped = true);
      const ordered = [...listeners.filter((entry) => entry.capture), ...listeners.filter((entry) => !entry.capture)];
      for (const entry of ordered) {
        if (stopped) break;
        if (entry.type === event.type) entry.listener(event);
      }
      return true;
    },
  };
  return win;
}

/** Upstream's model control listening first, the way it mounts before the overlay's effect runs. */
function upstreamControl(win) {
  const heard = [];
  for (const type of [...SESSION_PICKER_EVENTS, "lh:kbd-toggle-sidebar"]) win.addEventListener(type, () => heard.push(type));
  return heard;
}

test("an open session's model menu and thinking cycle never reach upstream's control", async () => {
  const win = fakeWindow("/chat/abc123");
  const heard = upstreamControl(win);
  freezeSessionPicker(win, available);

  win.dispatch("lh:kbd-open-model-picker");
  win.dispatch("lh:kbd-cycle-thinking");
  win.dispatch("lh:kbd-toggle-sidebar");
  await settle();
  assert.deepEqual(heard, ["lh:kbd-toggle-sidebar"]);
});

test("a session whose model is gone gets the model menu, upstream's way to pick a served one", async () => {
  const win = fakeWindow("/chat/abc123");
  const heard = upstreamControl(win);
  const checked = [];
  freezeSessionPicker(win, async (sessionId) => (checked.push(sessionId), true));

  win.dispatch("lh:kbd-open-model-picker");
  win.dispatch("lh:kbd-cycle-thinking");
  assert.deepEqual(heard, []);
  await settle();
  assert.deepEqual(heard, ["lh:kbd-open-model-picker"]);
  assert.deepEqual(checked, ["abc123"]);
});

test("a session left before its model check answered gets no menu", async () => {
  const win = fakeWindow("/chat/abc123");
  const heard = upstreamControl(win);
  freezeSessionPicker(win, removed);

  win.dispatch("lh:kbd-open-model-picker");
  win.location.pathname = "/chat/other";
  await settle();
  assert.deepEqual(heard, []);
});

test("the draft surfaces keep the keyboard, their pick sets the strong tier for the next session", () => {
  const win = fakeWindow("/");
  const heard = upstreamControl(win);
  freezeSessionPicker(win, removed);

  win.dispatch("lh:kbd-open-model-picker");
  win.location.pathname = "/chat/new";
  win.dispatch("lh:kbd-cycle-thinking");
  win.location.pathname = "/chat/s1";
  win.dispatch("lh:kbd-cycle-thinking");
  assert.deepEqual(heard, ["lh:kbd-open-model-picker", "lh:kbd-cycle-thinking"]);
});

test("stopping the freeze hands the keyboard back", () => {
  const win = fakeWindow("/chat/abc123");
  const heard = upstreamControl(win);
  const stop = freezeSessionPicker(win, removed);
  stop();

  win.dispatch("lh:kbd-open-model-picker");
  assert.deepEqual(heard, ["lh:kbd-open-model-picker"]);
});

/** Drive mode as the freeze sees it, recording what it hands the shell as `model-open`. */
function shellDrive(active = true) {
  const opened = [];
  return { opened, drive: { active: () => active, openModel: (sessionId) => opened.push(sessionId) } };
}

test("in drive mode the model menu never opens in the frame, a draft surface asks the shell at once", async () => {
  const win = fakeWindow("/chat/new");
  const heard = upstreamControl(win);
  const { opened, drive } = shellDrive();
  freezeSessionPicker(win, removed, drive);

  win.dispatch("lh:kbd-open-model-picker");
  win.location.pathname = "/";
  win.dispatch("lh:kbd-open-model-picker");
  win.dispatch("lh:kbd-cycle-thinking");
  await settle();
  assert.deepEqual(opened, [null, null]);
  // The draft's thinking cycle stays upstream's: the hidden control writes the strong tier, the shell sees the `model` post.
  assert.deepEqual(heard, ["lh:kbd-cycle-thinking"]);
});

test("in drive mode a session asks the shell only once its model is confirmed gone", async () => {
  const win = fakeWindow("/chat/abc123");
  const heard = upstreamControl(win);
  const served = shellDrive();
  const stop = freezeSessionPicker(win, available, served.drive);
  win.dispatch("lh:kbd-open-model-picker");
  win.dispatch("lh:kbd-cycle-thinking");
  await settle();
  assert.deepEqual(served.opened, []);
  stop();

  const gone = shellDrive();
  freezeSessionPicker(win, removed, gone.drive);
  win.dispatch("lh:kbd-open-model-picker");
  await settle();
  assert.deepEqual(gone.opened, ["abc123"]);
  assert.deepEqual(heard, [], "the hidden control's menu is never released");
});

test("without drive mode the draft surfaces keep upstream's menu", () => {
  const win = fakeWindow("/chat/new");
  const heard = upstreamControl(win);
  const { opened, drive } = shellDrive(false);
  freezeSessionPicker(win, removed, drive);

  win.dispatch("lh:kbd-open-model-picker");
  assert.deepEqual(heard, ["lh:kbd-open-model-picker"]);
  assert.deepEqual(opened, []);
});

/** Answers LearnHarness's reads from *routes* (path -> JSON body, an Error fails the request). */
function serve(routes) {
  globalThis.fetch = async (path) => {
    const body = routes[path];
    if (body === undefined) return { ok: false, status: 404, json: async () => ({}) };
    if (body instanceof Error) throw body;
    return { ok: true, status: 200, json: async () => body };
  };
}

const CATALOG = { models: [{ provider: "custom", id: "gemini/gemini-3.5-flash" }] };

test("a session's model is unavailable only when a good catalog read lacks it", async (t) => {
  const original = globalThis.fetch;
  t.after(() => (globalThis.fetch = original));
  const stamped = (id) => ({ settings: { model: { provider: "custom", id } } });

  serve({ "/api/chats/s1": stamped("gemini/gemini-3.5-flash"), "/api/models": CATALOG });
  assert.equal(await sessionModelUnavailable("s1"), false);

  serve({ "/api/chats/s1": stamped("litellm_proxy/anthropic/claude-gone"), "/api/models": CATALOG });
  assert.equal(await sessionModelUnavailable("s1"), true);

  serve({ "/api/chats/s1": stamped("litellm_proxy/anthropic/claude-gone"), "/api/models": new Error("offline") });
  assert.equal(await sessionModelUnavailable("s1"), false);

  serve({ "/api/models": CATALOG });
  assert.equal(await sessionModelUnavailable("s1"), false);
});

test("a legacy unstamped session runs on the strong tier", async (t) => {
  const original = globalThis.fetch;
  t.after(() => (globalThis.fetch = original));
  const strong = (id) => ({ settings: { models: { strong: { model: { provider: "custom", id } } } } });

  serve({ "/api/chats/s1": { settings: {} }, "/api/models": CATALOG, "/api/settings": strong("ollama/gone") });
  assert.equal(await sessionModelUnavailable("s1"), true);

  serve({ "/api/chats/s1": {}, "/api/models": CATALOG, "/api/settings": strong("gemini/gemini-3.5-flash") });
  assert.equal(await sessionModelUnavailable("s1"), false);
});
