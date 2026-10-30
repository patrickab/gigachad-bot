import assert from "node:assert/strict";
import { copyFileSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { createJiti } from "jiti";

const jiti = createJiti(import.meta.url, { tsconfigPaths: true });
const { createShellDispatcher, openLearnChannel, parseShellMessage } = await jiti.import("./embed/channel.ts");
const { DRIVE_CLASS, isShellDrive, setShellDrive } = await jiti.import("./embed/drive.ts");
const { LEARN_PROTOCOL_VERSION } = await jiti.import("@gigachad/chat-ui/learn-protocol");
const { pathFor, sessionIdOf } = await jiti.import("./embed/routes.ts");
const { leaveHiddenSection } = await jiti.import("./embed/settings.ts");
const { applyTheme, THEME_STORAGE_KEY } = await jiti.import("./embed/theme.ts");
const { overrideAliases, withCustom } = await jiti.import("./next.config.ts");
const { frameAncestorOrigins } = await jiti.import("./framing.ts");

const SHELL_ORIGIN = "http://127.0.0.1:2999";
const SHELL_ORIGINS = [SHELL_ORIGIN, "http://localhost:2999"];

/** A window whose parent records what the frame posts, and a way to deliver messages to the frame. */
function fakeWindow() {
  const listeners = new Set();
  const posted = [];
  const parent = { postMessage: (message, targetOrigin) => posted.push({ message, targetOrigin }) };
  const win = {
    parent,
    addEventListener: (_type, listener) => listeners.add(listener),
    removeEventListener: (_type, listener) => listeners.delete(listener),
  };
  const deliver = (data, { source = parent, origin = SHELL_ORIGIN } = {}) => listeners.forEach((listener) => listener({ data, source, origin }));
  return { win, posted, deliver, listeners };
}

test("the frame announces itself and its version to any origin and answers the shell only after hello, at the shell's origin", () => {
  const { win, posted, deliver } = fakeWindow();
  const channel = openLearnChannel(win, SHELL_ORIGINS);
  assert.deepEqual(posted, [{ message: { type: "ready", version: LEARN_PROTOCOL_VERSION, channel: "learn-embed" }, targetOrigin: "*" }]);

  channel.post({ type: "view", sessionId: "abc" });
  assert.equal(posted.length, 1);

  deliver({ channel: "learn-embed", type: "hello", theme: "light", version: LEARN_PROTOCOL_VERSION });
  assert.deepEqual(posted.at(-1), { message: { type: "view", sessionId: "abc", channel: "learn-embed" }, targetOrigin: SHELL_ORIGIN });

  channel.post({ type: "view", sessionId: null });
  assert.deepEqual(posted.at(-1), { message: { type: "view", sessionId: null, channel: "learn-embed" }, targetOrigin: SHELL_ORIGIN });
});

test("only the parent on the learn channel and at the pinned origin reaches the handlers", () => {
  const { win, deliver, listeners } = fakeWindow();
  const channel = openLearnChannel(win, SHELL_ORIGINS);
  const themes = [];
  channel.on("hello", ({ theme }) => themes.push(`hello:${theme}`));
  const unsubscribe = channel.on("set-theme", ({ theme }) => themes.push(theme));

  deliver({ channel: "learn-embed", type: "set-theme", theme: "dark" });
  deliver({ channel: "learn-embed", type: "hello", theme: "light" }, { source: { postMessage() {} } });
  deliver({ channel: "omp-embed", type: "hello", theme: "light" });
  assert.deepEqual(themes, []);

  deliver({ channel: "learn-embed", type: "hello", theme: "light" });
  deliver({ channel: "learn-embed", type: "set-theme", theme: "dark" }, { origin: "http://evil.test" });
  deliver({ channel: "learn-embed", type: "set-theme", theme: "dark" });
  assert.deepEqual(themes, ["hello:light", "dark"]);

  unsubscribe();
  deliver({ channel: "learn-embed", type: "set-theme", theme: "light" });
  assert.deepEqual(themes, ["hello:light", "dark"]);

  channel.close();
  assert.equal(listeners.size, 0);
});

test("a hello from an origin outside the configured shell origins is never answered", () => {
  const { win, posted, deliver } = fakeWindow();
  const channel = openLearnChannel(win, SHELL_ORIGINS);
  const navigations = [];
  channel.on("open-home", () => navigations.push("home"));
  channel.post({ type: "view", sessionId: "abc" });

  deliver({ channel: "learn-embed", type: "hello", theme: "light" }, { origin: "http://evil.test" });
  deliver({ channel: "learn-embed", type: "open-home" }, { origin: "http://evil.test" });
  assert.deepEqual(navigations, []);
  assert.equal(posted.length, 1);

  const unconfigured = fakeWindow();
  const closed = openLearnChannel(unconfigured.win, []);
  closed.on("open-home", () => navigations.push("home"));
  unconfigured.deliver({ channel: "learn-embed", type: "hello", theme: "light" });
  unconfigured.deliver({ channel: "learn-embed", type: "open-home" });
  assert.deepEqual(navigations, []);
});

test("malformed shell messages are dropped before any handler", () => {
  const { win, deliver } = fakeWindow();
  const channel = openLearnChannel(win, SHELL_ORIGINS);
  const seen = [];
  for (const type of ["hello", "set-theme", "open-session", "new-session", "open-home"]) channel.on(type, (message) => seen.push(message));

  deliver({ channel: "learn-embed", type: "hello" });
  deliver({ channel: "learn-embed", type: "hello", theme: "blue" });
  assert.deepEqual(seen, []);
  deliver({ channel: "learn-embed", type: "hello", theme: "dark", extra: 1 });
  deliver({ channel: "learn-embed", type: "open-session", sessionId: 42 });
  deliver({ channel: "learn-embed", type: "open-session", sessionId: "" });
  deliver({ channel: "learn-embed", type: "open-session" });
  deliver({ channel: "learn-embed", type: "set-theme", theme: null });
  deliver({ channel: "learn-embed", type: "unknown" });
  deliver(null);
  deliver("learn-embed");
  deliver({ channel: "learn-embed", type: "open-session", sessionId: "k3x9" });
  assert.deepEqual(seen, [{ type: "hello", theme: "dark", version: 1 }, { type: "open-session", sessionId: "k3x9" }]);
  assert.equal(parseShellMessage({ channel: "omp-embed", type: "open-home" }), null);
  assert.deepEqual(parseShellMessage({ channel: "learn-embed", type: "new-session" }), { type: "new-session", kind: "teach" });
});

test("every drive message is validated, stripped to its fields and malformed ones dropped", () => {
  const on = (message) => parseShellMessage({ channel: "learn-embed", ...message });
  const valid = [
    { type: "hello", theme: "light", version: 3 },
    { type: "new-session", kind: "practice" },
    { type: "set-model", sessionId: null, provider: "custom", modelId: "gemini/gemini-3.5-flash" },
    { type: "set-thinking", sessionId: "s1", level: "high" },
    { type: "open-settings", section: null },
    { type: "open-settings", section: "agent" },
    { type: "open-panel", panel: "plan", sessionId: "s1" },
    { type: "open-panel", panel: "coverage", sessionId: "s1" },
    { type: "open-panel", panel: "activity", sessionId: null },
    { type: "close-modal" },
  ];
  for (const message of valid) assert.deepEqual(on({ ...message, extra: true }), message);

  const malformed = [
    { type: "hello", theme: "light", version: 0 },
    { type: "hello", theme: "light", version: "2" },
    { type: "new-session", kind: "lecture" },
    { type: "set-model", sessionId: null, provider: "custom" },
    { type: "set-model", sessionId: null, provider: "", modelId: "m" },
    { type: "set-thinking", sessionId: null, level: "" },
    { type: "open-settings" },
    { type: "open-settings", section: 3 },
    { type: "open-panel", panel: "plan", sessionId: null },
    { type: "open-panel", panel: "activity", sessionId: "s1" },
    { type: "open-panel", panel: "toString", sessionId: null },
    { type: "open-panel", panel: "stats", sessionId: "s1" },
    { type: "hasOwnProperty" },
  ];
  for (const message of malformed) assert.equal(on(message), null, JSON.stringify(message));
});

test("drive messages reach the handlers only from a shell at this protocol version, theme and navigation at any", () => {
  const { win, deliver } = fakeWindow();
  const dispatcher = createShellDispatcher();
  openLearnChannel(win, SHELL_ORIGINS, dispatcher);
  const seen = [];
  for (const type of ["hello", "open-home", "close-modal", "open-settings"]) dispatcher.on(type, (message) => seen.push(message.type));

  deliver({ channel: "learn-embed", type: "close-modal" });
  deliver({ channel: "learn-embed", type: "hello", theme: "light" });
  deliver({ channel: "learn-embed", type: "close-modal" });
  deliver({ channel: "learn-embed", type: "open-home" });
  assert.deepEqual(seen, ["hello", "open-home"]);

  deliver({ channel: "learn-embed", type: "hello", theme: "light", version: LEARN_PROTOCOL_VERSION + 1 });
  deliver({ channel: "learn-embed", type: "open-settings", section: null });
  assert.deepEqual(seen, ["hello", "open-home", "hello"]);

  deliver({ channel: "learn-embed", type: "hello", theme: "light", version: LEARN_PROTOCOL_VERSION });
  deliver({ channel: "learn-embed", type: "close-modal" });
  deliver({ channel: "learn-embed", type: "open-settings", section: null });
  assert.deepEqual(seen, ["hello", "open-home", "hello", "hello", "close-modal", "open-settings"]);
});

test("every hello replays the latest state per type and session meta per session, never events", () => {
  const { win, posted, deliver } = fakeWindow();
  const channel = openLearnChannel(win, SHELL_ORIGINS);
  const meta = (sessionId, title) => ({ type: "session-meta", sessionId, title, kind: "teach", phase: null });
  channel.post({ type: "modal", open: true });
  channel.post({ type: "modal", open: false });
  channel.post(meta("s1", "Limits"));
  channel.post(meta("s2", "Series"));
  channel.post(meta("s1", "Limits, part 2"));
  channel.post({ type: "model-open", sessionId: null });

  const replayed = () => posted.filter(({ targetOrigin }) => targetOrigin === SHELL_ORIGIN).map(({ message: { channel: _, ...message } }) => message);
  deliver({ channel: "learn-embed", type: "hello", theme: "dark", version: LEARN_PROTOCOL_VERSION });
  assert.deepEqual(replayed(), [{ type: "modal", open: false }, meta("s1", "Limits, part 2"), meta("s2", "Series")]);

  posted.length = 0;
  channel.post({ type: "model-open", sessionId: "s1" });
  deliver({ channel: "learn-embed", type: "hello", theme: "dark", version: LEARN_PROTOCOL_VERSION });
  assert.deepEqual(replayed(), [{ type: "model-open", sessionId: "s1" }, { type: "modal", open: false }, meta("s1", "Limits, part 2"), meta("s2", "Series")]);
});

test("drive mode is the cf-shell-drive class on <html>, and listeners see it change", () => {
  const classes = new Set();
  const root = { classList: { toggle: (token, force) => (force ? classes.add(token) : classes.delete(token)) } };
  setShellDrive(true, root);
  assert.equal(isShellDrive(), true);
  assert.ok(classes.has(DRIVE_CLASS));
  setShellDrive(false, root);
  assert.equal(isShellDrive(), false);
  assert.equal(classes.has(DRIVE_CLASS), false);
});

test("a session page reports its id, every other page null", () => {
  assert.equal(sessionIdOf("/chat/k3x9_A-1"), "k3x9_A-1");
  assert.equal(sessionIdOf("/chat/k3x9/"), "k3x9");
  assert.equal(sessionIdOf("/chat/new"), null);
  assert.equal(sessionIdOf("/"), null);
  assert.equal(sessionIdOf("/graph"), null);
  assert.equal(sessionIdOf("/chat/k3x9/extra"), null);
  assert.equal(sessionIdOf("/chat/%E0%A4%A"), null);
  assert.equal(sessionIdOf("/chat/a%2Fb"), "a/b");
});

test("navigation messages open the session, a teaching or practice draft, or home", () => {
  assert.equal(pathFor({ type: "open-session", sessionId: "k3x9" }), "/chat/k3x9");
  assert.equal(pathFor({ type: "open-session", sessionId: "a/b" }), "/chat/a%2Fb");
  assert.equal(pathFor({ type: "new-session", kind: "teach" }), "/chat/new");
  assert.equal(pathFor({ type: "new-session", kind: "practice" }), "/chat/new?mode=practice");
  assert.equal(pathFor({ type: "open-home" }), "/");
});

test("the settings dialog leaves the hidden Providers section for Agent", () => {
  const clicks = [];
  const documentWith = (present) => ({
    querySelector: (selector) => (present.includes(selector) ? { click: () => clicks.push(selector) } : null),
  });
  const providersCurrent = '[data-testid="settings-nav-keys"][aria-current="true"]';
  const agent = '[data-testid="settings-nav-agent"]';

  assert.equal(leaveHiddenSection(documentWith([providersCurrent, agent])), true);
  assert.deepEqual(clicks, [agent]);
  assert.equal(leaveHiddenSection(documentWith([agent])), false);
  assert.equal(leaveHiddenSection(documentWith([])), false);
  assert.deepEqual(clicks, [agent]);
});

test("the shell theme toggles upstream's dark class and is remembered for the next load", () => {
  const classes = new Set(["dark"]);
  const root = { classList: { toggle: (token, force) => (force ? classes.add(token) : classes.delete(token)) } };
  const stored = new Map();
  const storage = { setItem: (key, value) => stored.set(key, value) };

  applyTheme("light", root, storage);
  assert.equal(classes.has("dark"), false);
  assert.equal(stored.get(THEME_STORAGE_KEY), "light");
  applyTheme("dark", root, storage);
  assert.equal(classes.has("dark"), true);
  assert.equal(stored.get(THEME_STORAGE_KEY), "dark");
  applyTheme("light", root, { setItem: () => { throw new Error("quota"); } });
  assert.equal(classes.has("dark"), false);
});

test("an override answers the bare and the suffixed @/ imports and the bare sibling import", () => {
  const target = "./custom/overrides/components/model-picker";
  assert.deepEqual(overrideAliases(["components/model-picker"]), {
    "@/components/model-picker": target,
    "@/components/model-picker.tsx": target,
    "@/components/model-picker.ts": target,
    "./model-picker": target,
  });
  assert.deepEqual(overrideAliases([]), {});
});

test("withCustom finds the overrides beside it, whatever directory the build runs from", async () => {
  const project = mkdtempSync(join(tmpdir(), "learnharness-overlay-"));
  const previousDirectory = process.cwd();
  try {
    mkdirSync(join(project, "custom", "overrides", "components"), { recursive: true });
    for (const file of ["next.config.ts", "framing.ts"]) copyFileSync(new URL(file, import.meta.url), join(project, "custom", file));
    writeFileSync(join(project, "custom", "overrides", "components", "model-picker.tsx"), "export {};\n");
    process.chdir(tmpdir());
    const { withCustom: servedWithCustom } = await jiti.import(join(project, "custom", "next.config.ts"));
    const config = servedWithCustom({ turbopack: { resolveAlias: { kept: "./kept" } } }, undefined, []);
    assert.equal(config.turbopack.resolveAlias["@/components/model-picker"], "./custom/overrides/components/model-picker");
    assert.equal(config.turbopack.resolveAlias.kept, "./kept");
  } finally {
    process.chdir(previousDirectory);
    rmSync(project, { recursive: true, force: true });
  }
});

test("only the configured shell origins may frame LearnHarness, and the bridge learns them at build time", async () => {
  const upstreamRule = { source: "/api/:path*", headers: [{ key: "Cache-Control", value: "no-store" }] };
  const config = withCustom({ env: { LH_VERSION: "0.4.0" }, headers: async () => [upstreamRule] }, "/nonexistent", SHELL_ORIGINS);
  assert.deepEqual(await config.headers(), [
    upstreamRule,
    { source: "/:path*", headers: [{ key: "Content-Security-Policy", value: `frame-ancestors 'self' ${SHELL_ORIGINS.join(" ")}` }] },
  ]);
  assert.deepEqual(config.env, { LH_VERSION: "0.4.0", GIGACHAD_LEARNHARNESS_FRAME_ANCESTORS: SHELL_ORIGINS.join(" ") });
  const unframed = withCustom({}, "/nonexistent", []);
  assert.deepEqual(await unframed.headers(), [{ source: "/:path*", headers: [{ key: "Content-Security-Policy", value: "frame-ancestors 'self'" }] }]);

  assert.deepEqual(frameAncestorOrigins("http://127.0.0.1:2999, https://gigachad.tail.ts.net"), ["http://127.0.0.1:2999", "https://gigachad.tail.ts.net"]);
  assert.deepEqual(frameAncestorOrigins(""), []);
  assert.throws(() => frameAncestorOrigins("http://127.0.0.1:2999/path"), /GIGACHAD_LEARNHARNESS_FRAME_ANCESTORS/);
  assert.throws(() => frameAncestorOrigins("*"), /not an origin/);
});

test("every place LearnHarness takes a provider key is hidden", () => {
  const css = readFileSync(new URL("./theme/settings.css", import.meta.url), "utf8").replace(/\/\*[\s\S]*?\*\//g, "");
  const hidden = [...css.matchAll(/([^{}]+)\{([^}]*)\}/g)]
    .filter(([, , body]) => /display:\s*none\s*!important/.test(body))
    .flatMap(([, selectors]) => selectors.split(",").map((selector) => selector.trim()));
  for (const testId of ["settings-nav-keys", "section-keys", "setup-stage-keys", "setup-cta-keys"]) {
    assert.ok(hidden.includes(`[data-testid="${testId}"]`), testId);
  }
});
