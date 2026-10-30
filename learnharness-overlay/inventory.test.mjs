// Upstream-feature inventory: every upstream input, event, prop, control, marker, export and URL the learn-mode overlay
// drives, hides or keys to is listed here. A LearnHarness ref that adds, renames or drops one fails a test naming it, so
// a new upstream feature gets a shell path (or is consciously left visible upstream) before the bump ships, never lost
// silently. Runs in the check checkout with the other overlay tests (`node --test custom/*.test.mjs`).
//
// Spike checks (`.docs/integration-learnharness.md` "Upgrade guards"): 2, 3, 5 and 7 are below. Check 6 (type and radius
// rungs) is look.test.mjs's "every size and radius class upstream uses". Checks 1 and 4 (inputs and composer-like call
// sites) retired: upstream's own composer sends in the frame. The launcher's marker list (run-learnharness.sh) refuses a
// ref before it is built.
import assert from "node:assert/strict";
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { join, posix, relative } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { createJiti } from "jiti";
import ts from "typescript";

/** The LearnHarness checkout this overlay lands in as `custom/`. */
const PROJECT_ROOT = fileURLToPath(new URL("..", import.meta.url));
const OVERLAY = "custom";
/** Upstream's app code: what the frame serves. electron/, scripts/ and tests/ never reach the browser. */
const UPSTREAM_DIRS = ["app", "components", "lib", "src"];
const SOURCE = /\.(tsx?|css)$/;

const jiti = createJiti(import.meta.url, { tsconfigPaths: true, jsx: true });
// jiti compiles JSX for the classic runtime.
globalThis.React = await import("react");
const { listOverriddenModules, overrideAliases } = await jiti.import("./next.config.ts");
const { SESSION_TITLE_CONTENT, SESSIONS_SIDEBAR_CONTENT } = await jiti.import("./embed/chrome.tsx");
const { KEYBOARD_ACTIONS } = await jiti.import("@/src/core");
const { LEARN_CHROME_KEYS } = await jiti.import("@gigachad/chat-ui/learn-protocol");

const read = (path) => readFileSync(join(PROJECT_ROOT, path), "utf8");

/** Project-relative paths of the source files under *dirs*. */
function filesUnder(dirs, pattern = SOURCE) {
  return dirs.flatMap((dir) =>
    readdirSync(join(PROJECT_ROOT, dir), { recursive: true, encoding: "utf8" })
      .filter((file) => pattern.test(file) && !file.includes("node_modules"))
      .map((file) => posix.join(dir, file.split("\\").join("/"))),
  );
}

const upstreamFiles = filesUnder(UPSTREAM_DIRS, /\.tsx?$/);
/** The overlay's own code and CSS, tests excluded: what ships in the frame. */
const overlayFiles = filesUnder([OVERLAY]).filter((file) => !file.endsWith(".test.mjs"));

/** *source* without comments: a name in a comment is neither a control nor a use. Keeps `//` after a `:` (URLs). */
function code(source) {
  return source.replace(/\/\*[\s\S]*?\*\//g, "").replace(/(^|[^:"'`\\])\/\/.*$/gm, "$1");
}

const upstreamCode = new Map(upstreamFiles.map((file) => [file, code(read(file))]));
const overlayCode = new Map(overlayFiles.map((file) => [file, code(read(file))]));

/** Each match of *pattern* (global, capture 1) in each file, as `file: value` lines for readable failures. */
function matchesIn(files, pattern) {
  return [...files].flatMap(([file, source]) => [...source.matchAll(pattern)].map((match) => [file, match[1]]));
}

/** The module a request names, project-relative without extension, or null for a package. */
function resolveRequest(fromFile, request) {
  let path;
  if (request.startsWith("@/")) path = request.slice(2);
  else if (request.startsWith(".")) path = posix.normalize(posix.join(posix.dirname(fromFile), request));
  else return null;
  return path.replace(/\.(tsx?|jsx?)$/, "");
}

/** The file of a project-relative module path. */
function moduleFile(module) {
  const file = ["", ".tsx", ".ts", "/index.ts", "/index.tsx"]
    .map((suffix) => module + suffix)
    .find((candidate) => existsSync(join(PROJECT_ROOT, candidate)) && statSync(join(PROJECT_ROOT, candidate)).isFile());
  assert.ok(file, `no file for ${module}`);
  return file;
}

/** Every module request (`import … from`, `export … from`, `import(…)`, side-effect `import`) in *source*. */
const REQUEST = /(?:\bfrom|\bimport)\s*\(?\s*["']([^"']+)["']/g;

// ─── Check 2: overrides keep upstream whole, and every upstream request reaches them ──────────────────────────────────

const overridden = listOverriddenModules(PROJECT_ROOT);
const aliases = overrideAliases(overridden);

test("check 2: every override wraps upstream with `export *` and keeps every upstream export", async () => {
  assert.ok(overridden.length > 0);
  for (const module of overridden) {
    const overrideFile = moduleFile(`${OVERLAY}/overrides/${module}`);
    const stars = [...read(overrideFile).matchAll(/^export \* from "([^"]+)";$/gm)].map((match) => resolveRequest(overrideFile, match[1]));
    assert.ok(stars.includes(module), `${overrideFile} lacks \`export * from\` its upstream ${module}`);
    const upstream = await jiti.import(`../${moduleFile(module)}`);
    const override = await jiti.import(`./${relative(OVERLAY, overrideFile)}`);
    assert.deepEqual(Object.keys(upstream).filter((name) => !(name in override)), [], module);
  }
  // The markdown wrapper re-defines ToolContentResponse: upstream's would call its own MessageResponse past the wrapper.
  const upstreamMessage = await jiti.import("../components/ai-elements/message.tsx");
  const message = await jiti.import("./overrides/components/ai-elements/message.tsx");
  assert.notEqual(message.ToolContentResponse, upstreamMessage.ToolContentResponse);
});

test("check 2: every upstream request for an overridden module is one the aliases capture, a bare `./X` key captures nothing else", () => {
  // The overlay reaches upstream's module by a relative path on purpose (that is how an override wraps it), so only
  // the bare-key rule applies to it.
  const sources = [...upstreamCode, ...overlayCode].filter(([file]) => !file.startsWith(`${OVERLAY}/overrides/`));
  for (const [file, request] of matchesIn(sources, REQUEST)) {
    const module = resolveRequest(file, request);
    if (!file.startsWith(`${OVERLAY}/`) && overridden.includes(module)) {
      assert.ok(Object.hasOwn(aliases, request), `${file} imports ${module} as "${request}", which no alias captures: it would bypass the override`);
    }
    if (request.startsWith("./") && Object.hasOwn(aliases, request)) {
      assert.equal(`./${OVERLAY}/overrides/${module}`, aliases[request], `${file}: "${request}" names another module but the alias captures it`);
    }
  }
});

// ─── Check 3: keyboard ───────────────────────────────────────────────────────────────────────────────────────────────

/**
 * What each upstream shortcut (`KeyAction`) does in the frame, where upstream's own keyboard handler runs.
 *
 * - `frame`: upstream handles it in place, on its visible UI
 * - `chrome-key`: its target is the hidden chrome, the frame swallows it in drive mode and the shell's chrome acts (`chrome.tsx`)
 * - `model-open`: the hidden model menu, the freeze swallows it and the shell's selector opens (`freeze.ts`)
 * - `frozen`: the freeze swallows it on a session page (read-only thinking level), upstream's otherwise
 */
const KEY_ACTIONS = {
  find: "frame",
  "toggle-user-thinking": "frame",
  "toggle-dictation": "frame",
  "new-session": "frame",
  "open-settings": "frame",
  "jump-graph": "frame",
  "cheat-sheet": "frame",
  "question-focus": "frame",
  "question-next-row": "frame",
  "question-prev-row": "frame",
  "question-pick-row": "frame",
  "question-submit": "frame",
  "toggle-sidebar": "chrome-key",
  "sessions-search": "chrome-key",
  "rename-session": "chrome-key",
  "open-model-picker": "model-open",
  "cycle-thinking": "frozen",
};

test("check 3: every upstream shortcut is classified, the hidden chrome's ones are the shell's chrome keys", () => {
  assert.deepEqual(Object.keys(KEY_ACTIONS).sort(), KEYBOARD_ACTIONS.map((action) => action.id).sort(), "classify the new shortcut in KEY_ACTIONS");
  const chromeKeys = Object.keys(KEY_ACTIONS).filter((action) => KEY_ACTIONS[action] === "chrome-key");
  assert.deepEqual(chromeKeys.sort(), [...LEARN_CHROME_KEYS].sort());
});

// ─── Check 5: lh: names ──────────────────────────────────────────────────────────────────────────────────────────────

/** Every `lh:` name upstream's app code spells, and what the learn mode does with it. */
const LH_NAMES = {
  "lh:chat-renamed": "event: the meta poster re-reads titles (session-meta)",
  "lh:graph-revision": "event: the meta poster dispatches it when a session is deleted, like upstream's onDeleted",
  "lh:turn-state": "event: the meta poster refreshes row states",
  "lh:settings-saved": "event: the model bridge re-reads the tiers (notifySettingsSaved)",
  "lh:insert-file-paths": "event: PathInput's desktop drop, absent in the browser",
  "lh:menu-open": "event: MenuShell bookkeeping inside the frame",
  "lh:menu-close": "event: MenuShell bookkeeping inside the frame",
  "lh:kbd-abort-turn": "shortcut: upstream's Esc in the frame",
  "lh:kbd-cancel-dictation": "shortcut: upstream's dictation in the frame",
  "lh:kbd-cycle-fields": "shortcut: Tab between dock fields, frame-only",
  "lh:kbd-cycle-thinking": "shortcut: ⇧Tab in the frame, the freeze swallows it on a session",
  "lh:kbd-find": "shortcut: find in the frame",
  "lh:kbd-open-model-picker": "shortcut: ⌘K in the frame, the freeze turns it into model-open",
  "lh:kbd-rename-session": "shortcut: swallowed in drive mode, posted as chrome-key",
  "lh:kbd-sessions-search": "shortcut: swallowed in drive mode, posted as chrome-key",
  "lh:kbd-toggle-dictation": "shortcut: upstream's dictation in the frame",
  "lh:kbd-": "kbdEventForAction's prefix: every KeyAction, classified in KEY_ACTIONS",
  "lh:keymap": "storage: the frame's keymap, the shell keeps its own chords",
  "lh:sidebar-open": "storage: hidden sidebar's open state, the swallow keeps it unchanged",
  "lh:session-kind-filter": "storage: hidden sidebar's kind filter, the shell stores its own",
  "lh:chatgpt": "server ↔ desktop subscription channel, keys and subscriptions are gated off",
  "lh:subscriptions": "server ↔ desktop subscription channel, keys and subscriptions are gated off",
};

const LH_NAME = /["'`](lh:[a-z0-9-]*)/g;

test("check 5: every lh: event, shortcut and storage name upstream uses is classified, the overlay uses only those", () => {
  const upstream = new Set(matchesIn(upstreamCode, LH_NAME).map(([, name]) => name));
  assert.deepEqual([...upstream].sort(), Object.keys(LH_NAMES).sort(), "a new or removed lh: name: classify it in LH_NAMES");
  const kbd = new Set(KEYBOARD_ACTIONS.map((action) => `lh:kbd-${action.id}`));
  for (const [file, name] of matchesIn(overlayCode, LH_NAME)) {
    assert.ok(upstream.has(name) || kbd.has(name), `${file} uses ${name}, which upstream no longer has`);
  }
});

// ─── Check 7: the URLs the overlay calls ─────────────────────────────────────────────────────────────────────────────

const ROUTES = filesUnder(["app/api"], /(^|\/)route\.ts$/).map((file) => file.replace(/^app/, "").replace(/\/route\.ts$/, "").split("/"));

test("check 7: every upstream API path the overlay calls is an upstream route", () => {
  const urls = matchesIn(overlayCode, /["'`](\/api\/[^"'`?]*)/g);
  assert.ok(urls.length > 0);
  for (const [file, url] of urls) {
    const segments = url.replace(/\$\{[^}]*\}/g, "*").replace(/\/$/, "").split("/");
    const served = ROUTES.some(
      (route) =>
        (route.length === segments.length || route.at(-1)?.startsWith("[...")) &&
        segments.every((segment, i) => route[i] !== undefined && (route[i] === segment || route[i].startsWith("[")) && (segment !== "*" || route[i].startsWith("["))),
    );
    assert.ok(served, `${file} calls ${url}, which no upstream route serves`);
  }
});

// ─── Hidden chrome: props, controls, markers, exports ────────────────────────────────────────────────────────────────

/** Prop names of an upstream component's first parameter, from the type checker: a new prop is a new feature. */
function propsOf(file, name, program) {
  const checker = program.getTypeChecker();
  const module = checker.getSymbolAtLocation(program.getSourceFile(join(PROJECT_ROOT, file)));
  const component = checker.getExportsOfModule(module).find((symbol) => symbol.name === name);
  assert.ok(component, `${file} no longer exports ${name}`);
  const [signature] = checker.getSignaturesOfType(checker.getTypeOfSymbol(component), ts.SignatureKind.Call);
  return checker.getPropertiesOfType(checker.getTypeOfSymbol(signature.parameters[0])).map((prop) => prop.name);
}

/** What the top-bar override does with each `TopBar` prop. */
const TOP_BAR_PROPS = {
  drag: "passed through",
  onMouseDown: "passed through",
  children: "the session title bar's content: title, rename, Plan, coverage, phase reach the shell (session-meta, open-panel)",
};

test("every TopBar prop is mapped", () => {
  const file = "components/top-bar.tsx";
  // The props are inline object types: names resolve without loading the import graph. A props type built from an
  // imported one would come out short (or empty) here, which fails the test rather than passing it.
  const program = ts.createProgram(
    [join(PROJECT_ROOT, file)],
    { noResolve: true, noLib: true, noEmit: true, jsx: ts.JsxEmit.Preserve, strict: true },
  );
  assert.deepEqual(propsOf(file, "TopBar", program).sort(), Object.keys(TOP_BAR_PROPS).sort(), "map the new TopBar prop");
});

/**
 * The upstream code drive mode hides, and where its controls live now. Mirrors the feature table in
 * `.docs/integration-learnharness.md` (ChromeParity's). `region` cuts a file to the part that is hidden.
 */
const HIDDEN_CHROME = {
  "components/sidebar.tsx": {
    shape: { "<button": 8, "<input": 1, "<a": 0, "<Button": 2, "<Input": 1, onClick: 10, onContextMenu: 2, onDoubleClick: 0, onKeyDown: 2 },
    controls: {
      "aria:Open sidebar": "⌘B → chrome-key → shell sidebar toggle; the reopen button is hidden and inert",
      "hint:Open sidebar": "as aria:Open sidebar",
      "aria:Close sidebar": "as aria:Open sidebar",
      "hint:Close sidebar": "as aria:Open sidebar",
      "aria:Home": "learn header action \"Open knowledge graph\"",
      "hint:Home": "as aria:Home",
      "jsx:Link": "the Home link, as aria:Home",
      "jsx:LhMark": "the Home link's logo",
      "testid:new-session": "learn header action \"New teaching session\" (new-session teach); content marker of the sidebar",
      "testid:new-session-practice": "learn header action \"New practice session\" (new-session practice)",
      "aria:Search sessions": "learn filter input, upstream's fuzzy ranking, ↑/↓, Enter; ⌘⇧F → chrome-key",
      "hint:Search sessions": "as aria:Search sessions",
      "testid:session-search": "as aria:Search sessions",
      "jsx:Input": "the search field, as aria:Search sessions",
      "testid:session-kind-filter": "Teaching / Practice / All PillTabs, persisted",
      "testid:kind-filter-${f}": "as testid:session-kind-filter",
      "jsx:SessionRow": "learn rows in gigachad's sidebar",
      'testid:kbdSelected ? "session-row-kbd" : "session-row"': "learn rows; the filter's ↑/↓ highlight",
      "jsx:SessionStatus": "row chip: working / waiting / problem waiting / compacting",
      "jsx:BrailleSpinner": "row chip \"working\"",
      "aria:Practice session": "row chip: practice tag",
      "jsx:RenameRow": "row rename (pencil, double-click, F2, chrome-key) → PATCH /api/learn/sessions/{id}",
      "testid:session-rename-row": "as jsx:RenameRow",
      "testid:session-rename-input": "as jsx:RenameRow",
      "aria:Rename session": "as jsx:RenameRow",
      "jsx:MenuShell": "the row context menu → the row's \"More actions\" strip",
      "testid:ctx-rename": "as jsx:RenameRow",
      "testid:ctx-export-md": "row menu \"Export as Markdown\" → GET /api/learn/sessions/{id}/markdown",
      "testid:ctx-delete": "row trash (preserve graph) and row menu \"Delete with graph traces\" (delete-nodes), with undo",
      "jsx:DeleteSessionDialog": "as testid:ctx-delete",
      "jsx:CoverageMenuBlock": "row menu \"Coverage\" → open-panel coverage → upstream's CoverageMenuBlock (Extract now)",
      "jsx:ActivitySection": "\"background agents\" folder; its action → open-panel activity → upstream's ActivitySection",
      "hint:Settings": "learn header action → open-settings",
      "testid:settings-entry": "as hint:Settings",
      "jsx:Button": "the primitive of the buttons above, each named by its own testid or label",
      "jsx:Hint": "tooltips of the controls above",
      "jsx:SessionsSidebar": "the hidden sidebar itself",
      "jsx:SettingsDialogProvider": "AppShell frame, not hidden: the settings dialog open-settings reaches",
      "jsx:FilePathDropZone": "AppShell frame, not hidden",
      "jsx:KeyboardShortcuts": "AppShell frame, not hidden: upstream's shortcuts, classified in KEY_ACTIONS",
      "jsx:SidebarContext": "AppShell frame, not hidden: the sidebar's open state (SidebarProvider)",
    },
  },
  "components/top-bar.tsx": {
    shape: { "<button": 0, "<input": 0, "<a": 0, "<Button": 0, "<Input": 0, onClick: 0, onContextMenu: 0, onDoubleClick: 0, onKeyDown: 0 },
    controls: { "jsx:SidebarToggle": "as components/sidebar.tsx aria:Open sidebar" },
  },
  "components/chat.tsx": {
    region: ["<TopBar", "</TopBar>"],
    shape: { "<button": 2, "<input": 1, "<a": 0, "<Button": 0, "<Input": 0, onClick: 2, onContextMenu: 0, onDoubleClick: 0, onKeyDown: 1 },
    controls: {
      "jsx:TopBar": "the hidden session title bar; its title is the shell tab and row title (session-meta)",
      "hint:Rename session": "row rename, F2",
      "aria:Session title": "row rename, F2",
      "hint:Open the full plan": "row menu \"Plan\" → open-panel plan → upstream's PlanPopup",
      "testid:plan-chip": "as hint:Open the full plan; session-plan tells the shell it exists",
      "jsx:CoverageRing": "row menu \"Coverage\" → open-panel coverage → upstream's CoverageRingDetail",
      "testid:phase-chip": "row chip (phase or practice tag); content marker of the title bar",
      "jsx:PhaseChip": "as testid:phase-chip",
      "jsx:Hint": "tooltips of the controls above",
    },
  },
};

/** What a control is called in *source*: test id, aria label, Hint label, or the component it renders (icons aside). */
function controlsOf(source) {
  const found = new Set();
  for (const match of source.matchAll(/data-testid=(?:"([^"]+)"|\{`([^`]+)`\}|\{([^}]+)\})/g)) found.add(`testid:${match[1] ?? match[2] ?? match[3]}`);
  for (const match of source.matchAll(/aria-label=(?:"([^"]+)"|\{([^}]+)\})/g)) found.add(`aria:${match[1] ?? match[2]}`);
  for (const match of source.matchAll(/<Hint\b[^>]*?\blabel=(?:"([^"]+)"|\{([^}]+)\})/g)) found.add(`hint:${match[1] ?? match[2]}`);
  for (const match of source.matchAll(/(?<![\w.$])<([A-Z]\w*)\b/g)) if (!match[1].endsWith("Icon")) found.add(`jsx:${match[1]}`);
  return found;
}

/** How many interactive elements and handlers *source* has: catches a new control that carries no name. */
function shapeOf(source, keys) {
  return Object.fromEntries(keys.map((key) => [key, source.split(key.startsWith("<") ? new RegExp(`${key}\\b`) : new RegExp(`\\b${key}=`)).length - 1]));
}

/** *file*'s code, cut to its hidden region. */
function hiddenSource(file, region) {
  const source = upstreamCode.get(file);
  if (!region) return source;
  const start = source.indexOf(region[0]);
  const end = source.indexOf(region[1], start);
  assert.ok(start >= 0 && end > start, `${file}: hidden region ${region.join("…")} not found`);
  return source.slice(start, end);
}

test("every control of the hidden sidebar and title bar has a shell path", () => {
  for (const [file, { region, shape, controls }] of Object.entries(HIDDEN_CHROME)) {
    const source = hiddenSource(file, region);
    const found = controlsOf(source);
    const unmapped = [...found].filter((control) => !Object.hasOwn(controls, control));
    assert.deepEqual(unmapped, [], `${file}: give each new control a shell path (or a frame action), then map it here`);
    const gone = Object.keys(controls).filter((control) => !found.has(control));
    assert.deepEqual(gone, [], `${file}: these mapped controls are gone upstream, check the shell path and drop them`);
    assert.deepEqual(shapeOf(source, Object.keys(shape)), shape, `${file}: an unnamed control appeared or went: find it, map it, update the shape`);
  }
});

/** The test id in a `[data-testid="…"]` selector constant. */
const testIdOf = (selector) => /^\[data-testid="([^"]+)"\]$/.exec(selector)?.[1];

test("the content markers the overlay keys hiding and panels to are where it expects them", () => {
  const titleBar = hiddenSource("components/chat.tsx", HIDDEN_CHROME["components/chat.tsx"].region);
  const markers = [
    [upstreamCode.get("components/sidebar.tsx"), testIdOf(SESSIONS_SIDEBAR_CONTENT), "the sessions sidebar (chrome.tsx SESSIONS_SIDEBAR_CONTENT)"],
    [titleBar, testIdOf(SESSION_TITLE_CONTENT), "the session title bar (chrome.tsx SESSION_TITLE_CONTENT)"],
    [titleBar, "plan-chip", "the Plan popup (open-panel plan)"],
  ];
  for (const [source, id, what] of markers) {
    assert.ok(id && source.includes(`data-testid="${id}"`), `${what}: upstream lacks data-testid="${id}"`);
  }
  const settings = upstreamCode.get("components/settings/settings-dialog.tsx");
  assert.ok(settings.includes('data-testid="settings-nav"') && settings.includes("aria-current"), "open-settings' section guard reads settings-nav's aria-current");
});

test("every test id and data attribute the overlay selects exists upstream", () => {
  const all = [...upstreamCode.values()].join("\n");
  const templates = [...all.matchAll(/data-testid=\{`([^`]+)`\}/g)].map(
    (match) => new RegExp(`^${match[1].split(/\$\{[^}]*\}/).map((part) => part.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join(".+")}$`),
  );
  // What the overlay renders itself is a JSX attribute (`<div data-x=…>`), never inside a selector (`[data-x=…]`).
  const ownIds = new Set(matchesIn(overlayCode, /(?<!\[)data-testid="([^"]+)"/g).filter(([file]) => file.endsWith(".tsx")).map(([, id]) => id));
  for (const [file, id] of matchesIn(overlayCode, /\[data-testid=\\?"([^"\\]+)\\?"\]/g)) {
    if (ownIds.has(id)) continue;
    assert.ok(all.includes(`"${id}"`) || templates.some((pattern) => pattern.test(id)), `${file} selects data-testid="${id}", which upstream no longer renders`);
  }
  const ownAttributes = new Set(matchesIn(overlayCode, /(?<![[\w-])(data-[a-z-]+)[=\s>]/g).filter(([file]) => file.endsWith(".tsx")).map(([, name]) => name));
  const streamdown = readdirSync(join(PROJECT_ROOT, "node_modules/streamdown/dist"))
    .filter((file) => file.endsWith(".js"))
    .map((file) => readFileSync(join(PROJECT_ROOT, "node_modules/streamdown/dist", file), "utf8"))
    .join("\n");
  for (const [file, name] of matchesIn(overlayCode, /\[(data-(?!testid\b|cf-)[a-z-]+)/g)) {
    if (ownAttributes.has(name)) continue;
    assert.ok(all.includes(name) || streamdown.includes(name), `${file} selects ${name}, which upstream no longer renders`);
  }
});

/** Upstream names ChromeParity's panels and meta poster render or call: listed so a reader sees why they are guarded. */
const CHROME_EXPORTS = {
  "@/components/activity-dock": ["ActivitySection", "toBlobs", "useActivity"],
  "@/components/extraction-coverage": ["CoverageMenuBlock", "CoverageRingDetail", "fetchGraphRead"],
  "@/components/popover-dismiss": ["usePopoverDismiss"],
  "@/components/settings/settings-dialog.tsx": ["useSettingsDialog"],
  "@/components/server-events": ["useServerEvent", "useServerResync"],
};

test("every upstream value the overlay imports is still exported, ChromeParity's included", async () => {
  const imports = new Map();
  for (const [file, source] of overlayCode) {
    for (const match of source.matchAll(/import\s+(?:type\s+)?\{([^}]*)\}\s+from\s+"([^"]+)"/g)) {
      if (match[0].startsWith("import type")) continue;
      const module = resolveRequest(file, match[2]);
      if (module === null || module.startsWith(`${OVERLAY}/`)) continue;
      const names = match[1].split(",").map((name) => name.trim()).filter((name) => name && !name.startsWith("type "));
      imports.set(module, new Set([...(imports.get(module) ?? []), ...names.map((name) => name.split(/\s+as\s+/)[0])]));
    }
  }
  for (const [request, names] of Object.entries(CHROME_EXPORTS)) {
    const module = resolveRequest(`${OVERLAY}/embed/chrome.tsx`, request);
    for (const name of names) assert.ok(imports.get(module)?.has(name), `the overlay no longer imports ${name} from ${module}: update CHROME_EXPORTS`);
  }
  for (const [module, names] of imports) {
    const exported = await jiti.import(`../${moduleFile(module)}`);
    for (const name of names) assert.ok(name in exported, `${module} no longer exports ${name}`);
  }
});
